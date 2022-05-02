import torch, os
import numpy as np
from tqdm import tqdm
from datasets import load_metric

from transformers.models.gpt2.modeling_gpt2 import GPT2LMHeadModel
from transformers import AutoTokenizer
from transformers.trainer_seq2seq import Trainer
from transformers.training_args_seq2seq import TrainingArguments
from transformers import SchedulerType

from src.model.model import IRNet
from src.data_loader import get_data_loader
from src.config import write_config_to_file, read_arguments_pretrain
from src.intermediate_representation import semQL
from src.spider import spider_utils
from src.optimizer import build_optimizer_encoder
from src.utils import setup_device, set_seed_everywhere, create_experiment_folder
from src.unsupervised_semparse.pretraining.pretrain_data_collator import DataCollatorSQL2SQL, DataCollatorText2Text
from src.unsupervised_semparse.pretraining.training_loops import pretrain_loop

metric = load_metric("sacrebleu")
# initialize experiment tracking @ Weights & Biases
import wandb


def batch_list(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]


def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [[label.strip()] for label in labels]

    return preds, labels

n_output = 1

def compute_metrics_decode_only(eval_preds):
    global n_output
    preds, labels = eval_preds
    model.eval()
    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, decoder_tokenizer.pad_token_id)
    decoded_labels = decoder_tokenizer.batch_decode(labels, skip_special_tokens=True)
    out_labels, out_preds = [], []

    n_decoding_steps = int(len(decoded_labels) / args.eval_batch_size) + 1
    for decoded_label_batch in tqdm(batch_list(decoded_labels, n=args.eval_batch_size), desc="Decoding:", total=n_decoding_steps):
        prefix_batch = [x.split('TEXT:')[0] + 'TEXT:' for x in decoded_label_batch]
        label_batch = [x.split('TEXT:')[1] for x in decoded_label_batch]
        prefix_batch_enc = decoder_tokenizer(prefix_batch, return_tensors='pt', padding=True)
        decoded_out = model.generate(
            prefix_batch_enc['input_ids'].to(device),
            max_length=labels.shape[1],
            pad_token_id=decoder_tokenizer.eos_token_id
        )
        pred_batch = decoder_tokenizer.batch_decode(decoded_out, skip_special_tokens=True)
        pred_batch_out = [x.split('TEXT:')[1].replace('\n', '') for x in pred_batch]

        out_labels.extend(label_batch)
        out_preds.extend(pred_batch_out)
    # Some simple post-processing
    decoded_preds, decoded_labels = postprocess_text(out_preds, out_labels)

    result = metric.compute(predictions=decoded_preds, references=decoded_labels)
    result = {"bleu": result["score"]}

    prediction_lens = [np.count_nonzero(pred != decoder_tokenizer.pad_token_id) for pred in preds]
    result["gen_len"] = np.mean(prediction_lens)
    result = {k: round(v, 4) for k, v in result.items()}
    out_n = int(n_output * eval_steps)
    with open(os.path.join(output_path, f'results_{out_n}.txt'), 'wt', encoding='utf-8') as f:
        f.write(f'BLEU: {result["bleu"]}\n')
        for pred, label in zip(decoded_preds, decoded_labels):
            f.write(f"{pred}\t{label[0]}\n")
    n_output += 1
    return result


if __name__ == '__main__':
    args = read_arguments_pretrain()

    # log hyperparameters to Weights & Biases
    experiment_name, output_path = create_experiment_folder(args.model_output_dir, args.exp_name)
    print("Run experiment '{}'".format(experiment_name))

    wandb.init(
        project="sql2text",
        group="pretraining",
        name=experiment_name
    )

    wandb.config.update(args)

    write_config_to_file(args, output_path)

    device, n_gpu = setup_device()
    set_seed_everywhere(args.seed, n_gpu)

    sql_data, table_data, val_sql_data, val_table_data = spider_utils.load_dataset(args.data_dir, use_small=args.toy)

    grammar = semQL.Grammar()

    if args.pretrain_type == 'sql2sql':
        train_loader, dev_loader = get_data_loader(sql_data, val_sql_data, args.batch_size, True, False)
        data_collator = DataCollatorSQL2SQL(
            grammar=grammar,
            schema=table_data,
            device=device
        )

        model = IRNet(args, device, grammar)
        model.to(device)

        num_train_steps = len(sql_data) * args.num_epochs
        optimizer, scheduler = build_optimizer_encoder(
            model,
            num_train_steps,
            args.lr_transformer, args.lr_connection, args.lr_base,
            args.scheduler_gamma
        )

        pretrain_loop(
            args,
            train_loader,
            dev_loader,
            data_collator,
            model,
            optimizer,
            scheduler,
            output_path
        )
    else:
        ignore_keys_for_eval = ['past_key_values', 'encoder_last_hidden_state', 'hidden_states', 'cross_attentions']
        ignore_keys_for_eval.append('logits')
        metrics_fn = compute_metrics_decode_only
        decoder_tokenizer = AutoTokenizer.from_pretrained(args.decoder_pretrained_model, add_prefix_space=True)
        decoder_tokenizer.padding_side = 'left'
        if decoder_tokenizer.pad_token_id is None:
            decoder_tokenizer.pad_token = decoder_tokenizer.bos_token
        if decoder_tokenizer.sep_token_id is None:
            decoder_tokenizer.sep_token = decoder_tokenizer.bos_token
        model = GPT2LMHeadModel.from_pretrained(args.decoder_pretrained_model)
        model.to(device)
        eval_steps = 1
        data_collator = DataCollatorText2Text(
            tokenizer=decoder_tokenizer,
            model=model,
            grammar=grammar,
            schema=table_data,
            device=device
        )
        nocuda = not str(device) == 'cuda'
        train_args = TrainingArguments(
            output_dir=output_path,
            logging_dir=output_path,
            overwrite_output_dir=True,
            do_train=True,
            do_eval=True,
            save_total_limit=2,
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            num_train_epochs=args.num_epochs,
            evaluation_strategy="epoch",
            eval_accumulation_steps=args.batch_size,
            no_cuda=nocuda,
            fp16=True,
            save_strategy="epoch",
            ignore_data_skip=True,
            logging_steps=10,
            learning_rate=args.lr_transformer,
            warmup_steps=100,
            dataloader_pin_memory=False,
            lr_scheduler_type=SchedulerType.LINEAR,
            label_smoothing_factor=0.0
        )

        trainer = Trainer(
            model=model,
            args=train_args,
            data_collator=data_collator,
            train_dataset=sql_data,
            eval_dataset=val_sql_data,
            compute_metrics=metrics_fn
        )

        trainer.train(ignore_keys_for_eval=ignore_keys_for_eval)
