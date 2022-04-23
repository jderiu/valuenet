import os.path

from transformers.trainer_seq2seq import Trainer
from transformers.training_args_seq2seq import TrainingArguments

from transformers import AutoTokenizer, pipeline
from src.model.encoder_decoder.modeling_encoder_decoder import EncoderDecoderModel
from transformers.models.gpt2.modeling_gpt2 import GPT2LMHeadModel
from transformers import SchedulerType
import numpy as np
from datasets import load_metric
from src.config import read_arguments_train, write_config_to_file
from src.intermediate_representation import semQL
from src.spider import spider_utils
from src.utils import setup_device, set_seed_everywhere, create_experiment_folder
from src.model.sql2text_data import DataCollatorForSQL2Text, DataCollartorForLMSQL2Text

metric = load_metric("sacrebleu")
# initialize experiment tracking @ Weights & Biases
import wandb
wandb.init(project="sql2text", name="sql2text_train")


def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [[label.strip()] for label in labels]

    return preds, labels

n_output = 1

def compute_metrics(eval_preds):
    global n_output
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    decoded_preds = decoder_tokenizer.batch_decode(preds.argmax(axis=2), skip_special_tokens=True)
    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, decoder_tokenizer.pad_token_id)
    decoded_labels = decoder_tokenizer.batch_decode(labels, skip_special_tokens=True)

    result = metric.compute(predictions=decoded_preds, references=decoded_labels)
    result = {"bleu": result["score"]}

    prediction_lens = [np.count_nonzero(pred != decoder_tokenizer.pad_token_id) for pred in preds]
    result["gen_len"] = np.mean(prediction_lens)
    result = {k: round(v, 4) for k, v in result.items()}
    out_n = int(n_output*eval_steps)
    with open(os.path.join(output_path, f'results_{out_n}.txt'), 'wt', encoding='utf-8') as f:
        f.write(f'BLEU: {result["bleu"]}\n')
        for pred, label in zip(decoded_preds, decoded_labels):
            f.write(f"{pred}\t{label[0]}\n")
    n_output += 1
    return result

def compute_metrics_decode_only(eval_preds):
    global n_output
    generator = pipeline('text-generation', model=model, tokenizer=decoder_tokenizer, device=0)
    preds, labels = eval_preds
    #decoded_preds = decoder_tokenizer.batch_decode(preds.argmax(axis=2), skip_special_tokens=True)
    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, decoder_tokenizer.pad_token_id)
    decoded_labels = decoder_tokenizer.batch_decode(labels, skip_special_tokens=True)
    out_labels, out_preds = [], []
    for decoded_label in decoded_labels:
        prefix = decoded_label.split('TEXT:')[0]
        label = decoded_label.split('TEXT:')[1]
        ret = generator(
            prefix + 'TEXT:',
            max_length=100,
            num_return_sequences=1,
            pad_token_id=decoder_tokenizer.eos_token_id
        )[0]['generated_text']
        pred = ret.split('TEXT:')[1].replace('\n', '')
        out_labels.append(label)
        out_preds.append(pred)
    # Some simple post-processing
    decoded_preds, decoded_labels = postprocess_text(out_preds, out_labels)

    result = metric.compute(predictions=decoded_preds, references=decoded_labels)
    result = {"bleu": result["score"]}

    prediction_lens = [np.count_nonzero(pred != decoder_tokenizer.pad_token_id) for pred in preds]
    result["gen_len"] = np.mean(prediction_lens)
    result = {k: round(v, 4) for k, v in result.items()}
    out_n = int(n_output*eval_steps)
    with open(os.path.join(output_path, f'results_{out_n}.txt'), 'wt', encoding='utf-8') as f:
        f.write(f'BLEU: {result["bleu"]}\n')
        for pred, label in zip(decoded_preds, decoded_labels):
            f.write(f"{pred}\t{label[0]}\n")
    n_output += 1
    return result


if __name__ == '__main__':
    args = read_arguments_train()

    # log hyperparameters to Weights & Biases
    wandb.config.update(args)

    experiment_name, output_path = create_experiment_folder(args.model_output_dir, args.exp_name)
    print("Run experiment '{}'".format(experiment_name))

    write_config_to_file(args, output_path)

    device, n_gpu = setup_device()
    set_seed_everywhere(args.seed, n_gpu)

    sql_data, table_data, val_sql_data, val_table_data = spider_utils.load_dataset(args.data_dir, use_small=args.toy)

    grammar = semQL.Grammar()

    if args.gen_type == 'encoder_decoder':
        encoder_tokenizer = AutoTokenizer.from_pretrained(args.encoder_pretrained_model, add_prefix_space=True)
        decoder_tokenizer = AutoTokenizer.from_pretrained(args.decoder_pretrained_model)
        if decoder_tokenizer.pad_token_id is None:
            decoder_tokenizer.pad_token = decoder_tokenizer.bos_token
        model = EncoderDecoderModel.from_encoder_decoder_pretrained(
            args.encoder_pretrained_model, args.decoder_pretrained_model
        )
        model.to(device)

        data_collator = DataCollatorForSQL2Text(
            encoder_tokenizer=encoder_tokenizer,
            decoder_tokenizer=decoder_tokenizer,
            model=model,
            grammar=grammar,
            schema=table_data,
            device=device
        )
    else:
        decoder_tokenizer = AutoTokenizer.from_pretrained(args.decoder_pretrained_model, add_prefix_space=True)
        if decoder_tokenizer.pad_token_id is None:
            decoder_tokenizer.pad_token = decoder_tokenizer.bos_token
        if decoder_tokenizer.sep_token_id is None:
            decoder_tokenizer.sep_token = decoder_tokenizer.bos_token
        model = GPT2LMHeadModel.from_pretrained(args.decoder_pretrained_model)
        model.to(device)
        data_collator = DataCollartorForLMSQL2Text(
            tokenizer=decoder_tokenizer,
            model=model,
            grammar=grammar,
            schema=table_data,
            device=device
        )

    pytorch_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    perc = 100 * (pytorch_trainable_params / pytorch_total_params)
    print(f'Training {pytorch_trainable_params} out of {pytorch_total_params} parameters ({perc})!')

    # track the model
    #wandb.watch(model, log='parameters')
    eval_steps = 1
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
        label_smoothing_factor=0.1
    )

    trainer = Trainer(
        model=model,
        args=train_args,
        data_collator=data_collator,
        train_dataset=sql_data,
        eval_dataset=val_sql_data,
        compute_metrics=compute_metrics_decode_only
    )

    trainer.train(ignore_keys_for_eval=['logits', 'past_key_values', 'encoder_last_hidden_state', 'hidden_states', 'cross_attentions'])
