import os.path

from transformers.trainer_seq2seq import Trainer
from transformers.training_args_seq2seq import TrainingArguments

from transformers import AutoTokenizer
from transformers.models.encoder_decoder import EncoderDecoderModel
from transformers import SchedulerType
import numpy as np
from datasets import load_metric
from src.config import read_arguments_train, write_config_to_file
from src.data_loader import get_data_loader
from src.intermediate_representation import semQL
from src.spider import spider_utils
from src.utils import setup_device, set_seed_everywhere, create_experiment_folder
from src.model.sql2text_data import DataCollatorForSQL2Text

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

    # Some simple post-processing
    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

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
    encoder_tokenizer = AutoTokenizer.from_pretrained(args.encoder_pretrained_model, add_prefix_space=True)
    decoder_tokenizer = AutoTokenizer.from_pretrained(args.decoder_pretrained_model)
    model = EncoderDecoderModel.from_encoder_decoder_pretrained(
        args.encoder_pretrained_model, args.decoder_pretrained_model
    )
    model.to(device)

    pytorch_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    perc = 100 * (pytorch_trainable_params / pytorch_total_params)
    print(f'Training {pytorch_trainable_params} out of {pytorch_total_params} parameters ({perc})!')

    data_collator = DataCollatorForSQL2Text(
        encoder_tokenizer=encoder_tokenizer,
        decoder_tokenizer=decoder_tokenizer,
        model=model,
        grammar=grammar,
        schema=table_data,
        device=device
    )

    # track the model
    #wandb.watch(model, log='parameters')
    eval_steps = 1000
    nocuda = not device == 'cuda'
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
        fp16=True and not nocuda,
        save_strategy="epoch",
        ignore_data_skip=True,
        logging_steps=10,
        learning_rate=3e-5,
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
        compute_metrics=compute_metrics
    )

    trainer.train(ignore_keys_for_eval=['past_key_values', 'encoder_last_hidden_state'])
