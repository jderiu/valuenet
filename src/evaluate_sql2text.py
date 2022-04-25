import json
import os.path

import torch
from transformers.trainer_seq2seq import Trainer
from transformers.training_args_seq2seq import TrainingArguments

from transformers import AutoTokenizer, pipeline
from src.model.encoder_decoder.modeling_encoder_decoder import EncoderDecoderModel
from transformers.models.gpt2.modeling_gpt2 import GPT2LMHeadModel
from transformers import SchedulerType
import numpy as np
from datasets import load_metric
from src.config import read_arguments_evaluation, write_config_to_file
from src.intermediate_representation import semQL
from src.spider import spider_utils
from src.utils import setup_device, set_seed_everywhere, create_experiment_folder
from src.model.sql2text_data import DataCollatorForSQL2Text, DataCollartorForLMSQL2Text
from tqdm import tqdm
metric = load_metric("sacrebleu")

def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [[label.strip()] for label in labels]

    return preds, labels

def batch_list(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]


def evaulate_decode_only(
        args,
        test_data,
        data_collator,
        model,
        tokenizer,
        logging_path,
        checkpoint_nr=0

):
    out_labels, out_preds = [], []
    n_eval_steps = int(len(test_data)// args.batch_size) + 1
    for batch in tqdm(batch_list(test_data, args.batch_size), total=n_eval_steps):
        preprocessed_batch = data_collator(batch, is_eval=True)
        with torch.no_grad():
            generated_out = model.generate(
                preprocessed_batch['input_ids'],
                attention_mask=preprocessed_batch['attention_mask'],
                max_length = preprocessed_batch['input_ids'].shape[1] + 32,
                num_beams=15,
                repetition_penalty=2.5,
                no_repeat_ngram_size=3,
                pad_token_id=tokenizer.pad_token_id,
            )
        decoded_out = tokenizer.batch_decode(generated_out, skip_special_tokens=True)
        pred_batch_out = [x.split('TEXT:')[1].replace('\n', '') for x in decoded_out]
        labels = [x['question'] for x in batch]

        out_labels.extend(labels)
        out_preds.extend(pred_batch_out)
    decoded_preds, decoded_labels = postprocess_text(out_preds, out_labels)
    result = metric.compute(predictions=decoded_preds, references=decoded_labels)
    result = {"bleu": result["score"]}

    with open(os.path.join(logging_path, f'results_final_{checkpoint_nr}.txt'), 'wt', encoding='utf-8') as f:
        f.write(f'BLEU: {result["bleu"]}\n')
        for pred, label in zip(decoded_preds, decoded_labels):
            f.write(f"{pred}\t{label[0]}\n")

def main():
    args = read_arguments_evaluation()

    device, n_gpu = setup_device()
    set_seed_everywhere(args.seed, n_gpu)

    sql_data, table_data, val_sql_data, val_table_data = spider_utils.load_dataset(args.data_dir, use_small=False)
    grammar = semQL.Grammar()

    print("Loading pre-trained model from '{}'".format(args.model_to_load))
    with open(os.path.join(args.model_to_load, "args.json"), "rt", encoding='utf-8') as f:
        train_args = json.load(f)

    if train_args['gen_type'] == 'encoder_decoder':
        encoder_tokenizer = AutoTokenizer.from_pretrained(train_args['encoder_pretrained_model'], add_prefix_space=True)
        decoder_tokenizer = AutoTokenizer.from_pretrained(train_args['decoder_pretrained_model'])
        if decoder_tokenizer.pad_token_id is None:
            decoder_tokenizer.pad_token = decoder_tokenizer.bos_token
        #model = EncoderDecoderModel.from_pretrained(os.path.join(args.model_to_load, f'checkpoint-{args.checkpoint}'))
        model = EncoderDecoderModel.from_encoder_decoder_pretrained(
            train_args['encoder_pretrained_model'], train_args['decoder_pretrained_model']
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
        decoder_tokenizer = AutoTokenizer.from_pretrained(train_args['decoder_pretrained_model'], add_prefix_space=True)
        decoder_tokenizer.padding_side = 'left'
        if decoder_tokenizer.pad_token_id is None:
            decoder_tokenizer.pad_token = decoder_tokenizer.bos_token
        if decoder_tokenizer.sep_token_id is None:
            decoder_tokenizer.sep_token = decoder_tokenizer.bos_token
        #model = GPT2LMHeadModel.from_pretrained('gpt2')
        model = GPT2LMHeadModel.from_pretrained(os.path.join(args.model_to_load, f'checkpoint-{args.checkpoint}'))
        model.to(device)
        data_collator = DataCollartorForLMSQL2Text(
            tokenizer=decoder_tokenizer,
            model=model,
            grammar=grammar,
            schema=table_data,
            device=device
        )

        evaulate_decode_only(
            args,
            val_sql_data,
            data_collator,
            model,
            decoder_tokenizer,
            args.model_to_load,
            args.checkpoint
        )

    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print(f'Number of Params: {pytorch_total_params}!')

if __name__ == '__main__':
    main()
