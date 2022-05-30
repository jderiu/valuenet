import copy
import json

import torch, os
import numpy as np
from tqdm import tqdm
from datasets import load_metric
from src.spider.test_suite_eval.process_sql import tokenize
from transformers.models.bart.modeling_bart import BartForSequenceClassification
from transformers.models.bert.modeling_bert import BertForSequenceClassification
from transformers.models.bert.tokenization_bert import BertTokenizer
from transformers.models.bart.tokenization_bart import BartTokenizer
from transformers.trainer_seq2seq import Trainer
from transformers.training_args_seq2seq import TrainingArguments
from transformers import SchedulerType

from src.config import write_config_to_file, read_arguments_pretrain
from src.intermediate_representation import semQL
from src.spider import spider_utils

from src.utils import setup_device, set_seed_everywhere, create_experiment_folder
from src.automated_evaluation.data_collator import DataCollatorSQLPlusText
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
# initialize experiment tracking @ Weights & Biases
import wandb

def compute_metrics(eval_preds):
    preds, labels = eval_preds
    preds = np.argmax(preds, axis=1)
    f1_score_out = f1_score(labels, preds, average='macro')
    precision_out = precision_score(labels, preds, average='macro')
    recall_out = recall_score(labels, preds, average='macro')
    accuracy_out = accuracy_score(labels, preds)
    return {
        'f1_score': f1_score_out,
        'precision': precision_out,
        'recall': recall_out,
        'accuracy': accuracy_out
    }

def batch_list(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]

if __name__ == '__main__':
    args = read_arguments_pretrain()

    model_path = os.path.join(args.model_to_load, f'checkpoint-{args.model_checkpoint}')
    bridge_eval_beams = []
    with open(os.path.join(args.model_to_load, 'bridge_table_output_withdb_nubia.txt'), 'rt', encoding='utf-8') as f:
        for line in f:
            bridge_eval_beams.append(json.loads(line))

    device, n_gpu = setup_device()
    #device = 'cpu'
    set_seed_everywhere(args.seed, n_gpu)

    sql_data, table_data, val_sql_data, val_table_data = spider_utils.load_dataset(args.data_dir, use_small=args.toy)
    #sql_data = spider_utils.negative_sampling_augmentation(sql_data, aug_num=5)
    #val_sql_data = spider_utils.negative_sampling_augmentation(val_sql_data, aug_num=1)
    grammar = semQL.Grammar()


    model = BartForSequenceClassification.from_pretrained(model_path)
    decoder_tokenizer = BartTokenizer.from_pretrained('facebook/bart-base', add_prefix_space=True)
    decoder_tokenizer.sep_token = decoder_tokenizer.cls_token
    model.to(device)

    data_collator = DataCollatorSQLPlusText(
        tokenizer=decoder_tokenizer,
        grammar=grammar,
        schema=table_data,
        device=device
    )

    nocuda = not str(device) == 'cuda'

    f = open(os.path.join(args.model_to_load, 'bridge_table_output_withdb_nubia_and_critic.txt'), 'wt', encoding='utf-8')
    #align valid data to beam outs
    reorderd_bridge_eval_beams, reorderd_val_sql_data = [], []
    for idx, entry in enumerate(val_sql_data):
        found_beam = False
        for beam_line in bridge_eval_beams:
            if entry['question'] == beam_line['beams'][0]['orig_question']:
                reorderd_bridge_eval_beams.append(beam_line)
                reorderd_val_sql_data.append(entry)
                found_beam = True
                break
        if not found_beam:
            print('Cannot find beam for question:', entry['question'])

    #align valid data to beam outs
    labels, preds = [], []
    counter = 0
    for batch in tqdm(batch_list(reorderd_val_sql_data, args.batch_size), desc="Evaluating"):
        for entry in batch:
            beam_line = reorderd_bridge_eval_beams[counter]
            beam_out = beam_line['beams']
            if not beam_out[0]['orig_question'] == entry['question']:
                print(beam_out[0]['orig_question'], entry['question'])
                counter += 1
                continue
            entry_batch = []
            for beam in beam_out:
                orig_entry = copy.deepcopy(entry)
                orig_entry['query'] = beam['inferred_code']
                orig_entry['query_toks'] = tokenize(beam['inferred_code'])
                orig_entry['label'] = int(beam['is_correct'])
                labels.append(int(beam['is_correct']))
                entry_batch.append(orig_entry)

            encoded_batch = data_collator(entry_batch)
            with torch.no_grad():
                outputs = model(**encoded_batch)
                probas = torch.softmax(outputs.logits, dim=1)[:, 1]
                predicted = torch.argmax(outputs.logits, dim=1)
                preds.extend(predicted.cpu().numpy().tolist())
            for idx, beam in enumerate(beam_out):
                score = float(probas[idx])
                beam['critic_score'] = score
            f.write(json.dumps(beam_line) + '\n')
            counter += 1

    labels, preds = np.array(labels), np.array(preds)
    f1_score_out = f1_score(labels, preds, average='macro')
    precision_out = precision_score(labels, preds, average='macro')
    recall_out = recall_score(labels, preds, average='macro')
    accuracy_out = accuracy_score(labels, preds)
    print({
        'f1_score': f1_score_out,
        'precision': precision_out,
        'recall': recall_out,
        'accuracy': accuracy_out
    })