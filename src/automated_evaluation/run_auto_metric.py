import torch, os
import numpy as np
from tqdm import tqdm
from datasets import load_metric

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

if __name__ == '__main__':
    args = read_arguments_pretrain()

    # log hyperparameters to Weights & Biases
    experiment_name, output_path = create_experiment_folder(args.model_output_dir, args.exp_name)
    print("Run experiment '{}'".format(experiment_name))

    wandb.init(
        project="sql2text",
        group="auto_metric",
        name=experiment_name
    )

    wandb.config.update(args)

    write_config_to_file(args, output_path)

    device, n_gpu = setup_device()
    #device = 'cpu'
    set_seed_everywhere(args.seed, n_gpu)

    sql_data, table_data, val_sql_data, val_table_data = spider_utils.load_dataset(args.data_dir, use_small=args.toy)
    sql_data = spider_utils.negative_sampling_augmentation(sql_data, aug_num=5)
    val_sql_data = spider_utils.negative_sampling_augmentation(val_sql_data, aug_num=1)
    grammar = semQL.Grammar()

    ignore_keys_for_eval = ['past_key_values', 'encoder_last_hidden_state', 'hidden_states', 'cross_attentions']
    #decoder_tokenizer = BertTokenizer.from_pretrained('bert-large-cased', add_prefix_space=True)
    decoder_tokenizer = BartTokenizer.from_pretrained('facebook/bart-base', add_prefix_space=True)
    decoder_tokenizer.sep_token = decoder_tokenizer.cls_token
    #decoder_tokenizer.eos_token = decoder_tokenizer.cls_token
    model = BertForSequenceClassification.from_pretrained('bert-large-cased', num_labels=2)
    #model = BartForSequenceClassification.from_pretrained('facebook/bart-base', num_labels=2)
    model.to(device)
    model.config.problem_type = 'single_label_classification'
    data_collator = DataCollatorSQLPlusText(
        tokenizer=decoder_tokenizer,
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
        #eval_steps=2,
        eval_accumulation_steps=args.batch_size,
        no_cuda=nocuda,
        #fp16=True,
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
        compute_metrics=compute_metrics
    )

    trainer.train(ignore_keys_for_eval=ignore_keys_for_eval)
