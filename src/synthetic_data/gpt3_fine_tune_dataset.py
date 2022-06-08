import argparse, json
import os
import random
from pathlib import Path
import time
import openai
from dotenv import load_dotenv
from collections import defaultdict

load_dotenv()

def create_fine_tune_dataset(args):
    with open(os.path.join(args.data_path, 'train.json'), 'rt', encoding='utf8') as ifile:
        data = json.load(ifile)
    db_ids = set([x['db_id'] for x in data])
    dataset = []
    for db_id in db_ids:
        dp_for_id = [dp for dp in data if dp['db_id'] == db_id]
        dp_for_query = defaultdict(lambda :[])
        for dp in dp_for_id:
            dp_for_query[dp['query']].append(dp)
        sampled_queries = random.sample(list(dp_for_query), k=args.n_samples_pred_db)
        samples_for_db = [random.choice(dp_for_query[query]) for query in sampled_queries]
        for sample_for_db in samples_for_db:
            dataset.append({
                'prompt': f'{sample_for_db["query"]} ->',
                'completion': f'{sample_for_db["question"]}###'
            })
    return dataset

if __name__ == '__main__':
    random.seed(42)
    openai.api_key = os.environ.get("OPENAI_API_KEY")

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--data_path', type=str, default='data/spider/original')
    arg_parser.add_argument('--output_folder', type=str, default='experiments/sql2text_decode_gpt3_ft')
    arg_parser.add_argument('--number_of_choices', type=int, default=1)
    arg_parser.add_argument('--toy', default=False, action='store_true')
    arg_parser.add_argument('--gpt3_finetuned_model', type=str, default='text-davinci-002')
    arg_parser.add_argument('--n_samples_pred_db', type=int, default=3)

    args = arg_parser.parse_args()
    out_path = args.output_folder
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    dataset = create_fine_tune_dataset(args)
    with open(os.path.join(args.output_folder, 'train_set.jsonl'), 'wt', encoding='utf-8') as ofile:
        for dp in dataset:
            line = json.dumps(dp)
            ofile.write(f'{line}\n')

    response = openai.File.create(file=open(os.path.join(args.output_folder, 'train_set.jsonl')), purpose="fine-tune")
    file_id = response['id']
    print(file_id)
    running = True
    while running:
        time.sleep(2)
        response = openai.File.retrieve(id=file_id)
        if response['status'] == 'processed':
            running = False
            print(f'Done Processing: {file_id}')
        else:
            print(f'Still Processing: {file_id}')