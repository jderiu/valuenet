import argparse, json
import os
import random
from pathlib import Path
import time
import openai
from dotenv import load_dotenv
from collections import defaultdict

load_dotenv()


def fine_tune(file_id, base_model_id):
    response = openai.FineTune.create(training_file=file_id, model=base_model_id)
    ft_id = response['id']
    print(response)
    while response['status'] != 'succeeded':
        time.sleep(5)
        response = openai.FineTune.retrieve(id=ft_id)
        print(response)

if __name__ == '__main__':
    random.seed(42)
    openai.api_key = os.environ.get("OPENAI_API_KEY")

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--data_path', type=str, default='data/spider/original')
    arg_parser.add_argument('--output_folder', type=str, default='experiments/sql2text_decode_gpt3_ft')
    arg_parser.add_argument('--number_of_choices', type=int, default=1)
    arg_parser.add_argument('--toy', default=False, action='store_true')
    arg_parser.add_argument('--gpt3_finetuned_model', type=str, default='davinci')
    arg_parser.add_argument('--n_samples_pred_db', type=int, default=3)

    args = arg_parser.parse_args()
    with open(os.path.join(args.output_folder, 'FileID'), 'rt', encoding='utf-8') as ifile:
        file_id = ifile.readline().replace('\n', '')
        print(file_id)

    fine_tune(file_id, args.gpt3_finetuned_model)
