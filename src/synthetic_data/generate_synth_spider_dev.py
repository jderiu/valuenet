import argparse, json
import os
import random
from pathlib import Path
import time
import openai
from dotenv import load_dotenv
from collections import defaultdict

load_dotenv()

def ask_gpt(prompt: str, number_of_choices, model_id: str):
    # prompt = sample + '\n\n###\n\n'
    response = openai.Completion.create(
        model=model_id,
        prompt=prompt,
        temperature=0.1,
        max_tokens=100,
        n=number_of_choices,
        stop=['###']
    )

    print(response)
    return response, prompt


def gen_questions(args):
    start_secs = 0.25
    with open(os.path.join(args.data_path, 'dev.json'), 'rt', encoding='utf8') as ifile:
        data = json.load(ifile)
    out_json = []
    sql_to_synth_text = {}
    sql_to_orig_text = defaultdict(lambda :[])
    if args.toy:
        data = data[:20]
    with open(os.path.join(args.output_folder, 'results_final_cordis_0.txt'), 'wt', encoding='utf8') as ofile:
        counter = 0
        n_dps = len(data)
        while counter < n_dps:
            print(f'Try Sample Nr: {counter + 1}')
            try:
                entry = data[counter]
                sql_query = entry['query']
                orig_question = entry['question']
                sql_to_orig_text[sql_query].append(orig_question)
                if sql_to_synth_text.get(sql_query) is not None:
                    counter += 1
                    continue
                curr_prompt = "#{} ->".format(sql_query)
                response, prompt = ask_gpt(
                    curr_prompt,
                    number_of_choices=args.number_of_choices,
                    model_id=args.gpt3_finetuned_model
                )
                synth_question = response['choices'][0].text.replace('#', '').replace('\n', '')

                out_json.append({'query': sql_query, 'synthetic_answer': synth_question})
                ofile.write(f"{synth_question}\t{orig_question}\t{sql_query}\n")
                counter += 1
                sql_to_synth_text[sql_query] = synth_question
                time.sleep(start_secs)
                start_secs = 0.25
            except Exception as e:
                print(f'Exception Maybe due to Time: {e}')
                start_secs*=2
                time.sleep(start_secs)

    for out_json_entry in out_json:
        out_json_entry['questions'] = sql_to_orig_text[out_json_entry['query']]

    with open(os.path.join(args.output_folder, 'out_final_cordis_0.json'), 'wt', encoding='utf8') as ofile:
        json.dump(out_json, ofile)


def single_request(args):
    sql_query = """
SELECT funding_schemes.title FROM funding_schemes JOIN projects ON funding_schemes.code = projects.ec_fund_scheme WHERE projects.unics_id = 156767
""".strip()

    prompt = f"#Transate SQL to Natural Language\n#SQL:{sql_query}\n#Natural Language:"

    response, prompt = ask_gpt(prompt,
                               number_of_choices=args.number_of_choices,
                               model_id=args.gpt3_finetuned_model)

    gpt_choices = [f"({idx}) {c['text'].strip()}" for idx, c in enumerate(response['choices'])]

    with open(Path(args.output_folder) / f'999.txt', 'w') as f:
        f.write(prompt)
        f.write('\nGPT-3 choices:\n')
        f.write('\n'.join(gpt_choices))


if __name__ == '__main__':
    random.seed(42)
    openai.api_key = os.environ.get("OPENAI_API_KEY")

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--data_path', type=str, default='data/cordis/original')
    arg_parser.add_argument('--output_folder', type=str, default='experiments/sql2text_decode_gpt3_ft')
    arg_parser.add_argument('--number_of_choices', type=int, default=1)
    arg_parser.add_argument('--toy', default=False, action='store_true')
    arg_parser.add_argument('--gpt3_finetuned_model', type=str, default='davinci:ft-zhaw-2022-06-01-16-39-16')

    args = arg_parser.parse_args()
    out_path = args.output_folder
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    gen_questions(args)
