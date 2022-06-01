import argparse
import os
import random
from pathlib import Path

import openai


def ask_gpt(prompt: str, number_of_choices: int, model_id: str):
    #prompt = sample + '\n\n###\n\n'
    response = openai.Completion.create(
        model=model_id,
        prompt=prompt,
        # top_p=0.9,
        temperature=0.1,
        max_tokens=100,
        n=1,
        # frequency_penalty=0.5,
        # presence_penalty=0.5,
    )

    print(response)
    return response, prompt


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
    openai.api_key = os.getenv("OPENAI_API_KEY")

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--data_path', type=str, default='data/cordis')
    arg_parser.add_argument('--output_folder', type=str, default='data/cordis/generative')
    arg_parser.add_argument('--number_of_choices', type=int, default=8)
    arg_parser.add_argument('--gpt3_finetuned_model', type=str, default='text-davinci-002')

    args = arg_parser.parse_args()
    single_request(args)
