import json
import torch
import os.path

from flask import Flask, make_response, abort, request
from flask_cors import CORS

from transformers import AutoTokenizer
from transformers.models.gpt2.modeling_gpt2 import GPT2LMHeadModel
from src.config import read_arguments_evaluation
from src.intermediate_representation import semQL
from src.utils import setup_device, set_seed_everywhere
from src.synthetic_data.gpt2_data_collator import DataCollartorForLMSQL2Text
from src.spider import spider_utils

args = read_arguments_evaluation()

with open(os.path.join(args.model_to_load, "args.json"), "rt", encoding='utf-8') as f:
    train_args = json.load(f)

app = Flask(__name__, instance_path=f"/{os.getcwd()}/instance")
CORS(app)

device, n_gpu = setup_device()

decoder_tokenizer = AutoTokenizer.from_pretrained(train_args['decoder_pretrained_model'], add_prefix_space=True)
decoder_tokenizer.padding_side = 'left'
if decoder_tokenizer.pad_token_id is None:
    decoder_tokenizer.pad_token = decoder_tokenizer.bos_token
if decoder_tokenizer.sep_token_id is None:
    decoder_tokenizer.sep_token = decoder_tokenizer.bos_token

data_names = ['cordis',
              'hack_zurich',
              'spider',
              'world_cup_data_v2',
              ]

data_collators_for_schema = {data_name:
    DataCollartorForLMSQL2Text(
        tokenizer=decoder_tokenizer,
        grammar=semQL.Grammar(),
        schema=spider_utils.load_schema(os.path.join("data", data_name, "original", "tables.json"))[1],
        device=device
    )
    for data_name in data_names
}


model = GPT2LMHeadModel.from_pretrained(os.path.join(args.model_to_load, f'checkpoint-{args.checkpoint}'))
model.to(device)
model.eval()
pytorch_total_params = sum(p.numel() for p in model.parameters())
print(f'Number of Params: {pytorch_total_params}!')

def _verify_api_key():
    api_key = request.headers.get('X-API-Key', default='No API Key provided', type=str)
    print(f'provided API-Key is {api_key}')
    if not args.api_key == api_key:
        print('Invalid API-Key! Abort with 403')
        abort(403, description="Please provide a valid API Key")

@app.route("/question/<domain>", methods=["PUT"])
@app.route("/api/question/<domain>", methods=["PUT"])  # this is a fallback for local usage, as the reverse-proxy on nginx will add this prefix
def pose_question(domain):
    _verify_api_key()
    data = request.get_json(silent=True)
    query = data['query']
    db_id = data['db_id']

    beam_size = 15
    if 'beam_size' in data:
        beam_size = data['beam_size']

    data_collator = data_collators_for_schema[domain]
    encoded_dp = data_collator(
        query,
        db_id
    )

    with torch.no_grad():
        generated_out = model.generate(
            encoded_dp['input_ids'],
            attention_mask=encoded_dp['attention_mask'],
            max_length=encoded_dp['input_ids'].shape[1] + 32,
            num_beams=beam_size,
            repetition_penalty=2.5,
            no_repeat_ngram_size=3,
            pad_token_id=decoder_tokenizer.pad_token_id,
        )
    decoded_out = decoder_tokenizer.batch_decode(generated_out, skip_special_tokens=True)
    pred_batch_out = [x.split('TEXT:')[1].replace('\n', '').replace('TEXT :', '').replace('TEXT', '') for x in decoded_out][0]
    return {
        'query': query,
        'db_id': db_id,
        'generated_question': pred_batch_out
    }


if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0')