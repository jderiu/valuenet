import json
import torch
import os.path

from transformers import AutoTokenizer
from transformers.models.gpt2.modeling_gpt2 import GPT2LMHeadModel
from src.config import read_arguments_evaluation
from src.intermediate_representation import semQL
from src.utils import setup_device, set_seed_everywhere
from src.synthetic_data.gpt2_data_collator import DataCollartorForLMSQL2Text
from src.spider import spider_utils
from tqdm import tqdm


def main():
    args = read_arguments_evaluation()

    device, n_gpu = setup_device()
    set_seed_everywhere(args.seed, n_gpu)

    table_path = os.path.join(args.data_dir, "original", "tables.json")
    schemas_raw, schemas_dict = spider_utils.load_schema(table_path)
    grammar = semQL.Grammar()

    print("Loading pre-trained model from '{}'".format(args.model_to_load))
    with open(os.path.join(args.model_to_load, "args.json"), "rt", encoding='utf-8') as f:
        train_args = json.load(f)

    decoder_tokenizer = AutoTokenizer.from_pretrained(train_args['decoder_pretrained_model'], add_prefix_space=True)
    decoder_tokenizer.padding_side = 'left'
    if decoder_tokenizer.pad_token_id is None:
        decoder_tokenizer.pad_token = decoder_tokenizer.bos_token
    if decoder_tokenizer.sep_token_id is None:
        decoder_tokenizer.sep_token = decoder_tokenizer.bos_token

    data_collator = DataCollartorForLMSQL2Text(
        tokenizer=decoder_tokenizer,
        grammar=grammar,
        schema=schemas_dict,
        device=device
    )

    preprocessed_dataset = []
    with open(os.path.join(args.data_dir, "original/train.json"), "rt", encoding='utf-8') as f:
        data = json.load(f)
        queries_only = {dp['query'].replace("\n", " ").replace("\t", " "): dp['db_id'] for dp in data}

        for query, db_id in queries_only.items():
            try:
                encoded_dp = data_collator(
                    query,
                    db_id
                )
                preprocessed_dataset.append((query, db_id, encoded_dp))
            except Exception as e:
                print("Error:", query, e)

    # model = GPT2LMHeadModel.from_pretrained('gpt2')
    model = GPT2LMHeadModel.from_pretrained(os.path.join(args.model_to_load, f'checkpoint-{args.checkpoint}'))
    model.to(device)
    model.eval()
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print(f'Number of Params: {pytorch_total_params}!')

    out_data = []
    for query, db_id, batch in tqdm(preprocessed_dataset):
        with torch.no_grad():
            generated_out = model.generate(
                batch['input_ids'],
                attention_mask=batch['attention_mask'],
                max_length=batch['input_ids'].shape[1] + 32,
                num_beams=15,
                repetition_penalty=2.5,
                no_repeat_ngram_size=3,
                pad_token_id=decoder_tokenizer.pad_token_id,
            )
        decoded_out = decoder_tokenizer.batch_decode(generated_out, skip_special_tokens=True)
        pred_batch_out = [x.split('TEXT:')[1].replace('\n', '').replace('TEXT :', '').replace('TEXT', '') for x in decoded_out][0]
        print(pred_batch_out)
        out_dp = {
            'query': query,
            'db_id':db_id,
            'question':pred_batch_out
        }
        out_data.append(out_dp)
    with open(os.path.join(args.data_dir, "sql2nl_output.jsonl"), "wt", encoding='utf-8') as f:
        for dp in out_data:
            f.write(json.dumps(dp) + '\n')


if __name__ == '__main__':
    main()
