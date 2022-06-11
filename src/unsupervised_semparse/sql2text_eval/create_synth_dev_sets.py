import json, os
from src.config import read_arguments_evaluation




def main():
    args = read_arguments_evaluation()
    logging_path = args.model_to_load
    checkpoint_nr = args.checkpoint

    with open(os.path.join(args.data_dir, "original/dev.json"), "rt", encoding='utf-8') as f:
        data = json.load(f)
        queries_only = {dp['query'].replace("\n", " ").replace("\t", " "): dp['db_id'] for dp in data}

    with open(os.path.join(logging_path, f'out_final_{checkpoint_nr}.json'), 'rt', encoding='utf-8') as f2:
        in_json = json.load(f2)

    out_dps = []
    for entry in in_json:
        db_id = queries_only.get(entry['query'])
        if db_id is None:
            print('Not Found', entry['query'])
            continue
        out_dp = {
            "query": entry['query'],
            "db_id": db_id,
            "question": entry['synthetic_answer']
        }
        out_dps.append(out_dp)

    with open(os.path.join(logging_path, f'sql2nl_output.json'), 'wt', encoding='utf-8') as f2:
        json.dump(out_dps, f2)

if __name__ == '__main__':
    main()
