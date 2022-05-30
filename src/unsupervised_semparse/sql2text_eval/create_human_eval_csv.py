import json, os, csv

from src.config import read_arguments_evaluation


def main():
    args = read_arguments_evaluation()
    logging_path = args.model_to_load
    checkpoint_nr = args.checkpoint

    with open(os.path.join(logging_path, f'out_final_{checkpoint_nr}.json'), 'rt', encoding='utf-8') as f2:
        in_json = json.load(f2)
    if args.toy:
        in_json = in_json[:50]

    #create csv file
    with open(os.path.join(logging_path, f'out_final_{checkpoint_nr}.csv'), 'wt', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['question', 'query', 'synthetic_answer'])
        writer.writeheader()
        for row in in_json:
            writer.writerow(row)

if __name__ == '__main__':
    main()
