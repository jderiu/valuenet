import json, os, random, copy, csv
from src.spider import spider_utils
from collections import defaultdict

dataset_paths = [
    'sql2text_decode_gpt2',
    'sql2text_decode_gpt3',
    'sql2text_decode_gpt3_ft',
    'T5explain'
]

n_annotators = 7
n_samples_4_ann = 100
n_samples = int(n_annotators*n_samples_4_ann/len(dataset_paths))

def main():
    base_path = 'experiments/'
    _, table_data, val_sql_data, val_table_data = spider_utils.load_dataset('data/spider', use_small=False)
    sql_to_dp = {dp['query']: copy.deepcopy(dp) for dp in val_sql_data}

    system_to_data = {}
    for system in dataset_paths:
        with open(os.path.join(base_path, system, 'out_final_0.json'), 'rt', encoding='utf-8') as ifile:
            data = json.load(ifile)
            sql_to_data = {x['query']: x for x in data}
            system_to_data[system] = sql_to_data

    #remove duplicates first
    index = 100
    index_to_system, index_to_data = {}, defaultdict(lambda : [])
    to_annotate_samples = random.sample(sql_to_dp.items(), k=n_samples)
    for query, dp in to_annotate_samples:
        for system, data_for_system in system_to_data.items():
            data = data_for_system[query]
            index_to_system[index] = system
            index_to_data[query].append((index, dp['db_id'], data['questions'][0], query, data['synthetic_answer']))
            index += 1

    print(index_to_system, index_to_data)
    data_for_annotator = defaultdict(lambda : [])
    curr_ann = 0
    for query, data in index_to_data.items():
        data_for_annotator[curr_ann].extend(data)
        curr_ann = (curr_ann + 1) % n_annotators

    for annotator, data in data_for_annotator.items():
        with open(f'experiments/expert_annotation/{annotator}_to_annotate.csv', 'wt', encoding='utf-8', newline='') as ofile:
            fieldnames = ['Sample Id','DB','Original Question','Original SQL','Synthetic Question','Is Correct','Notes']
            writer = csv.DictWriter(ofile, fieldnames=fieldnames)
            writer.writeheader()
            for dp in data:
                writer.writerow({
                    'Sample Id': dp[0],
                    'DB': dp[1],
                    'Original Question': dp[2],
                    'Original SQL': dp[3],
                    'Synthetic Question': dp[4],
                    'Is Correct': '',
                    'Notes': ''
                })

    with open(f'experiments/expert_annotation/index_to_system.json', 'wt', encoding='utf-8') as ofile:
        json.dump(index_to_system, ofile)

if __name__ == '__main__':
    main()