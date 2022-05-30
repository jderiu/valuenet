import csv, os
from src.config import read_arguments_evaluation
from collections import defaultdict


def main():
    args = read_arguments_evaluation()
    logging_path = args.model_to_load
    checkpoint_nr = args.checkpoint

    sample_to_labels = defaultdict(list)
    with open(os.path.join(logging_path, f'out_final_{checkpoint_nr}_res1.csv'), 'rt', encoding='utf-8') as f2:
        reader = csv.reader(f2)
        next(reader)
        for row in reader:
            sample_id = row[0]
            label = row[-1]
            print(sample_id, int(label[0]))
            sample_to_labels[sample_id].append(int(label[0]))

    equivalent_samples, n_sampels = 0, 0
    for sample_id, labels in sample_to_labels.items():
        if len(labels) < 3:
            continue
        n_sampels += 1
        if sum(labels) > 1:
            equivalent_samples += 1
    print(f'{equivalent_samples}/{n_sampels}={equivalent_samples/n_sampels} samples are equivalent')

if __name__ == '__main__':
    main()
