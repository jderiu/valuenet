import csv, os, json
from collections import defaultdict

data_dir = 'experiments/expert_annotation/'

with open(os.path.join(data_dir, 'index_to_system.json'), 'rt', encoding='utf-8') as ifile:
    idx2system = json.load(ifile)

system_to_ratings = defaultdict(lambda : [])
for fname in os.listdir(data_dir):
    if 'to_annotate_annotated' in fname:
        with open(os.path.join(data_dir, fname), 'rt', encoding='utf-8') as ifile:
            reader = csv.DictReader(ifile, delimiter=';')
            for row in reader:
                system = idx2system[row['Sample Id']]
                rating = int(row['Is Correct'])
                system_to_ratings[system].append(rating)

for system, ratings in system_to_ratings.items():
    acc = sum(ratings)/len(ratings)
    print(system, sum(ratings), len(ratings), acc)