import torch, random
import numpy as np
from torch.utils.data import RandomSampler
from src.spider.test_suite_eval.evaluation import Evaluator, convert_sql_to_dict


def get_data_loader(data_train, data_dev, batch_size, shuffle_train=True, shuffle_dev=False) -> (torch.utils.data.DataLoader, torch.utils.data.DataLoader):
    train_loader = torch.utils.data.DataLoader(
        batch_size=batch_size,
        dataset=data_train,
        shuffle=shuffle_train,
        collate_fn=lambda x: x  # now dictionary values are not merged!
    )

    dev_loader = torch.utils.data.DataLoader(
        batch_size=batch_size,
        dataset=data_dev,
        shuffle=shuffle_dev,
        collate_fn=lambda x: x  # now dictionary values are not merged!
    )

    return train_loader, dev_loader


def get_random_sampler(data_train, data_dev, batch_size, db_names_to_schema, n_boxes) -> (torch.utils.data.RandomSampler, torch.utils.data.DataLoader):
    train_loader = CurriculumIterator(data_train, db_names_to_schema, n_boxes=n_boxes)
    dev_loader = torch.utils.data.DataLoader(
        batch_size=batch_size,
        dataset=data_dev,
        shuffle=False,
        collate_fn=lambda x: x  # now dictionary values are not merged!
    )

    return train_loader, dev_loader


class CurriculumIterator():

    difficulty_map = {
        'easy': 0,
        'medium': 1,
        'hard': 2,
        'extra': 3
    }

    def __init__(
            self,
            dataset,
            db_names_to_schema,
            n_boxes=5
    ):
        self.dataset = dataset
        self.db_names_to_schema = db_names_to_schema
        self.db_names = list(db_names_to_schema.keys())
        random.shuffle(self.db_names)
        self.extend_hardness()
        self.boxes = [set() for _ in range(n_boxes)]
        self.current_difficulty = 0
        self.current_db_pointer = 0
        self.inverse_difficulty_map = {v: k for k, v in self.difficulty_map.items()}
        self.sample_id_to_box = {}
        self.initialize()

    def __len__(self):
        return len(self.dataset)

    def initialize(self):
        candidates = []
        while len(candidates) == 0:
            curr_db_name = self.db_names[self.current_db_pointer]
            candidates = [x['id'] for x in self.dataset if x['db_id'] == curr_db_name and x['difficulty'] == self.inverse_difficulty_map[self.current_difficulty]]
            if len(candidates) == 0:
                self.current_db_pointer = (self.current_db_pointer + 1) % len(self.db_names)
        self.boxes[0].update(candidates)
        for candidate in candidates:
            self.sample_id_to_box[candidate] = self.current_difficulty

    def get_deck_size(self):
        deck_size = sum([len(box) for box in self.boxes])
        return deck_size

    def update_difficulty(self):
        self.current_db_pointer = (self.current_db_pointer + 1) % len(self.db_names)
        if self.current_db_pointer == 0:
            self.current_difficulty = min(self.current_difficulty + 1, self.difficulty_map['extra'])
        self.initialize()

    def extend_hardness(self):
        evaluator = Evaluator()
        for i, entry in enumerate(self.dataset):
            db_name = entry['db_id']
            sql_str = entry['query']
            sql_dict = convert_sql_to_dict(sql_str, self.db_names_to_schema[db_name])
            entry['difficulty'] = evaluator.eval_hardness(sql_dict)
            entry['id'] = i

    def compute_box_probas(self):
        box_probs = [0.5**i for i in range(1, len(self.boxes))]
        box_probs.append(1 - sum(box_probs))
        box_probs = np.array([x if len(b) > 0 else 0.0 for x, b in zip(box_probs, self.boxes)])
        box_probs = box_probs /box_probs.sum()

        return box_probs

    def sample_batch(self, batch_size):
        sample_ids = [self.sample_next() for _ in range(batch_size)]
        return sample_ids

    def sample_next(self):
        box_probs = self.compute_box_probas()
        current_box = random.choices(self.boxes, box_probs)[0]
        current_sample_id = random.sample(current_box, 1)[0]
        return current_sample_id

    def get_box_distribution(self):
        box_probs = np.array([len(b) for b in self.boxes])
        return box_probs

    def update_sample(self, sample_id, is_correct):
        if is_correct:
            box_id = self.sample_id_to_box[sample_id]
            #already in best box
            if box_id == len(self.boxes) - 1:
                return
            self.boxes[box_id].remove(sample_id)
            self.boxes[box_id + 1].add(sample_id)
            self.sample_id_to_box[sample_id] = box_id + 1
        else:
            box_id = self.sample_id_to_box[sample_id]
            #already in worst box
            if box_id == 0:
                return
            self.boxes[box_id].remove(sample_id)
            self.boxes[box_id - 1].add(sample_id)
            self.sample_id_to_box[sample_id] = box_id - 1

        n_samples_in_pool = sum([len(b) for b in self.boxes])
        if len(self.boxes[0]) < n_samples_in_pool*0.25:
            self.update_difficulty()
