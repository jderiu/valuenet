import copy

import torch
from transformers import PreTrainedTokenizer
from src.spider.example import Batch
from src.spider.example_builder import build_example, build_sql2text_example
from src.model.encoder.input_features import encode_input_sql2text, encode_input


class DataCollatorText2SQL:
    def __init__(
            self,
            grammar,
            schema,
            device
    ):
        self.schema = schema
        self.grammar = grammar
        self.device = device

    def __call__(self, batch):
        examples, original_rows = [], []
        for i, data_row in enumerate(batch):
            #do not destroy the original data :)
            original_row = copy.deepcopy(data_row)
            try:
                original_rows.append(original_row)
                example = build_example(original_row, self.schema)
                examples.append(example)
            except RuntimeError as e:
                print("Exception while building example (training): {}".format(e))
        return examples, original_rows


class DataCollatorSQL2Text:
    def __init__(
            self,
            tokenizer: PreTrainedTokenizer,
            grammar,
            schema,
            device
    ):
        self.tokenizer = tokenizer
        self.schema = schema
        self.grammar = grammar
        self.device = device

    def __call__(self, batch, return_question_mask=False, is_eval=False):
        examples, original_rows = [], []
        for data_row in batch:
            original_row = copy.deepcopy(data_row)
            try:
                example = build_sql2text_example(data_row, self.schema, is_decode_only=True, is_eval=is_eval)
                examples.append(example)
                original_rows.append(original_row)
            except RuntimeError as e:
                print("Exception while building example (training): {}".format(e))
        batch = Batch(examples, self.grammar, cuda=self.device != 'cpu')

        input_ids_tensor, attention_mask_tensor, input_lengths = encode_input_sql2text(
            batch.all_question_tokens,
            batch.all_query_tokens,
            batch.all_column_tokens,
            batch.all_table_names,
            batch.values,
            self.tokenizer,
            self.tokenizer.model_max_length,
            self.device,
            add_sep_token=not is_eval,
        )

        out_batch = {
            "input_ids": input_ids_tensor,
            "attention_mask": attention_mask_tensor,
            "labels": input_ids_tensor,
        }

        if return_question_mask:
            question_mask = torch.zeros_like(input_ids_tensor)
            for i, length in enumerate(list(zip(*input_lengths))):
                question_pos = sum([sum(x) for x in length[:-1]])
                question_mask[i, question_pos:] = 1
            return out_batch, original_rows, question_mask
        return out_batch, original_rows