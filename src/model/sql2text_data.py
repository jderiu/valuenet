import torch, copy
from transformers import PreTrainedTokenizer
from src.spider.example import Batch
from src.spider.example_builder import build_sql2text_example, build_example
from src.model.encoder.input_features import encode_input_sql2text, encode_input
from spacy.lang.en import English

class DataCollartorForLMSQL2Text:
    label_pad_token_id: int = -100

    def __init__(
            self,
            tokenizer: PreTrainedTokenizer,
            model,
            grammar,
            schema,
            device
    ):
        self.tokenizer = tokenizer
        self.schema = schema
        self.grammar = grammar
        self.device = device
        self.model = model

    def __call__(self, batch, return_tensors=None, is_eval=False):
        examples = []
        for data_row in batch:
            try:
                example = build_sql2text_example(data_row, self.schema, is_decode_only=True, is_eval=is_eval)
                examples.append(example)
            except RuntimeError as e:
                print("Exception while building example (training): {}".format(e))
        batch = Batch(examples, self.grammar, cuda=self.device)

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

        return out_batch


class DataCollatorForSQL2Text:
    label_pad_token_id: int = -100

    def __init__(
            self,
            encoder_tokenizer: PreTrainedTokenizer,
            decoder_tokenizer: PreTrainedTokenizer,
            model,
            grammar,
            schema,
            device
    ):
        self.encoder_tokenizer = encoder_tokenizer
        self.decoder_tokenizer = decoder_tokenizer
        self.schema = schema
        self.grammar = grammar
        self.device = device
        self.model = model

        if self.decoder_tokenizer.bos_token_id is not None:
            self.model.config.decoder_start_token_id = self.decoder_tokenizer.bos_token_id
        else:
            self.model.config.decoder_start_token_id = self.decoder_tokenizer.eos_token_id

        if self.decoder_tokenizer.pad_token_id is not None:
            self.model.config.pad_token_id = self.decoder_tokenizer.pad_token_id
        else:
            self.model.config.pad_token_id = self.decoder_tokenizer.eos_token_id
        self.encoder_tokenizer.do_basic_tokenize = False

    def __call__(self, batch, return_tensors=None):
        examples = []
        for data_row in batch:
            try:
                example = build_sql2text_example(data_row, self.schema)
                examples.append(example)
            except RuntimeError as e:
                print("Exception while building example (training): {}".format(e))
        batch = Batch(examples, self.grammar, cuda=self.device)

        input_ids_tensor, attention_mask_tensor, input_lengths = encode_input(batch.all_query_tokens,
                                                                              batch.all_column_tokens,
                                                                              batch.all_table_names,
                                                                              batch.values,
                                                                              self.encoder_tokenizer,
                                                                              self.encoder_tokenizer.model_max_length,
                                                                              self.device)

        questions = [example.question for example in examples]
        # labels = self.decoder_tokenizer(
        #     questions,
        #     padding=True,
        #     truncation=True,
        #     max_length=self.decoder_tokenizer.model_max_length,
        #     return_tensors="pt"
        # )

        labels_vanilla = self.decoder_tokenizer(
            questions,
            truncation=True,
            max_length=self.decoder_tokenizer.model_max_length,
            add_special_tokens=False
        )['input_ids']

        max_label_length = max(len(l) for l in labels_vanilla)
        for van_label in labels_vanilla:
            remainder = [self.label_pad_token_id] * (max_label_length - len(van_label))
            van_label.append(self.decoder_tokenizer.eos_token_id)
            van_label.extend(remainder)
        labels = torch.tensor(labels_vanilla, dtype=torch.long, device=self.device)

        out_batch = {
            "input_ids": input_ids_tensor,
            "attention_mask": attention_mask_tensor,
            "labels": labels,
            "return_dict": True
        }

        # prepare decoder_input_ids
        if (
                labels is not None
                and self.model is not None
                and hasattr(self.model, "prepare_decoder_input_ids_from_labels")
        ):
            decoder_input_ids = self.model.prepare_decoder_input_ids_from_labels(labels=out_batch["labels"])
            out_batch["decoder_input_ids"] = decoder_input_ids

        return out_batch


class DataCollatorCycle():
    def __init__(
            self,
            grammar,
            schema,
            device
    ):
        self.schema = schema
        self.grammar = grammar
        self.device = device
        self.tokenizer = English().tokenizer

    def __call__(self, batch, alt_questions):
        examples, original_rows = [], []
        for i, (data_row, alt_question) in enumerate(zip(batch, alt_questions)):
            original_row = copy.deepcopy(data_row)
            try:
                if alt_question is not None:
                    question_tokenized = self.tokenizer(alt_question)
                    question_tokenized = [str(token) for token in question_tokenized]
                    original_row['question_toks'] = question_tokenized
                example = build_example(original_row, self.schema)
                examples.append(example)
            except RuntimeError as e:
                print("Exception while building example (training): {}".format(e))
            original_rows.append(original_row)
        return examples, original_rows