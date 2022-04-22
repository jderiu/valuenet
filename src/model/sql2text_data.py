from transformers import PreTrainedTokenizer
from src.spider.example import Batch
from src.spider.example_builder import build_sql2text_example
from src.model.encoder.input_features import encode_input


class DataCollatorForSQL2Text:
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

        self.model.config.decoder_start_token_id = self.decoder_tokenizer.bos_token_id
        self.model.config.pad_token_id = self.decoder_tokenizer.pad_token_id
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

        input_ids_tensor, attention_mask_tensor, input_lengths = encode_input(batch.all_question_tokens,
                                                                              batch.all_column_tokens,
                                                                              batch.all_table_names,
                                                                              batch.values,
                                                                              self.encoder_tokenizer,
                                                                              self.encoder_tokenizer.model_max_length,
                                                                              self.device)

        questions = [example.question for example in examples]
        labels = self.decoder_tokenizer(
            questions,
            padding=True,
            truncation=True,
            max_length=self.decoder_tokenizer.model_max_length,
            return_tensors="pt"
        )

        out_batch = {
            "input_ids": input_ids_tensor,
            "attention_mask": attention_mask_tensor,
            "labels": labels.data['input_ids'].to(self.device),
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

