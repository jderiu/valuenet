from transformers import PreTrainedTokenizer
from src.spider.example_builder import build_table_column_mappings
from src.model.encoder.input_features import encode_input_sql2text
from src.spider.test_suite_eval.process_sql import tokenize, get_sql
from src.preprocessing.sql2SemQL import Parser
from src.tools.training_data_builder.schema import SchemaIndex
from collections import defaultdict

class DataCollartorForLMSQL2Text:
    label_pad_token_id: int = -100

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
        self.db_schema = {}
        for db_id, schema in schema.items():
            column_names_original_lower = [[x[0], x[1].lower()] for x in schema['column_names_original']]
            table_names_original_lower = [x.lower() for x in schema['table_names_original']]
            table_name_to_column_names = defaultdict(lambda : [])
            for tbl_id, col_name in column_names_original_lower:
                tbl_name = table_names_original_lower[tbl_id]
                table_name_to_column_names[tbl_name].append(col_name)
            self.db_schema[db_id] = {
                'column_names_original': column_names_original_lower,
                'table_names_original': table_names_original_lower,
                'table_name_to_column_names': table_name_to_column_names
            }

    def __call__(
            self,
            sql: str,
            db_id: str,
            return_tensors=None,
    ):
        query_tokens = tokenize(sql)
        schema = self.schema[db_id]
        col_set = {'col_set': [x[1] for x in schema['column_names']]}
        column_names, column_set, column_table_dict, columns_per_table, table_names = build_table_column_mappings(col_set, schema)
        column_names_original_lower = self.db_schema[db_id]['column_names_original']
        table_names_original_lower = self.db_schema[db_id]['table_names_original']
        table_name_to_column_names = self.db_schema[db_id]['table_name_to_column_names']

        schema_idx = SchemaIndex(table_name_to_column_names, column_names_original_lower, table_names_original_lower)

        spider_sql_structure, sql_tokenized = get_sql(schema_idx, sql)
        parser = Parser(build_value_list=True)
        keys = {}
        for kv in schema['foreign_keys']:
            keys[kv[0]] = kv[1]
            keys[kv[1]] = kv[0]
        for id_k in schema['primary_keys']:
            keys[id_k] = id_k

        query = {
            'sql': spider_sql_structure,
            'col_set': [x[1] for x in schema['column_names']],
            'col_table': [x[0] for x in schema['column_names']],
            'names': [x[1] for x in schema['column_names']],
            'table_names': [x for x in schema['table_names']],
            'question': 'Dummy quesiton',
            'query': sql,
            'query_toks_no_value': sql_tokenized,
            'keys': keys
        }

        parser.full_parse(query)

        input_ids_tensor, attention_mask_tensor, input_lengths = encode_input_sql2text(
            [[['TEXT:']]],
            [[[token] for token in query_tokens]],
            [column_set],
            [table_names],
            [parser.values],
            self.tokenizer,
            self.tokenizer.model_max_length,
            self.device,
            add_sep_token=not True,
        )

        out_batch = {
            "input_ids": input_ids_tensor,
            "attention_mask": attention_mask_tensor,
            "labels": input_ids_tensor,
        }

        return out_batch
