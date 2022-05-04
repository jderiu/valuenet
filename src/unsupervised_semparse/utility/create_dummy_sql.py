import random, json
from src.config import read_arguments_cycletrain
from src.spider.test_suite_eval.process_sql import get_sql, Schema
from src.manual_inference.helper import get_schemas_spider
from src.tools.training_data_builder.schema import build_schema_mapping
from src.preprocessing.sql2SemQL import Parser

def main():
    with open('data/spider/original/tables.json', 'r', encoding='utf8') as f:
        table_datas = json.load(f)

    tables = {x['db_id']: x for x in table_datas}
    query_tempate = "SELECT * FROM {}"
    args = read_arguments_cycletrain()
    schemas_raw_spider, schemas_dict_spider, schema_path_spider, database_path_spider = get_schemas_spider()
    parser = Parser()
    dummy_data_for_db_id = {}
    for db_id, data in tables.items():
        print(db_id)
        schema_mapping = build_schema_mapping(data)
        schema = Schema(schema_mapping)

        random_tbl = random.choice(data['table_names_original'])
        sql_query = query_tempate.format(random_tbl)
        spider_sql_structure, sql_tokenized = get_sql(schema, sql_query)
        keys = {}
        for kv in tables[db_id]['foreign_keys']:
            keys[kv[0]] = kv[1]
            keys[kv[1]] = kv[0]
        for id_k in tables[db_id]['primary_keys']:
            keys[id_k] = id_k

        query = {
            'sql': spider_sql_structure,
            'col_set': [x[1] for x in tables[db_id]['column_names']],
            'col_table': [x[0] for x in tables[db_id]['column_names']],
            'names': [x[1] for x in tables[db_id]['column_names']],
            'table_names': [x for x in tables[db_id]['table_names']],
            'question': 'Dummy quesiton',
            'query': sql_query,
            'keys': keys
        }

        semql_result = parser.full_parse(query)
        rule_label = " ".join([str(x) for x in semql_result])
        dummy_data_for_db_id[db_id] = {
            'query': sql_query,
            'rule_label': rule_label,
            'query_toks': sql_tokenized
        }
    with open('data/spider/dummy_queries.json', 'w', encoding='utf8') as f:
        json.dump(dummy_data_for_db_id, f, indent=4)


if __name__ == '__main__':
    main()