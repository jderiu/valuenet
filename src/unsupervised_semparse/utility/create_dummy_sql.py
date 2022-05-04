import random, json, os
from src.spider.test_suite_eval.process_sql import get_sql, get_schema
from src.tools.training_data_builder.schema import SchemaIndex
from src.preprocessing.sql2SemQL import Parser

def main():
    with open('data/spider/original/tables.json', 'r', encoding='utf8') as f:
        table_datas = json.load(f)
    db_dir = 'data/spider/testsuite_databases'
    tables = {x['db_id']: x for x in table_datas}
    query_tempate = "SELECT count(*) FROM {}"
    parser = Parser()
    dummy_data_for_db_id = {}
    for db_id, data in tables.items():
        print(db_id)
        db = os.path.join(db_dir, db_id, db_id + ".sqlite")
        column_names_original_lower = [[x[0], x[1].lower()] for x in data['column_names_original']]
        table_names_original_lower = [x.lower() for x in data['table_names_original']]
        schema = SchemaIndex(get_schema(db), column_names_original_lower, table_names_original_lower )

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