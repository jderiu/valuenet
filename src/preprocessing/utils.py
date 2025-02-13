# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

# -*- coding: utf-8 -*-
"""
# @Time    : 2019/5/24
# @Author  : Jiaqi&Zecheng
# @File    : utils.py
# @Software: PyCharm
"""
import os
import json
from pattern.text.en import lemma
from nltk.stem import WordNetLemmatizer

VALUE_FILTER = ['what', 'how', 'list', 'give', 'show', 'find', 'id', 'order', 'when']
AGG = ['average', 'sum', 'max', 'min', 'minimum', 'maximum', 'between']

wordnet_lemmatizer = WordNetLemmatizer()


def load_dataSets(args):
    with open(args.table_path, 'r', encoding='utf8') as f:
        table_datas = json.load(f)
    with open(args.data_path, 'r', encoding='utf8') as f:
        datas = json.load(f)

    output_tab = {}
    tables = {}
    tabel_name = set()
    for i in range(len(table_datas)):
        table = table_datas[i]
        temp = {}
        temp['col_map'] = table['column_names']
        temp['table_names'] = table['table_names']
        tmp_col = []
        for cc in [x[1] for x in table['column_names']]:
            if cc not in tmp_col:
                tmp_col.append(cc)
        table['col_set'] = tmp_col
        db_name = table['db_id']
        tabel_name.add(db_name)
        table['schema_content'] = [col[1] for col in table['column_names']]
        table['col_table'] = [col[0] for col in table['column_names']]
        output_tab[db_name] = temp
        tables[db_name] = table

    for d in datas:
        d['names'] = tables[d['db_id']]['schema_content']
        d['table_names'] = tables[d['db_id']]['table_names']
        d['col_set'] = tables[d['db_id']]['col_set']
        d['col_table'] = tables[d['db_id']]['col_table']
        keys = {}
        for kv in tables[d['db_id']]['foreign_keys']:
            keys[kv[0]] = kv[1]
            keys[kv[1]] = kv[0]
        for id_k in tables[d['db_id']]['primary_keys']:
            keys[id_k] = id_k
        d['keys'] = keys
    return datas, tables

# todo: refactor this code with method above
def merge_data_with_schema(schema, data):
    output_tab = {}
    tables = {}
    tabel_name = set()
    for i in range(len(schema)):
        table = schema[i]
        temp = {}
        temp['col_map'] = table['column_names']
        temp['table_names'] = table['table_names']
        tmp_col = []
        for cc in [x[1] for x in table['column_names']]:
            if cc not in tmp_col:
                tmp_col.append(cc)
        table['col_set'] = tmp_col
        db_name = table['db_id']
        tabel_name.add(db_name)
        table['schema_content'] = [col[1] for col in table['column_names']]
        table['col_table'] = [col[0] for col in table['column_names']]
        output_tab[db_name] = temp
        tables[db_name] = table

    for d in data:
        d['names'] = tables[d['db_id']]['schema_content']
        d['table_names'] = tables[d['db_id']]['table_names']
        d['col_set'] = tables[d['db_id']]['col_set']
        d['col_table'] = tables[d['db_id']]['col_table']
        keys = {}
        for kv in tables[d['db_id']]['foreign_keys']:
            keys[kv[0]] = kv[1]
            keys[kv[1]] = kv[0]
        for id_k in tables[d['db_id']]['primary_keys']:
            keys[id_k] = id_k
        d['keys'] = keys
    return data, tables


def find_table_of_star_column(sql, select):
    """
    Find table of column '*'
    """
    if len(sql['sql']['from']['table_units']) == 1:
        if sql['sql']['from']['table_units'][0][0] != 'sql':
            return sql['sql']['from']['table_units'][0][1]
        else:
            # here we select from a sub-query, therefore finding the "table" for the special column * is not so easy.
            # All the queries in spider with sub-queries do an aggregation over the "*", so it basically doesn't matter which one we choose.
            # To make it as correct as possible, we take the first table from the sub-query.
            # Example query: SELECT count(*) FROM (SELECT * FROM endowment WHERE amount  >  8.5 GROUP BY school_id HAVING count(*)  >  1)
            # print(sql['query'])
            return sql['sql']['from']['table_units'][0][1]['from']['table_units'][0][1]
    else:
        table_list = []
        for tmp_t in sql['sql']['from']['table_units']:
            if type(tmp_t[1]) == int:
                table_list.append(tmp_t[1])
        table_set, other_set = set(table_list), set()
        for sel_p in select:
            if sel_p[1][1][1] != 0:
                other_set.add(sql['col_table'][sel_p[1][1][1]])

        if len(sql['sql']['where']) == 1:
            other_set.add(sql['col_table'][sql['sql']['where'][0][2][1][1]])
        elif len(sql['sql']['where']) == 3:
            other_set.add(sql['col_table'][sql['sql']['where'][0][2][1][1]])
            other_set.add(sql['col_table'][sql['sql']['where'][2][2][1][1]])
        elif len(sql['sql']['where']) == 5:
            other_set.add(sql['col_table'][sql['sql']['where'][0][2][1][1]])
            other_set.add(sql['col_table'][sql['sql']['where'][2][2][1][1]])
            other_set.add(sql['col_table'][sql['sql']['where'][4][2][1][1]])
        table_set = table_set - other_set
        if len(table_set) == 1:
            return list(table_set)[0]
        elif len(table_set) == 0 and sql['sql']['groupBy'] != []:
            return sql['col_table'][sql['sql']['groupBy'][0][1]]
        else:
            question = sql['question']
            print('column * table error. Question: {}'.format(question))
            return sql['sql']['from']['table_units'][0][1]


def get_multi_token_match(question_tokens, idx, n_question_tokens, column_header_tokens):
    for endIdx in reversed(range(idx + 1, n_question_tokens + 1)):  # we go backwards though the remaining tokens (so from idx to the last word of the question)
        sub_tokens = question_tokens[idx: endIdx]
        if len(sub_tokens) > 1:   # and as long as we still have tokens
            sub_tokens = " ".join(sub_tokens)
            if sub_tokens in column_header_tokens:  # we check if this tokens (e.g. "artist song name") are a column header
                return endIdx, sub_tokens  # if yes, we return the new end-index and the sub-tokens, which will then be marked with "col"
    return idx, None


def get_single_token_match(question_tokens, idx, n_question_tokens, header_tokens):
    for endIdx in reversed(range(idx + 1, n_question_tokens + 1)):
        sub_tokens = question_tokens[idx: endIdx]
        sub_tokens = " ".join(sub_tokens)
        if sub_tokens in header_tokens:
            return endIdx, sub_tokens
    return idx, None


def get_partial_match(question_tokens, idx, header_tokens):
    """
    Try to find partial matches (e.g. for the question tokens "release year" and the column header "song release year")
    """
    def check_in(list_one, list_two):
        if len(set(list_one) & set(list_two)) == len(list_one) and (len(list_two) <= 3):
            return True

    for endIdx in reversed(range(idx + 1, len(question_tokens))):
        sub_toks = question_tokens[idx: min(endIdx, len(question_tokens))]
        if len(sub_toks) > 1:  # a match is defined by a minimum of 2 matching tokens
            flag_count = 0
            tmp_heads = None
            for heads in header_tokens:
                if check_in(sub_toks, heads):
                    flag_count += 1
                    tmp_heads = heads
            if flag_count == 1:
                return endIdx, tmp_heads
    return idx, None


def symbol_filter(questions):
    question_tmp_q = []
    for q_id, q_val in enumerate(questions):
        if len(q_val) > 2 and q_val[0] in ["'", '"', '`', '鈥�', '鈥�'] and q_val[-1] in ["'", '"', '`', '鈥�']:
            question_tmp_q.append("'")
            question_tmp_q += ["".join(q_val[1:-1])]
            question_tmp_q.append("'")
        elif len(q_val) > 2 and q_val[0] in ["'", '"', '`', '鈥�']:
            question_tmp_q.append("'")
            question_tmp_q += ["".join(q_val[1:])]
        elif len(q_val) > 2 and q_val[-1] in ["'", '"', '`', '鈥�']:
            question_tmp_q += ["".join(q_val[0:-1])]
            question_tmp_q.append("'")
        elif q_val in ["'", '"', '`', '鈥�', '鈥�', '``', "''"]:
            question_tmp_q += ["'"]
        else:
            question_tmp_q += [q_val]
    return question_tmp_q


def group_values(toks, idx, num_toks):
    def check_isupper(tok_lists):
        for tok_one in tok_lists:
            if tok_one[0].isupper() is False:
                return False
        return True

    for endIdx in reversed(range(idx + 1, num_toks + 1)):
        sub_toks = toks[idx: endIdx]

        if len(sub_toks) > 1 and check_isupper(sub_toks) is True:
            return endIdx, sub_toks
        if len(sub_toks) == 1:
            if sub_toks[0][0].isupper() and sub_toks[0].lower() not in VALUE_FILTER and \
                    sub_toks[0].lower().isalnum() is True:
                return endIdx, sub_toks
    return idx, None


def group_digital(toks, idx):
    test = toks[idx].replace(':', '')
    test = test.replace('.', '')
    if test.isdigit():
        return True
    else:
        return False


def group_symbol(toks, idx, num_toks):
    if toks[idx - 1] == "'":
        for i in range(0, min(3, num_toks - idx)):
            if toks[i + idx] == "'":
                return i + idx, toks[idx:i + idx]
    return idx, None


def set_header(toks, header_toks, tok_concol, idx, num_toks):
    def check_in(list_one, list_two):
        if set(list_one) == set(list_two):
            return True

    for endIdx in range(idx, num_toks):
        toks += tok_concol[endIdx]
        if len(tok_concol[endIdx]) > 1:
            break
        for heads in header_toks:
            if check_in(toks, heads):
                return heads
    return None


def re_lemma(string):
    lema = lemma(string.lower())
    if len(lema) > 0:
        return lema
    else:
        return string.lower()
