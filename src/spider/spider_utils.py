# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

# -*- coding: utf-8 -*-
"""
# @Time    : 2019/5/25
# @Author  : Jiaqi&Zecheng
# @File    : utils.py
# @Software: PyCharm
"""

import json
import random

import copy
import numpy as np
import os
import torch
from nltk.stem import WordNetLemmatizer
from src.spider.test_suite_eval.process_sql import get_schema, Schema, get_sql
from collections import defaultdict

wordnet_lemmatizer = WordNetLemmatizer()


def lower_keys(x):
    if isinstance(x, list):
        return [lower_keys(v) for v in x]
    elif isinstance(x, dict):
        return dict((k.lower(), lower_keys(v)) for k, v in x.items())
    else:
        return x


def get_table_colNames(tab_ids, tab_cols):
    table_col_dict = {}
    for ci, cv in zip(tab_ids, tab_cols):
        if ci != -1:
            table_col_dict[ci] = table_col_dict.get(ci, []) + cv
    result = []
    for ci in range(len(table_col_dict)):
        result.append(table_col_dict[ci])
    return result


def get_col_table_dict(tab_cols, tab_ids, sql):
    table_dict = {}
    for c_id, c_v in enumerate(sql['col_set']):
        for cor_id, cor_val in enumerate(tab_cols):
            if c_v == cor_val:
                table_dict[tab_ids[cor_id]] = table_dict.get(tab_ids[cor_id], []) + [c_id]

    col_table_dict = {}
    for key_item, value_item in table_dict.items():
        for value in value_item:
            col_table_dict[value] = col_table_dict.get(value, []) + [key_item]
    col_table_dict[0] = [x for x in range(len(table_dict) - 1)]
    return col_table_dict


def schema_linking(question_arg, question_arg_type, one_hot_type, col_set_type, col_set_iter, sql):
    for count_q, t_q in enumerate(question_arg_type):
        t = t_q[0]
        if t == 'NONE':
            continue
        elif t == 'table':
            one_hot_type[count_q][0] = 1
            question_arg[count_q] = ['table'] + question_arg[count_q]
        elif t == 'col':
            one_hot_type[count_q][1] = 1
            try:
                col_set_type[col_set_iter.index(question_arg[count_q])][1] = 5
                question_arg[count_q] = ['column'] + question_arg[count_q]
            except:
                print(col_set_iter, question_arg[count_q])
                raise RuntimeError("not in col set")
        elif t == 'agg':
            one_hot_type[count_q][2] = 1
        elif t == 'MORE':
            one_hot_type[count_q][3] = 1
        elif t == 'MOST':
            one_hot_type[count_q][4] = 1
        elif t == 'value':
            one_hot_type[count_q][5] = 1
            question_arg[count_q] = ['value'] + question_arg[count_q]
        else:
            if len(t_q) == 1:
                for col_probase in t_q:
                    if col_probase == 'asd':
                        continue
                    try:
                        col_set_type[sql['col_set'].index(col_probase)][2] = 5
                        question_arg[count_q] = ['value'] + question_arg[count_q]
                    except:
                        print(sql['col_set'], col_probase)
                        raise RuntimeError('not in col')
                    one_hot_type[count_q][5] = 1
            else:
                for col_probase in t_q:
                    if col_probase == 'asd':
                        continue
                    col_set_type[sql['col_set'].index(col_probase)][3] += 1


def load_data_new(sql_path, table_data, use_small=False):
    sql_data = []

    # sql_data basically is what we see in the original spider-data: https://github.com/taoyds/spider. it is though
    # already enriched with some information as e.g. POS (stanford_pos and nltk_pos) and NER (stanford_ner) The most
    # complex field is the sql-dict, which contains the structured sql similar to the "sql" attribute in spider For
    # more details on this structure see the example in
    # https://github.com/taoyds/spider/blob/master/preprocess/parsed_sql_examples.sql

    with open(sql_path, encoding='utf-8') as inf:
        data = json.load(inf)
        # resize before lower_keys() to reduce computation effort
        if use_small:
            data = data[:80]
        data = lower_keys(data)
        sql_data += data

    print("Load data from {}. N={}".format(sql_path, len(sql_data)))

    table_dict = {table['db_id']: table for table in table_data}

    return sql_data, table_dict


def load_dataset(dataset_dir, use_small=False, train_db_id=None):
    """

    @param dataset_dir: the directory of the dataset
    @param use_small: if True, only use the first 10 samples
    @param train_db_id: If we only train on a single db
    @return: a list of sql_data, a list of table_dict
    """
    print("Loading from datasets...")

    table_path = os.path.join(dataset_dir, "original", "tables.json")
    train_path = os.path.join(dataset_dir, "train.json")
    dev_path = os.path.join(dataset_dir, "dev.json")
    with open(table_path, encoding='utf-8') as inf:
        # table_data is basically a dict with all the 200 (in train ca. 166) datasets of spider.
        # Each sub-dict contains the name of all tables, as well as relations between them (foreign keys, primary keys)
        table_data = json.load(inf)
        print("Load data from {}. N={}".format(table_path, len(table_data)))

    train_sql_data, train_table_data = load_data_new(train_path, table_data, use_small=use_small)
    val_sql_data, val_table_data = load_data_new(dev_path, table_data, use_small=use_small)
    if train_db_id is not None:
        train_sql_data = [x for x in train_sql_data if x['db_id'] == train_db_id]
    return train_sql_data, train_table_data, val_sql_data, val_table_data


def negative_sampling_augmentation(sql_data, aug_num=1):
    augmented_dataset = []
    db_id_to_sql_data = defaultdict(lambda: [])
    for data_row in sql_data:
        db_id_to_sql_data[data_row['db_id']].append(data_row)

    for data_row in sql_data:
        negative_samples = random.sample(db_id_to_sql_data[data_row['db_id']], aug_num)
        data_row['label'] = 1
        augmented_dataset.append(data_row)
        for i in range(aug_num):
            new_data_row = copy.deepcopy(data_row)
            new_data_row['label'] = 0
            new_data_row['question'] = negative_samples[i]['question']
            new_data_row['question_toks'] = negative_samples[i]['question_toks']
            new_data_row['question_arg'] = negative_samples[i]['question_arg']
            new_data_row['question_arg_type'] = negative_samples[i]['question_arg_type']
            new_data_row['values'] = negative_samples[i]['values']
            augmented_dataset.append(new_data_row)
    print("Augmented dataset size: {}".format(len(augmented_dataset)))
    return augmented_dataset


def load_schema(schema_path):

    with open(schema_path, encoding='utf-8') as inf:
        # table_data is basically a dict with all the 200 (in train ca. 166) datasets of spider.
        # Each sub-dict contains the name of all tables, as well as relations between them (foreign keys, primary keys)
        table_data = json.load(inf)
        print("Load data from {}. N={}".format(schema_path, len(table_data)))

    table_dict = {table['db_id']: table for table in table_data}
    return table_data, table_dict


def lower_case_info(table_dict):
    for db_id, data in table_dict.items():
        data['column_names_original'] = [[x[0], x[1].lower()] for x in data['column_names_original']]
        data['table_names_original'] = [x.lower() for x in data['table_names_original']]

def load_all_schema_data(db_dir, db_names):
    db_names_to_schema = {}
    for db_name in db_names:
        s = Schema(get_schema(os.path.join(db_dir, db_name, db_name + ".sqlite")))
        db_names_to_schema[db_name] = s
    return db_names_to_schema