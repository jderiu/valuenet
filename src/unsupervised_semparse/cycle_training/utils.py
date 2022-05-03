import nltk
from src.named_entity_recognition.local_ner.spacy_ner import local_named_entity_recognition
from src.preprocessing.pre_process import lookup_database, add_full_column_match, lemmatize_list
from src.preprocessing.utils import wordnet_lemmatizer, symbol_filter, get_multi_token_match, \
    get_single_token_match, get_partial_match, AGG


def postprocess_text(preds, labels):
    preds = [pred.strip() if len(pred.strip()) > 0 else 'What?' for pred in preds]
    labels = [[label.strip()] for label in labels]

    return preds, labels

def pre_process_simple(
        question,
        question_toks,
        table_names,
        col_set,
        ner_information,
        db_value_finder,
        is_training
):
    question_tokens = [wordnet_lemmatizer.lemmatize(x.lower()) for x in symbol_filter(question_toks)]

    tables, table_list = lemmatize_list(table_names)

    columns, columns_list = lemmatize_list(col_set)

    pos_tagging = nltk.pos_tag(question_tokens)

    token_grouped = []
    token_types = []

    # this will contain what we call the "column hints" --> information how often a column has been "hit" in a question
    column_matches = [{"column_joined": '',
                       "full_column_match": False,
                       "partial_column_match": 0,
                       "full_value_match": False,
                       "partial_value_match": 0} for _ in columns]

    n_tokens = len(question_tokens)

    idx = 0
    while idx < len(question_tokens):

        # checking if we find a full column header with more than one token (e.g. "song name")
        end_idx, multi_token_column_name = get_multi_token_match(question_tokens, idx, n_tokens, columns)
        if multi_token_column_name:
            token_grouped.append(question_tokens[idx: end_idx])
            token_types.append(["col"])
            idx = end_idx

            add_full_column_match(multi_token_column_name, columns, column_matches)
            continue

        # check for table
        end_idx, table_name = get_single_token_match(question_tokens, idx, n_tokens, tables)
        if table_name:
            token_grouped.append(question_tokens[idx: end_idx])
            token_types.append(["table"])
            idx = end_idx
            continue

        # check for column
        end_idx, column_name = get_single_token_match(question_tokens, idx, n_tokens, columns)
        if column_name:
            token_grouped.append(question_tokens[idx: end_idx])
            token_types.append(["col"])
            idx = end_idx

            add_full_column_match(column_name, columns, column_matches)
            continue

        # check for partial column matches (min 2 tokens need to match)
        end_idx, column_name_extended = get_partial_match(question_tokens, idx, columns_list)
        if column_name_extended:
            token_grouped.append(column_name_extended)
            token_types.append(["col"])
            idx = end_idx

            add_full_column_match(column_name_extended, columns_list, column_matches)
            continue

        # check for aggregation
        end_idx, agg = get_single_token_match(question_tokens, idx, n_tokens, AGG)  # check the AGG - it's basically looking for words like "average, maximum, minimum" etc.
        if agg:
            token_grouped.append(question_tokens[idx: end_idx])
            token_types.append(["agg"])
            idx = end_idx
            continue

        if pos_tagging[idx][1] == 'RBR' or pos_tagging[idx][1] == 'JJR':  # with the help of NLTK part of speech we are
            token_grouped.append([question_tokens[idx]])  # looking for comparative words like "better", "bigger"
            token_types.append(['MORE'])
            idx += 1
            continue

        if pos_tagging[idx][1] == 'RBS' or pos_tagging[idx][1] == 'JJS':  # with the help of NLTK part of speech we are
            token_grouped.append([question_tokens[idx]])  # looking for superlative words like "best", "biggest"
            token_types.append(['MOST'])
            idx += 1
            continue

        token_grouped.append([question_tokens[idx]])
        token_types.append(['NONE'])
        idx += 1
        continue

    # This extra-loop is a bit special: we gather partial column matches by going through the question tokens and columns and finding partial matches.
    # Full matches should already have been found further up in the loop.
    # TODO not sure if that's a good thing, and even if, there might be room for improvement (e.g. a match when Table and Column has been hit).
    for column_idx, column in enumerate(columns_list):
        column_matches[column_idx]['column_joined'] = str(column)

        for question_idx, question_token in enumerate(question_tokens):
            if question_token in column:
                # if we have a match between a partial column token (e.g. "horse id") and a token in the question (e.g. "horse")
                # we will increase the counter for this column
                column_matches[column_idx]['partial_column_match'] += 1

    # a lot of interesting stuff happens here - make sure you are aware of it!
    example = {
        'question': question,
        'question_toks': question_toks,
    }
    value_candidates, all_values_found, column_matches = lookup_database(example, ner_information, columns,
                                                                         question_tokens, column_matches,
                                                                         db_value_finder,
                                                                         add_values_from_ground_truth=is_training)

    return value_candidates


def get_values(question,question_toks, table_names, col_set, db_value_finder):
    ner_information = local_named_entity_recognition(question)
    value_candidates = pre_process_simple(
        question,
        question_toks,
        table_names,
        col_set,
        ner_information,
        db_value_finder,
        False
    )
    return value_candidates