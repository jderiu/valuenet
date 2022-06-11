import json, os

from src.config import read_arguments_evaluation
from src.spider.test_suite_eval.exec_eval import eval_exec_match

def main():
    args = read_arguments_evaluation()
    logging_path = args.model_to_load

    pred_picard_base_sql_to_dp = {}
    with open(os.path.join(logging_path, f'pred_picard_base.json'), 'rt', encoding='utf-8') as f2:
        pred_picard_base = json.load(f2)
        for dp in pred_picard_base:
            pred_picard_base_sql_to_dp[dp['query']] = dp

    predictions_picard_sql_to_dp = {}
    with open(os.path.join(logging_path, f'predictions_picard.json'), 'rt', encoding='utf-8') as f2:
        predictions_picard = json.load(f2)
        for dp in predictions_picard:
            predictions_picard_sql_to_dp[dp['query']] = dp

    db_dir = 'data/spider/testsuite_databases'

    sql_to_base_score = {}
    for gold_query, base_dp in pred_picard_base_sql_to_dp.items():
        pred_query = base_dp['prediction'].split('|')[1].strip()
        db_id = base_dp['db_id']
        db = os.path.join(db_dir, db_id, db_id + ".sqlite")
        exec_score = eval_exec_match(db=db, p_str=pred_query, g_str=gold_query, plug_value=False,
                                     keep_distinct=False,
                                     progress_bar_for_each_datapoint=False,
                                     quickmode=False)
        sql_to_base_score[gold_query] = exec_score

    sql_to_pred_score = {}
    for gold_query, dp in predictions_picard_sql_to_dp.items():
        pred_query = dp['prediction'].split('|')[1].strip()
        db_id = dp['db_id']
        db = os.path.join(db_dir, db_id, db_id + ".sqlite")
        exec_score = eval_exec_match(db=db, p_str=pred_query, g_str=gold_query, plug_value=False,
                                     keep_distinct=False,
                                     progress_bar_for_each_datapoint=False,
                                     quickmode=False)
        sql_to_pred_score[gold_query] = exec_score

    precision_count = 0
    total_base_count = 0
    total_pred_count = 0

    for query, base_score in sql_to_base_score.items():
        pred_score = sql_to_pred_score.get(query)
        if pred_score is None:
            print('Not Found: ', query)
            continue
        total_base_count += base_score
        total_pred_count += pred_score
        if base_score == 1:
            precision_count += pred_score

    base_acc = total_base_count/len(sql_to_base_score)
    pred_acc = total_pred_count/len(sql_to_base_score)
    ratio = pred_acc/base_acc
    print(f'Precision: {precision_count/total_base_count}')
    print(f'Ratio: {ratio}')
    print(f'Base Acc: {base_acc}')
    print(f'Pred Acc: {pred_acc}')


if __name__ == '__main__':
    main()