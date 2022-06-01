import copy

import json, os, torch
from datasets import load_metric
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm

from src.utils import setup_device
from src.config import read_arguments_evaluation
from src.intermediate_representation import semQL
from src.model.model import IRNet
from src.model.sql2text_data import DataCollatorCycle
from src.spider import spider_utils
from src.intermediate_representation.sem2sql.sem2SQL import transform_semQL_to_sql
import src.spider.test_suite_eval.evaluation as spider_evaluation

metric = load_metric("sacrebleu")


def postprocess_text(preds, labels):
    preds = [pred.strip() if len(pred.strip()) > 0 else 'What?' for pred in preds]
    n_ref = max([len(x) for x in labels])
    for label in labels:
        llen = len(label)
        diff = n_ref - llen
        for i in range(diff):
            label.append('What?')

    return preds, labels


def compute_bleu(in_json):
    preds = [x['synthetic_answer'] for x in in_json]
    labels = [x['questions'] for x in in_json]

    preds, labels = postprocess_text(preds, labels)
    result = metric.compute(predictions=preds, references=labels)
    return result['score']


def compute_sem_sim(in_json):
    similarity_model = SentenceTransformer('all-MiniLM-L6-v2')

    preds = [x['synthetic_answer'] for x in in_json]
    labels = [x['questions'] for x in in_json]
    n_refs = max([len(ref) for ref in labels])

    flat_labels = [item for sublist in labels for item in sublist]

    embeddings_preds = similarity_model.encode(preds, convert_to_tensor=True)
    embeddings_labels = similarity_model.encode(flat_labels, convert_to_tensor=True)

    cosine_scores = util.cos_sim(embeddings_preds, embeddings_labels)
    sim_scores = []
    for i, cosine_score in enumerate(cosine_scores):
        cands = cosine_score[n_refs*i:n_refs*i+n_refs].max()
        sim_scores.append(float(cands))
    avg_sem_sim = sum(sim_scores) / len(sim_scores)
    return avg_sem_sim


def batch_list(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]


def predict_sql_from_text(
        args,
        test_data,
        pred_texts,
        data_collator,
        model
):
    sketch_correct, rule_label_correct, found_in_beams, not_all_values_found, total = 0, 0, 0, 0, 0
    n_eval_steps = int(len(test_data) // args.batch_size) + 1
    if pred_texts is None:
        pred_texts = [None for _ in range(len(test_data))]
    data = list(zip(test_data, pred_texts))
    predictions = []
    for batch in tqdm(batch_list(data, args.batch_size), total=n_eval_steps):
        d_batch = [x[0] for x in batch]
        pred_text_batch = [x[1] for x in batch]
        examples, original_rows = data_collator(d_batch, pred_text_batch)
        for example, original_row in zip(examples, original_rows):
            with torch.no_grad():
                results_all = model.parse(example, beam_size=args.beam_size)
            results = results_all[0]
            all_predictions = []
            try:
                # here we set assemble the predicted actions (including leaf-nodes) as string
                full_prediction = " ".join([str(x) for x in results[0].actions])
                for beam in results:
                    all_predictions.append(" ".join([str(x) for x in beam.actions]))
            except Exception as e:
                # print(e)
                full_prediction = ""

            prediction = original_row
            prediction['sketch_result'] = " ".join(str(x) for x in results_all[1])
            prediction['model_result'] = full_prediction

            truth_sketch = " ".join([str(x) for x in example.sketch])
            truth_rule_label = " ".join([str(x) for x in example.semql_actions])

            if prediction['all_values_found']:
                if truth_sketch == prediction['sketch_result']:
                    sketch_correct += 1
                if truth_rule_label == prediction['model_result']:
                    rule_label_correct += 1
                elif truth_rule_label in all_predictions:
                    found_in_beams += 1
            else:
                question = prediction['question']
                tqdm.write(
                    f'Not all values found during pre-processing for question "{question}". Replace values with dummy to make query fail')
                prediction['values'] = [1] * len(prediction['values'])
                not_all_values_found += 1

            total += 1

            predictions.append(prediction)
    return float(sketch_correct) / float(total), float(rule_label_correct) / float(total), float(
        not_all_values_found) / float(total), predictions


def cycle_eval(args, in_json):
    device, n_gpu = setup_device()
    grammar = semQL.Grammar()
    decoded_preds = [x['synthetic_answer'] for x in in_json]
    _, table_data, val_sql_data, val_table_data = spider_utils.load_dataset(args.data_dir, use_small=args.toy)
    sql_to_dp = {dp['query']: copy.deepcopy(dp)  for dp in val_sql_data}

    red_val_sql_data, red_in_json= [], []
    for entry in in_json:
        if sql_to_dp.get(entry['query']) is None:
            print(entry['query'])
            continue
        dp = sql_to_dp[entry['query']]
        red_val_sql_data.append(dp)
        red_in_json.append(entry)

    decoded_preds = [x['synthetic_answer'] for x in red_in_json]
    model = IRNet(args, device, grammar)
    model.to(device)
    model.load_state_dict(torch.load(args.ir_model_to_load), strict=False)

    data_collator = DataCollatorCycle(
        grammar,
        table_data,
        device
    )

    sketch_acc, acc, not_all_values_found, predictions = predict_sql_from_text(
        args,
        val_sql_data,
        None,
        data_collator,
        model
    )

    with open(os.path.join(args.prediction_dir, 'predictions_sem_ql.json'), 'w', encoding='utf-8') as f:
        json.dump(predictions, f, indent=2)

    transform_semQL_to_sql(val_table_data, predictions, args.prediction_dir)

    exec_score_gold, scores_list_gold = spider_evaluation.evaluate(
        os.path.join(args.prediction_dir, 'ground_truth.txt'),
        os.path.join(args.prediction_dir, 'output.txt'),
        os.path.join(args.data_dir, "testsuite_databases"),
        'exec', None, False, False, False, 1, quickmode=False, log_wandb=False)

    sketch_acc_preds, acc_preds, _, predictions_preds = predict_sql_from_text(
        args,
        red_val_sql_data,
        decoded_preds,
        data_collator,
        model
    )

    with open(os.path.join(args.prediction_dir, 'predictions_sem_ql_from_preds.json'), 'w', encoding='utf-8') as f:
        json.dump(predictions_preds, f, indent=2)

    transform_semQL_to_sql(val_table_data, predictions_preds, args.prediction_dir, ofname='output_from_preds.txt')

    exec_score_pred, scores_list_pred = spider_evaluation.evaluate(
        os.path.join(args.prediction_dir, 'ground_truth.txt'),
        os.path.join(args.prediction_dir, 'output_from_preds.txt'),
        os.path.join(args.data_dir, "testsuite_databases"),
        'exec', None, False, False, False, 1, quickmode=False, log_wandb=False)

    total, match = 0, 0
    for ip, ig in zip(scores_list_pred['exec'], scores_list_gold['exec']):
        if ig == 1:
            total += 1
            if ip == 1:
                match += 1
    precision = float(match) / float(total)

    return exec_score_pred, exec_score_gold, precision, acc, acc_preds


def main():
    args = read_arguments_evaluation()
    logging_path = args.model_to_load
    checkpoint_nr = args.checkpoint

    with open(os.path.join(logging_path, f'out_final_{checkpoint_nr}.json'), 'rt', encoding='utf-8') as f2:
        in_json = json.load(f2)

    bleu_score = compute_bleu(in_json)
    print(f'BLEU score: {bleu_score}')
    sem_sim_score = compute_sem_sim(in_json)
    print(f'Semantic Similarity Score: {sem_sim_score}')
    exec_score_pred, exec_score_gold, precision, acc, acc_preds = cycle_eval(args, in_json)
    print(f'Execution score: {exec_score_pred}')
    print(f'Execution score (gold): {exec_score_gold}')
    print(f'Precision: {precision}')
    print(f'Accuracy: {acc}')
    print(f'Accuracy (preds): {acc_preds}')


if __name__ == '__main__':
    main()
