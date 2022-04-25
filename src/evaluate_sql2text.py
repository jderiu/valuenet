import json
import os.path

import torch

from transformers import AutoTokenizer
from src.model.encoder_decoder.modeling_encoder_decoder import EncoderDecoderModel
from transformers.models.gpt2.modeling_gpt2 import GPT2LMHeadModel
from src.intermediate_representation.sem2sql.sem2SQL import transform_semQL_to_sql
from datasets import load_metric
from src.config import read_arguments_evaluation
from src.intermediate_representation import semQL
from src.spider import spider_utils
from src.utils import setup_device, set_seed_everywhere
from src.model.sql2text_data import DataCollatorForSQL2Text, DataCollartorForLMSQL2Text, DataCollatorCycle
from tqdm import tqdm
from src.model.model import IRNet
import src.spider.test_suite_eval.evaluation as spider_evaluation

metric = load_metric("sacrebleu")


def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [[label.strip()] for label in labels]

    return preds, labels


def batch_list(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]


def cycle_eval(
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
    for batch, pred_text_batch in tqdm(batch_list(data, args.batch_size), total=n_eval_steps):
        examples, original_rows = data_collator(batch, pred_text_batch)
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
    return float(sketch_correct) / float(total), float(rule_label_correct) / float(total), float(not_all_values_found) / float(total), predictions


def evaluate_encode_decode(
        args,
        test_data,
        data_collator,
        model,
        tokenizer,
        logging_path,
        checkpoint_nr=0
):
    out_labels, out_preds = [], []
    n_eval_steps = int(len(test_data) // args.batch_size) + 1
    for batch in tqdm(batch_list(test_data, args.batch_size), total=n_eval_steps):
        preprocessed_batch = data_collator(batch)
        with torch.no_grad():
            generated_out = model.generate(
                preprocessed_batch['input_ids'],
                attention_mask=preprocessed_batch['attention_mask'],
                max_length=32,
                num_beams=15,
                repetition_penalty=2.5,
                no_repeat_ngram_size=3,
                pad_token_id=tokenizer.pad_token_id,
            )
        pred_batch_out = tokenizer.batch_decode(generated_out, skip_special_tokens=True)
        labels = [x['question'] for x in batch]
        out_labels.extend(labels)
        out_preds.extend(pred_batch_out)

    decoded_preds, decoded_labels = postprocess_text(out_preds, out_labels)
    result = metric.compute(predictions=decoded_preds, references=decoded_labels)
    result = {"bleu": result["score"]}

    with open(os.path.join(logging_path, f'results_final_{checkpoint_nr}.txt'), 'wt', encoding='utf-8') as f:
        f.write(f'BLEU: {result["bleu"]}\n')
        for pred, label in zip(decoded_preds, decoded_labels):
            f.write(f"{pred}\t{label[0]}\n")
    return decoded_preds, [x[0] for x in decoded_labels]


def evaulate_decode_only(
        args,
        test_data,
        data_collator,
        model,
        tokenizer,
        logging_path,
        checkpoint_nr=0

):
    out_labels, out_preds = [], []
    n_eval_steps = int(len(test_data) // args.batch_size) + 1
    for batch in tqdm(batch_list(test_data, args.batch_size), total=n_eval_steps):
        preprocessed_batch = data_collator(batch, is_eval=True)
        with torch.no_grad():
            generated_out = model.generate(
                preprocessed_batch['input_ids'],
                attention_mask=preprocessed_batch['attention_mask'],
                max_length=preprocessed_batch['input_ids'].shape[1] + 32,
                num_beams=15,
                repetition_penalty=2.5,
                no_repeat_ngram_size=3,
                pad_token_id=tokenizer.pad_token_id,
            )
        decoded_out = tokenizer.batch_decode(generated_out, skip_special_tokens=True)
        pred_batch_out = [x.split('TEXT:')[1].replace('\n', '') for x in decoded_out]
        labels = [x['question'] for x in batch]

        out_labels.extend(labels)
        out_preds.extend(pred_batch_out)
    decoded_preds, decoded_labels = postprocess_text(out_preds, out_labels)
    result = metric.compute(predictions=decoded_preds, references=decoded_labels)
    result = {"bleu": result["score"]}

    with open(os.path.join(logging_path, f'results_final_{checkpoint_nr}.txt'), 'wt', encoding='utf-8') as f:
        f.write(f'BLEU: {result["bleu"]}\n')
        for pred, label in zip(decoded_preds, decoded_labels):
            f.write(f"{pred}\t{label[0]}\n")
    return decoded_preds, [x[0] for x in decoded_labels]


def main():
    args = read_arguments_evaluation()

    device, n_gpu = setup_device()
    device = 'cpu'
    set_seed_everywhere(args.seed, n_gpu)

    sql_data, table_data, val_sql_data, val_table_data = spider_utils.load_dataset(args.data_dir, use_small=True)
    grammar = semQL.Grammar()

    print("Loading pre-trained model from '{}'".format(args.model_to_load))
    with open(os.path.join(args.model_to_load, "args.json"), "rt", encoding='utf-8') as f:
        train_args = json.load(f)

    if train_args['gen_type'] == 'encoder_decoder':
        encoder_tokenizer = AutoTokenizer.from_pretrained(train_args['encoder_pretrained_model'], add_prefix_space=True)
        decoder_tokenizer = AutoTokenizer.from_pretrained('gpt2')
        if decoder_tokenizer.pad_token_id is None:
            decoder_tokenizer.pad_token = decoder_tokenizer.bos_token
        model = EncoderDecoderModel.from_pretrained(os.path.join(args.model_to_load, f'checkpoint-{args.checkpoint}'))
        model.to(device)
        pytorch_total_params = sum(p.numel() for p in model.parameters())
        print(f'Number of Params: {pytorch_total_params}!')

        data_collator = DataCollatorForSQL2Text(
            encoder_tokenizer=encoder_tokenizer,
            decoder_tokenizer=decoder_tokenizer,
            model=model,
            grammar=grammar,
            schema=table_data,
            device=device
        )

        decoded_preds, decoded_labels = evaluate_encode_decode(
            args,
            val_sql_data,
            data_collator,
            model,
            decoder_tokenizer,
            args.model_to_load,
            args.checkpoint
        )

    else:
        decoder_tokenizer = AutoTokenizer.from_pretrained(train_args['decoder_pretrained_model'], add_prefix_space=True)
        decoder_tokenizer.padding_side = 'left'
        if decoder_tokenizer.pad_token_id is None:
            decoder_tokenizer.pad_token = decoder_tokenizer.bos_token
        if decoder_tokenizer.sep_token_id is None:
            decoder_tokenizer.sep_token = decoder_tokenizer.bos_token
        # model = GPT2LMHeadModel.from_pretrained('gpt2')
        model = GPT2LMHeadModel.from_pretrained(os.path.join(args.model_to_load, f'checkpoint-{args.checkpoint}'))
        model.to(device)
        pytorch_total_params = sum(p.numel() for p in model.parameters())
        print(f'Number of Params: {pytorch_total_params}!')

        data_collator = DataCollartorForLMSQL2Text(
            tokenizer=decoder_tokenizer,
            model=model,
            grammar=grammar,
            schema=table_data,
            device=device
        )

        decoded_preds, decoded_labels = evaulate_decode_only(
            args,
            val_sql_data,
            data_collator,
            model,
            decoder_tokenizer,
            args.model_to_load,
            args.checkpoint
        )

    # do cycle consistency evaluation
    model = IRNet(args, device, grammar)
    model.to(device)
    model.load_state_dict(torch.load(args.ir_model_to_load))

    data_collator = DataCollatorCycle(
        grammar,
        table_data,
        device
    )

    sketch_acc, acc, not_all_values_found, predictions = cycle_eval(
        args,
        val_sql_data,
        None,
        data_collator,
        model
    )
    print( "Predicted {} examples. Start now converting them to SQL. Sketch-Accuracy: {}, Accuracy: {}, Not all values found: {}".format(
            len(val_sql_data), sketch_acc, acc, not_all_values_found))

    with open(os.path.join(args.prediction_dir, 'predictions_sem_ql.json'), 'w', encoding='utf-8') as f:
        json.dump(predictions, f, indent=2)

    count_success, count_failed = transform_semQL_to_sql(val_table_data, predictions, args.prediction_dir)

    spider_evaluation.evaluate(
        os.path.join(args.prediction_dir, 'ground_truth.txt'),
        os.path.join(args.prediction_dir, 'output.txt'),
        os.path.join(args.data_dir, "testsuite_databases"),
        'exec', None, False, False, False, 1, quickmode=False)

    sketch_acc, acc, not_all_values_found, predictions = cycle_eval(
        args,
        val_sql_data,
        decoded_preds,
        data_collator,
        model
    )
    print("Predicted {} examples. Start now converting them to SQL. Sketch-Accuracy: {}, Accuracy: {}, Not all values found: {}".format(
            len(val_sql_data), sketch_acc, acc, not_all_values_found))

    with open(os.path.join(args.prediction_dir, 'predictions_sem_ql_from_preds.json'), 'w', encoding='utf-8') as f:
        json.dump(predictions, f, indent=2)

    count_success, count_failed = transform_semQL_to_sql(val_table_data, predictions, args.prediction_dir)

    spider_evaluation.evaluate(
        os.path.join(args.prediction_dir, 'ground_truth.txt'),
        os.path.join(args.prediction_dir, 'output.txt'),
        os.path.join(args.data_dir, "testsuite_databases"),
        'exec', None, False, False, False, 1, quickmode=False)


if __name__ == '__main__':
    main()
