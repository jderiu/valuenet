import torch
import random
import wandb
from tqdm import tqdm
import copy
from src.manual_inference.helper import _semql_to_sql
from src.spider.test_suite_eval.evaluation import match_evaluation_single
from src.spider.test_suite_eval.process_sql import tokenize
from spider.example_builder import build_example


def train(global_step,
          train_dataloader,
          schema,
          model,
          optimizer,
          clip_grad,
          sketch_loss_weight=1,
          lf_loss_weight=1):
    tr_loss = 0.0
    model.zero_grad()
    model.train()

    for step, batch in enumerate(tqdm(train_dataloader, desc="Training")):
        examples = []
        for data_row in batch:
            try:
                example = build_example(data_row, schema)
                examples.append(example)
            except RuntimeError as e:
                print("Exception while building example (training): {}".format(e))

        examples.sort(key=lambda e: -len(e.question_tokens))

        sketch_loss, lf_loss = model.forward(examples)

        mean_sketch_loss = torch.mean(-sketch_loss)
        mean_lf_loss = torch.mean(-lf_loss)

        loss = lf_loss_weight * mean_lf_loss + sketch_loss_weight * mean_sketch_loss

        wandb.log(
            {
                'train/loss': float(loss),
                'train/mean_lf_loss': float(mean_lf_loss),
                'train/mean_sketch_loss': float(mean_sketch_loss),
            }
        )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)

        tr_loss += loss.item()

        optimizer.step()
        model.zero_grad()  # after we optimized the weights, we set the gradient back to zero.

        global_step += 1

    return global_step


def pretrain_decoder(global_step,
          train_dataloader,
          schema,
          model,
          optimizer,
          clip_grad,
          sketch_loss_weight=1,
          lf_loss_weight=1):
    tr_loss = 0.0
    model.zero_grad()
    model.train()

    for step, batch in enumerate(tqdm(train_dataloader, desc="Training")):
        examples = []
        for data_row in batch:
            try:
                example = build_example(data_row, schema)
                # remove text here--
                example.question_tokens = [[tok] for tok in data_row['query_toks']]
                examples.append(example)
            except RuntimeError as e:
                print("Exception while building example (training): {}".format(e))

        examples.sort(key=lambda e: -len(e.question_tokens))

        sketch_loss, lf_loss = model.forward(examples)

        mean_sketch_loss = torch.mean(-sketch_loss)
        mean_lf_loss = torch.mean(-lf_loss)

        loss = lf_loss_weight * mean_lf_loss + sketch_loss_weight * mean_sketch_loss

        wandb.log(
            {
                'train/loss': float(loss),
                'train/mean_lf_loss': float(mean_lf_loss),
                'train/mean_sketch_loss': float(mean_sketch_loss),
            }
        )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)

        tr_loss += loss.item()

        optimizer.step()
        model.zero_grad()  # after we optimized the weights, we set the gradient back to zero.

        global_step += 1

    return global_step


def train_step_actor_critic(
        data_row,
        schema,
        kmaps,
        db_names_to_schema,
        model,
        optimizer,
        val_optimizer,
        clip_grad,
        sketch_loss_weight=1,
        lf_loss_weight=1
):
    original_row = copy.deepcopy(data_row)
    try:
        example = build_example(data_row, schema)
    except Exception as e:
        print("Exception while building example (evaluation): {}".format(e))
        return None, None, -1

    with torch.no_grad():
        for _ in range(15):
            results = model.parse(example, beam_size=15)
            if len(results[1]) > 0:
                break
    if len(results[1]) == 0:
        return None, None, -1

    update_example = copy.deepcopy(example)
    value_example = copy.deepcopy(example)

    if len(results[0]) > 0:
        rand_result = random.choice(results[0])
        full_prediction = " ".join([str(x) for x in rand_result.actions])
        original_row['model_result'] = full_prediction
        try:
            sql = _semql_to_sql(original_row, schema).replace('"', '')
            db_name = original_row['db_id']
            eval_results = match_evaluation_single(
                original_row['query'],
                sql,
                db_name,
                db_names_to_schema[db_name],
                kmaps
            )
            #reward = -1.0 if eval_results['exact'] == 0 else 1.0
            reward = sum([v['acc'] for k, v in eval_results['partial'].items()])/len(eval_results['partial'])
            reward = 2*reward - 1
            #reward = reward - 1.0 if reward < 1.0 else 1.0
        except Exception as e:
            sql = ''
            reward = -1.0

        update_example.semql_actions = results[0][0].actions
        update_example.sketch = results[1]
        update_example.sql = sql

        value_example.question_tokens = example.question_tokens + [[model.encoder.tokenizer.sep_token]] + [[tok] for tok in tokenize(sql)]
    else:
        update_example.semql_actions = []
        update_example.sketch = results[1]
        update_example.sql = ''
        value_example.question_tokens = value_example.question_tokens + [['']]
        reward = -1.0

    #value_funct = model.value_function_forward([update_example])
    #val_loss = (value_funct - reward).pow(2).mean()

    # val_optimizer.zero_grad()
    # val_loss.backward()
    # torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
    # val_optimizer.step()

    sketch_loss, lf_loss = model.forward([update_example])
    mean_sketch_loss = torch.mean(-sketch_loss)
    mean_lf_loss = torch.mean(-lf_loss)
    loss = lf_loss_weight * mean_lf_loss + sketch_loss_weight * mean_sketch_loss

    #ac_loss = loss*(reward - float(value_funct))
    ac_loss = loss*reward
    optimizer.zero_grad()
    ac_loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
    optimizer.step()

    return float(ac_loss), float(loss), float(reward)
