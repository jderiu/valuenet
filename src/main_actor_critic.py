import json
import os

import torch
from pytictoc import TicToc
from tqdm import tqdm

from src.model.model import IRNet
from src.config import read_arguments_train, write_config_to_file
from src.data_loader import get_random_sampler, get_data_loader
from src.evaluation import evaluate, transform_to_sql_and_evaluate_with_spider
from src.intermediate_representation import semQL
from src.optimizer import build_optimizer_encoder
from src.spider import spider_utils
from src.training import train_step_actor_critic, pretrain_decoder
from src.utils import setup_device, set_seed_everywhere, save_model, create_experiment_folder

from src.spider.test_suite_eval.evaluation import build_foreign_key_map_from_json
# initialize experiment tracking @ Weights & Biases
import wandb

wandb.init(project="proton")

if __name__ == '__main__':
    args = read_arguments_train()

    # log hyperparameters to Weights & Biases
    wandb.config.update(args)

    experiment_name, output_path = create_experiment_folder(args.model_output_dir, args.exp_name)
    print("Run experiment '{}'".format(experiment_name))

    write_config_to_file(args, output_path)

    device, n_gpu = setup_device()
    set_seed_everywhere(args.seed, n_gpu)

    sql_data, table_data, val_sql_data, val_table_data = spider_utils.load_dataset(args.data_dir, use_small=args.toy)
    db_names_to_schema = spider_utils.load_all_schema_data(os.path.join(args.data_dir, 'testsuite_databases'), list(table_data.keys()))
    pretrain_data_loader, _ = get_data_loader(sql_data, val_sql_data, args.batch_size, True, False)

    train_loader, dev_loader = get_random_sampler(
        sql_data,
        val_sql_data,
        batch_size=1,
        db_names_to_schema=db_names_to_schema,
        n_boxes=5
    )

    grammar = semQL.Grammar()
    model = IRNet(args, device, grammar)
    model.to(device)
    if args.model_to_load is not None:
        model.load_state_dict(torch.load(args.model_to_load), strict=False)
        print("Load pre-trained model from '{}'".format(args.model_to_load))

    # track the model
    wandb.watch(model, log='parameters')

    kmaps = build_foreign_key_map_from_json(os.path.join(args.data_dir, "original", 'tables.json'))

    num_train_steps = len(train_loader) * args.num_epochs
    optimizer, scheduler = build_optimizer_encoder(model,
                                                   num_train_steps,
                                                   args.lr_transformer, args.lr_connection, args.lr_base,
                                                   args.scheduler_gamma)

    val_optimizer, val_scheduler = build_optimizer_encoder(model,
                                                   num_train_steps,
                                                   args.lr_transformer, args.lr_connection, args.lr_base,
                                                   args.scheduler_gamma)

    global_step = 0
    best_acc = 0.0
    pretrain_epochs = args.pretrain_epochs
    for epoch in tqdm(range(pretrain_epochs), desc='Pretrain Decoder'):
        sketch_loss_weight = 1 if epoch < args.loss_epoch_threshold else args.sketch_loss_weight
        pretrain_decoder(global_step,
                            pretrain_data_loader,
                            table_data,
                            model,
                            optimizer,
                            args.clip_grad,
                            sketch_loss_weight=sketch_loss_weight)


    n_steps = int(args.num_epochs * len(train_loader))
    normal_eval_steps = len(train_loader)
    spider_eval_steps = len(train_loader)
    print("Start training with {} epochs".format(args.num_epochs))
    t = TicToc()
    for step in tqdm(range(n_steps)):
        t.tic()
        sample_id = train_loader.sample_next()
        data_row = train_loader.dataset[sample_id]
        ac_loss, loss, reward = train_step_actor_critic(
            data_row,
            table_data,
            kmaps,
            db_names_to_schema,
            model,
            optimizer,
            val_optimizer,
            args.clip_grad,
            sketch_loss_weight=args.sketch_loss_weight)
        train_loader.update_sample(sample_id, reward==1)
        if ac_loss is None:
            continue

        wandb.log(
            {
                'train/ac_loss': float(ac_loss),
                'train/loss': float(loss),
                'train/reward': float(reward),
            }
        )

        train_time = t.tocvalue()

        if step % normal_eval_steps == 0 and step > 0:
            with torch.no_grad():
                sketch_acc, acc, _, predictions = evaluate(
                    model,
                    dev_loader,
                    table_data,
                    args.beam_size
                )

            with open(os.path.join(output_path, 'predictions_sem_ql.json'), 'w', encoding='utf-8') as f:
                json.dump(predictions, f, indent=2)

            eval_results_string = "Epoch: {}    Sketch-Accuracy: {}     Accuracy: {}".format(step + 1, sketch_acc, acc)
            tqdm.write(eval_results_string)

            if acc > best_acc:
                save_model(model, os.path.join(output_path))
                tqdm.write(
                    "Accuracy of this epoch ({}) is higher then the so far best accuracy ({}). Save model.".format(acc,
                                                                                                                   best_acc))
                best_acc = acc

            with open(os.path.join(output_path, "eval_results.log"), "a+", encoding='utf-8') as writer:
                writer.write(eval_results_string + "\n")
            scheduler.step()  # Update learning rate schedule
            wandb.log({"eval/sketch-accuracy": sketch_acc, "eval/accuracy": acc})

        if step % spider_eval_steps == 0 and step > 0:
            total_transformed, fail_transform, spider_eval_results = transform_to_sql_and_evaluate_with_spider(
                predictions,
                table_data,
                output_path,
                args.data_dir,
                step + 1)

            tqdm.write("Successfully transformed {} of {} from SemQL to SQL.".format(total_transformed - fail_transform, total_transformed))
            tqdm.write("Results from Spider-Evaluation:")
            log_dict = {}
            for key, value in spider_eval_results.items():
                tqdm.write("{}: {}".format(key, value))
                log_dict = {
                    f'spider_eval/{key}': value
                }

            wandb.log(log_dict)





