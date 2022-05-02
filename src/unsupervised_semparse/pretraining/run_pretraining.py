import torch

from datasets import load_metric

from src.model.model import IRNet
from src.data_loader import get_data_loader
from src.config import write_config_to_file, read_arguments_pretrain
from src.intermediate_representation import semQL
from src.spider import spider_utils
from src.optimizer import build_optimizer_encoder
from src.utils import setup_device, set_seed_everywhere, create_experiment_folder
from src.unsupervised_semparse.pretraining.pretrain_data_collator import DataCollatorSQL2SQL
from src.unsupervised_semparse.pretraining.training_loops import pretrain_loop

metric = load_metric("sacrebleu")
# initialize experiment tracking @ Weights & Biases
import wandb

if __name__ == '__main__':
    args = read_arguments_pretrain()

    # log hyperparameters to Weights & Biases
    experiment_name, output_path = create_experiment_folder(args.model_output_dir, args.exp_name)
    print("Run experiment '{}'".format(experiment_name))

    wandb.init(
        project="sql2text",
        group="pretraining",
        name=experiment_name
    )

    wandb.config.update(args)

    write_config_to_file(args, output_path)

    device, n_gpu = setup_device()
    set_seed_everywhere(args.seed, n_gpu)

    sql_data, table_data, val_sql_data, val_table_data = spider_utils.load_dataset(args.data_dir, use_small=args.toy)

    grammar = semQL.Grammar()

    if args.pretrain_type == 'sql2sql':
        train_loader, dev_loader = get_data_loader(sql_data, val_sql_data, args.batch_size, True, False)
        data_collator = DataCollatorSQL2SQL(
            grammar=grammar,
            schema=table_data,
            device=device
        )

        model = IRNet(args, device, grammar)
        model.to(device)

        num_train_steps = len(sql_data) * args.num_epochs
        optimizer, scheduler = build_optimizer_encoder(
            model,
            num_train_steps,
            args.lr_transformer, args.lr_connection, args.lr_base,
            args.scheduler_gamma
        )

        pretrain_loop(
            args,
            train_loader,
            dev_loader,
            data_collator,
            model,
            optimizer,
            scheduler,
            output_path
        )
