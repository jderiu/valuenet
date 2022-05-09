import wandb, json
import torch, os

from transformers.models.gpt2.modeling_gpt2 import GPT2LMHeadModel
from transformers import AutoTokenizer


from src.model.model import IRNet
from src.config import write_config_to_file, read_arguments_cycletrain
from src.intermediate_representation import semQL
from src.spider import spider_utils
from src.utils import setup_device, set_seed_everywhere, create_experiment_folder
from src.unsupervised_semparse.cycle_training.training_loop import CycleTrainer
from src.unsupervised_semparse.cycle_training.naive_training_loop import NaiveCycleTrainer
from src.unsupervised_semparse.cycle_training.soft_training_loop import SoftUpdateTrainer
from src.named_entity_recognition.database_value_finder.database_value_finder_sqlite import DatabaseValueFinderSQLite
from src.manual_inference.helper import get_schemas_spider


if __name__ == '__main__':
    args = read_arguments_cycletrain()

    experiment_name, output_path = create_experiment_folder(args.model_output_dir, args.exp_name)
    print("Run experiment '{}'".format(experiment_name))

    wandb.init(
        project="sql2text",
        group="cycle_training",
        name=experiment_name
    )

    wandb.config.update(args)
    write_config_to_file(args, output_path)
    device, n_gpu = setup_device()
    #device = 'cpu'
    set_seed_everywhere(args.seed, n_gpu)
    sql_data, table_data, val_sql_data, val_table_data = spider_utils.load_dataset(args.data_dir, use_small=args.toy)
    #adapt this later for curriculum learning (start with easy and then increase difficulty)
    grammar = semQL.Grammar()

    schemas_raw_spider, schemas_dict_spider, schema_path_spider, database_path_spider = get_schemas_spider()
    db_value_finders = {
        db_name : DatabaseValueFinderSQLite(database_path_spider, db_name, schema_path_spider, use_paralelization=False) for db_name in schemas_dict_spider.keys()
    }

    with open(os.path.join(args.data_dir, 'dummy_queries.json'), 'rt', encoding='utf-8') as f:
        dummy_queries = json.load(f)

    # Load pretrained model
    ir_model = IRNet(args, device, grammar)
    ir_model.load_state_dict(torch.load(args.ir_model_to_load), strict=False)
    print("Load pre-trained IR model from '{}'".format(args.ir_model_to_load))
    #reinitialize encoder to BART (encoder was ruined to learn SQL)
    #ir_model.init_encoder()
    print("Re-loaded BART encoder")
    ir_model.to(device)

    num_train_steps = len(sql_data) * args.num_epochs

    with open(os.path.join(args.gpt2_model_to_load, "args.json"), "rt", encoding='utf-8') as f:
        gpt_train_args = json.load(f)

    decoder_tokenizer = AutoTokenizer.from_pretrained(gpt_train_args['decoder_pretrained_model'], add_prefix_space=True)
    decoder_tokenizer.padding_side = 'left'
    if decoder_tokenizer.pad_token_id is None:
        decoder_tokenizer.pad_token = decoder_tokenizer.bos_token
    if decoder_tokenizer.sep_token_id is None:
        decoder_tokenizer.sep_token = decoder_tokenizer.bos_token
    gpt2_model = GPT2LMHeadModel.from_pretrained(os.path.join(args.gpt2_model_to_load, f'checkpoint-{args.gpt2_checkpoint}'))
    gpt2_model.to(device)
    print("Loaded pre-trained GPT2 model from '{}'".format(args.gpt2_model_to_load))

    trainer = NaiveCycleTrainer(
        ir_model,
        gpt2_model,
        decoder_tokenizer,
        args,
        sql_data,
        val_sql_data,
        grammar,
        table_data,
        db_value_finders,
        dummy_queries,
        device
    )

    trainer.train()