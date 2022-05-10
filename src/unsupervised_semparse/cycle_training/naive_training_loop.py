import torch, os
from torch import parse_ir

from src.data_loader import get_data_loader, get_random_sampler
from src.manual_inference.helper import _semql_to_sql
from src.unsupervised_semparse.cycle_training.data_collators import DataCollatorText2SQL, DataCollatorSQL2Text
from src.spider.test_suite_eval.process_sql import tokenize
from src.manual_inference.helper import tokenize_question
from src.spider.spider_utils import load_all_schema_data
from src.evaluation import evaluate
from src.unsupervised_semparse.cycle_training.utils import get_values, postprocess_text
from src.spider.test_suite_eval.evaluation import match_evaluation_single, build_foreign_key_map_from_json
from datasets import load_metric
from src.optimizer import build_optimizer_encoder, build_optimizer_base
from spacy.lang.en import English
import wandb 
from tqdm import tqdm


def batch_list(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]

class NaiveCycleTrainer:

    def __init__(
            self,
            ir_model,
            gpt2_model,
            gpt2_tokenizer,
            args,
            train_data,
            valid_data,
            grammar,
            schema,
            db_value_finders,
            dummy_queries,
            device
    ):
        nlp = English()
        self.ir_model = ir_model
        self.gpt2_model = gpt2_model
        self.gpt2_tokenizer = gpt2_tokenizer
        self.nlp_tokenizer = nlp.tokenizer
        self.args = args
        self.schema = schema
        self.device = device
        self.dummy_queries = dummy_queries
        self.db_value_finders = db_value_finders
        self.kmaps = build_foreign_key_map_from_json(os.path.join(args.data_dir, "original", 'tables.json'))
        self.db_names_to_schema = load_all_schema_data(os.path.join(args.data_dir, 'testsuite_databases'),
                                                       list(schema.keys()))
        self.bleu_metric = load_metric("sacrebleu")

        num_train_steps = len(train_data) * args.num_epochs
        self.ir_optimizer, self.ir_scheduler = build_optimizer_encoder(
            ir_model,
            num_train_steps,
            args.lr_transformer, args.lr_connection, args.lr_base,
            args.scheduler_gamma,
            use_rmsprop=True
        )

        self.gpt2_optimizer, self.gpt2_scheduler = build_optimizer_base(gpt2_model, num_train_steps, args.lr_base, args.scheduler_gamma, use_rmsprop=True)
        self.gpt2_scaler = torch.cuda.amp.GradScaler(init_scale=65536.0, growth_factor=2.0, backoff_factor=0.5,
                                                     growth_interval=2000, enabled=True)
        self.ir_scaler = torch.cuda.amp.GradScaler(init_scale=65536.0, growth_factor=2.0, backoff_factor=0.5,
                                                   growth_interval=2000, enabled=True)
        self.train_loader, self.dev_loader = get_data_loader(train_data, valid_data, 2*args.batch_size, True, False)
        db_names_to_schema = load_all_schema_data(os.path.join(args.data_dir, 'testsuite_databases'), list(schema.keys()))
        #self.train_loader, self.dev_loader = get_random_sampler(train_data, valid_data, args.batch_size, db_names_to_schema, 5)
        self.text2sql_collator = DataCollatorText2SQL(
            grammar=grammar,
            schema=schema,
            device=device
        )
        self.sql2text_collator = DataCollatorSQL2Text(
            tokenizer=gpt2_tokenizer,
            grammar=grammar,
            schema=schema,
            device=device
        )

        self.sql_baseline = []
        self.bleu_baseline = []

    def train(self):
        num_train_steps = int((len(self.train_loader) * self.args.num_epochs))
        train_data = self.train_loader.dataset
        for epoch in range(self.args.num_epochs):
            #generate fake data + filter using cycle
            fake_text_data, fake_sql_data = [], []
            filter_fake_text_data, filter_fake_sql_data = [], []
            gen_batch_size = 2*self.args.batch_size
            for step, batch in enumerate(tqdm(batch_list(train_data, gen_batch_size), desc="Generate Fake SQL", total=len(train_data)//gen_batch_size)):
                fake_sql_batch = self.text2sql(batch)
                fake_sql_data.extend(fake_sql_batch)

            #filter using cycle
            for fake_sql_batch in tqdm(batch_list(fake_sql_data, gen_batch_size), desc="Filter Fake SQL", total=len(fake_sql_data)//gen_batch_size):
                cycled_text_batch = self.sql2text(fake_sql_batch, skip_vals=True)
                text_rewards = self.reward_text(fake_sql_batch, cycled_text_batch)
                for i in range(len(text_rewards)):
                    if text_rewards[i] > 0.2:
                        filter_fake_sql_data.append(fake_sql_batch[i])

            for step, batch in enumerate(tqdm(batch_list(train_data, gen_batch_size), desc="Generate Fake Text", total=len(train_data)//gen_batch_size)):
                fake_text_batch = self.sql2text(batch, skip_vals=True)
                fake_text_data.extend(fake_text_batch)

            #filter using cycle
            for fake_text_batch in tqdm(batch_list(fake_text_data, gen_batch_size), desc="Filter Fake Text", total=len(fake_text_data)//gen_batch_size):
                cycled_sql_batch = self.text2sql(fake_text_batch)
                sql_rewards = self.reward_sql(fake_text_batch, cycled_sql_batch)
                for i in range(len(sql_rewards)):
                    if sql_rewards[i] == 1:
                        filter_fake_text_data.append(fake_text_batch[i])


            filtered_data = filter_fake_text_data + filter_fake_sql_data
            #train on fake data
            for batch in tqdm(batch_list(filtered_data, self.args.batch_size), desc="Training on fake text", total=len(filtered_data)//self.args.batch_size):
                logs = self.train_sql2text(batch)
                wandb.log(logs)

            for batch in tqdm(batch_list(filtered_data, self.args.batch_size), desc="Training on fake sql", total=len(filtered_data)//self.args.batch_size):
                logs = self.train_text2sql(batch)
                wandb.log(logs)

            eval_logs = self.evaluation()
            wandb.log(eval_logs)

            #update trainset
            train_data = self.update_train_set(self.train_loader.dataset, filtered_data)
            print("Number of fake text:", len(filter_fake_text_data))
            print("Number of fake sql:", len(filter_fake_sql_data))
            wandb.log({
                "Number of fake text": len(filter_fake_text_data),
                "Number of fake sql": len(filter_fake_sql_data),
                "Number of train data": len(train_data)
            })

    def train_sql2text(self, batch):
        examples, original_rows = self.sql2text_collator(batch, is_eval=False)
        with torch.cuda.amp.autocast():
            outputs = self.gpt2_model(**examples)
            loss = outputs.loss

        self.gpt2_optimizer.zero_grad()
        if self.device =='cpu':
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.gpt2_model.parameters(), self.args.clip_grad)
            self.gpt2_optimizer.step()
        else:
            self.gpt2_scaler.scale(loss).backward()
            self.gpt2_scaler.unscale_(self.gpt2_optimizer)
            torch.nn.utils.clip_grad_norm_(self.gpt2_model.parameters(), self.args.clip_grad)
            self.gpt2_scaler.step(self.gpt2_optimizer)
            self.gpt2_scaler.update()

        return {
            'gpt2/loss': float(loss.mean()),
        }

    def train_text2sql(self, batch):
        examples, original_rows = self.text2sql_collator(batch)
        with torch.cuda.amp.autocast():
            sketch_loss, lf_loss = self.ir_model.forward(examples)

            mean_sketch_loss = torch.mean(-sketch_loss)
            mean_lf_loss = torch.mean(-lf_loss)

            loss = mean_lf_loss + mean_sketch_loss

        mean_sketch_loss = float(torch.mean(-sketch_loss))
        mean_lf_loss = float(torch.mean(-lf_loss))

        self.ir_optimizer.zero_grad()
        if self.device == 'cpu':
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.ir_model.parameters(), self.args.clip_grad)
            self.ir_optimizer.step()
        else:
            self.ir_scaler.scale(loss).backward()
            self.ir_scaler.unscale_(self.ir_optimizer)
            torch.nn.utils.clip_grad_norm_(self.ir_model.parameters(), self.args.clip_grad)
            self.ir_scaler.step(self.ir_optimizer)
            self.ir_scaler.update()

        return {
            'ir/loss_sketch': mean_sketch_loss,
            'ir/loss_lf': mean_lf_loss
        }

    def sql2text(self, batch, skip_vals=False):
        beam_size = self.args.beam_size
        encoded_batch, original_rows = self.sql2text_collator(batch, is_eval=True)
        with torch.no_grad(), torch.cuda.amp.autocast():
            generated_out = self.gpt2_model.generate(
                encoded_batch['input_ids'],
                attention_mask=encoded_batch['attention_mask'],
                max_length=encoded_batch['input_ids'].shape[1] + 16,
                num_beams=beam_size,
                repetition_penalty=2.5,
                no_repeat_ngram_size=3,
                pad_token_id=self.gpt2_tokenizer.pad_token_id,
            )
        decoded_out = self.gpt2_tokenizer.batch_decode(generated_out, skip_special_tokens=True)
        pred_batch_out = [x.split('TEXT:')[1].replace('\n', '').replace('TEXT :', '').replace('TEXT', '') for x in decoded_out]
        for i, pred_out in enumerate(pred_batch_out):
            if len(pred_out) < 2:
                pred_out = 'What is this?'
            original_rows[i]['question'] = pred_out
            original_rows[i]['question_toks'] = tokenize_question(self.nlp_tokenizer, pred_out)
            if not skip_vals:
                original_rows[i]['values'] = get_values(
                    pred_out,
                    original_rows[i]['question_toks'],
                    original_rows[i]['table_names'],
                    original_rows[i]['col_set'],
                    self.db_value_finders[original_rows[i]['db_id']]
                )
        return original_rows

    def text2sql(self, batch):
        beam_size = self.args.beam_size
        examples, original_rows = self.text2sql_collator(batch)
        fake_batch = []
        for example, original_row in zip(examples, original_rows):
            with torch.no_grad(), torch.cuda.amp.autocast():
                results_all = self.ir_model.parse(example, beam_size=beam_size)
            results = results_all[0]
            all_predictions = []
            try:
                # here we set assemble the predicted actions (including leaf-nodes) as string
                full_prediction = " ".join([str(x) for x in results[0].actions])
                for beam in results:
                    all_predictions.append(" ".join([str(x) for x in beam.actions]))
            except Exception as e:
                full_prediction = ""

            if not full_prediction == "":
                # here we set assemble the predicted sketch actions as strings
                try:
                    original_row['rule_label'] = full_prediction
                    original_row['sketch_result'] = " ".join(str(x) for x in results_all[1])
                    original_row['model_result'] = full_prediction
                    sql = _semql_to_sql(original_row, self.schema).replace('"', '')
                    original_row['query'] = sql
                    original_row['query_toks'] = tokenize(sql)
                except Exception as e:
                    self.create_dummy_row(original_row)
            else:
                self.create_dummy_row(original_row)
            fake_batch.append(original_row)
        return fake_batch

    def reward_text(self, fake_sql_batch, cycled_text_batch):
        rewards = []
        for fake_sql_sample, cycled_text_sample in zip(fake_sql_batch, cycled_text_batch):
            text_in = fake_sql_sample['question']
            text_out = cycled_text_sample['question']

            decoded_preds, decoded_labels = postprocess_text([text_out], [text_in])
            result = self.bleu_metric.compute(predictions=decoded_preds, references=decoded_labels)['score'] / 100
            rewards.append(result)
        return rewards

    def update_train_set(self, train_set, filtered_train_set):
        updated_train_set_ids = set()
        for i, example in enumerate(filtered_train_set):
            sql_out = example['query']
            db_id = example['db_id']
            db_train_set = [(i, x) for i,x in enumerate(train_set) if x['db_id'] == db_id]
            for j, dp in db_train_set:
                sql_in = dp['query']
                eval_results = match_evaluation_single(
                    sql_in,
                    sql_out,
                    db_id,
                    self.db_names_to_schema[db_id],
                    self.kmaps
                )
                partial_score = sum([v['acc'] for k, v in eval_results['partial'].items()]) / len(
                    eval_results['partial'].items())
                if partial_score >= 0.8:
                    updated_train_set_ids.add(j)
        updated_train_set = [train_set[i] for i in updated_train_set_ids]
        return updated_train_set

    def reward_sql(self, fake_text_batch, cycled_sql_batch):
        rewards = []
        for fake_text_sample, cycled_sql_sample in zip(fake_text_batch, cycled_sql_batch):
            sql_in = fake_text_sample['query']
            sql_out = cycled_sql_sample['query']
            db_name = fake_text_sample['db_id']
            eval_results = match_evaluation_single(
                sql_in,
                sql_out,
                db_name,
                self.db_names_to_schema[db_name],
                self.kmaps
            )
            partial_score = sum([v['acc'] for k, v in eval_results['partial'].items()])/len(eval_results['partial'].items())
            reward = partial_score if partial_score == 1 else 0
            #reward = -1.0 if eval_results['exact'] == 0 else 1.0
            rewards.append(reward)
        return rewards

    def create_dummy_row(self, original_row):
        db_id = original_row['db_id']
        original_row['rule_label'] = self.dummy_queries[db_id]['rule_label']
        original_row['query'] = self.dummy_queries[db_id]['query']
        original_row['query_toks'] = self.dummy_queries[db_id]['query_toks']

    def evaluation(self):
        sketch_acc, acc, _, predictions = evaluate(self.ir_model,
                                                   self.dev_loader,
                                                   self.schema,
                                                   self.args.beam_size)
        return {
            'eval/sketch_acc': sketch_acc,
            'eval/acc': acc,
        }