import torch, os
from src.data_loader import get_data_loader, get_random_sampler
from src.manual_inference.helper import _semql_to_sql
from src.unsupervised_semparse.cycle_training.data_collators import DataCollatorText2SQL, DataCollatorSQL2Text
from src.spider.test_suite_eval.process_sql import tokenize
from src.manual_inference.helper import tokenize_question
from src.spider.spider_utils import load_all_schema_data
from src.unsupervised_semparse.cycle_training.utils import get_values, postprocess_text
from src.spider.test_suite_eval.evaluation import match_evaluation_single, build_foreign_key_map_from_json
from datasets import load_metric
from src.optimizer import build_optimizer_encoder, build_optimizer_base
from torch.nn import CrossEntropyLoss
from spacy.lang.en import English
import wandb
from tqdm import tqdm


class CycleTrainer:

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
            args.scheduler_gamma
        )

        self.gpt2_optimizer, self.gpt2_scheduler = build_optimizer_base(gpt2_model, num_train_steps, args.lr_base, args.scheduler_gamma)
        self.gpt2_scaler = torch.cuda.amp.GradScaler(init_scale=65536.0, growth_factor=2.0, backoff_factor=0.5,
                                                     growth_interval=2000, enabled=True)
        self.ir_scaler = torch.cuda.amp.GradScaler(init_scale=65536.0, growth_factor=2.0, backoff_factor=0.5,
                                                   growth_interval=2000, enabled=True)
        #self.train_loader, self.dev_loader = get_data_loader(train_data, valid_data, args.batch_size, True, False)
        db_names_to_schema = load_all_schema_data(os.path.join(args.data_dir, 'testsuite_databases'), list(schema.keys()))
        self.train_loader, self.dev_loader = get_random_sampler(train_data, valid_data, args.batch_size, db_names_to_schema, 5)
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

        self.sql_baseline = [-0.95]
        self.bleu_baseline = [0.1]

    def train(self):
        num_train_steps = int((len(self.train_loader) * self.args.num_epochs)/self.args.batch_size)
        for step in tqdm(range(num_train_steps), desc="Training", total=num_train_steps):
            sample_ids = self.train_loader.sample_batch(self.args.batch_size)
            batch = [self.train_loader.dataset[sample_id] for sample_id in sample_ids]

            fake_text_batch = self.sql2text(batch)
            fake_sql_batch = self.text2sql(batch)

            cycled_sql_batch = self.text2sql(fake_text_batch)
            cycled_text_batch = self.sql2text(fake_sql_batch)

            sql_rewards = self.reward_sql(fake_text_batch, cycled_sql_batch)
            text_rewards = self.reward_text(fake_sql_batch, cycled_text_batch)

            sql_rewards_torch = torch.tensor(sql_rewards, dtype=torch.float, device=self.device)
            text_rewards_torch = torch.tensor(text_rewards, dtype=torch.float, device=self.device)

            bleu_baseline = sum(self.bleu_baseline) / len(self.bleu_baseline)
            sql_baseline = sum(self.sql_baseline) / len(self.sql_baseline)
            ir_res = self.train_text2sql(fake_sql_batch, text_rewards_torch, bleu_baseline)
            gpt_train_res = self.train_sql2text(fake_text_batch, sql_rewards_torch, sql_baseline)

            self.bleu_baseline.extend(text_rewards)
            self.sql_baseline.extend(sql_rewards_torch)

            self.bleu_baseline = self.bleu_baseline[-100:]
            self.sql_baseline = self.sql_baseline[-100:]

            for i, sample_id in enumerate(sample_ids):
                if sql_rewards[i] == 1 or text_rewards[i] > 0.2:
                    self.train_loader.update_sample(sample_id, True)
                else:
                    self.train_loader.update_sample(sample_id, False)

            bos_distr = self.train_loader.get_box_distribution()
            bin_nr = [i for i in range(len(self.train_loader.boxes) + 1)]
            logs = {**ir_res, **gpt_train_res}
            logs['sql_rewards_torch'] = float(sql_rewards_torch.mean())
            logs['text_rewards_torch'] = float(text_rewards_torch.mean())
            logs['bleu_baseline'] = float(bleu_baseline)
            logs['sql_baseline'] = float(sql_baseline)
            logs['box_distr:'] = wandb.Histogram(np_histogram=(bos_distr, bin_nr))
            logs['deck_size'] = self.train_loader.get_deck_size()
            wandb.log(logs)

    def train_sql2text(self, batch, rewards_batch, baseline):
        examples, original_rows = self.sql2text_collator(batch, is_eval=False)
        examples_no_loss = {
            "input_ids": examples["input_ids"],
            "attention_mask": examples["attention_mask"]
        }
        with torch.cuda.amp.autocast():
            outputs = self.gpt2_model(**examples_no_loss)
            logits = outputs.logits

            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = examples['labels'][..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss(reduction='none')
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            loss = loss.view(shift_logits.size(0), -1).mean(dim=1)
            loss_adv = (loss * (rewards_batch - baseline)).mean()

        self.gpt2_optimizer.zero_grad()
        if self.device =='cpu':
            loss_adv.backward()
            torch.nn.utils.clip_grad_norm_(self.gpt2_model.parameters(), self.args.clip_grad)
            self.gpt2_optimizer.step()
        else:
            self.gpt2_scaler.scale(loss_adv).backward()
            self.gpt2_scaler.step(self.gpt2_optimizer)
            self.gpt2_scaler.update()

        return {
            'gpt2_loss': float(loss.mean()),
            'gpt2_loss_adv': float(loss_adv.mean()),
        }

    def train_text2sql(self, batch, rewards_batch, baseline):
        examples, original_rows = self.text2sql_collator(batch)
        with torch.cuda.amp.autocast():
            sketch_loss, lf_loss = self.ir_model.forward(examples)

            loss_batch = -sketch_loss - lf_loss
            loss_batch_adv = loss_batch * (rewards_batch - baseline)
            loss_adv = loss_batch_adv.mean()

        mean_sketch_loss = float(torch.mean(-sketch_loss))
        mean_lf_loss = float(torch.mean(-lf_loss))

        self.ir_optimizer.zero_grad()
        if self.device == 'cpu':
            loss_adv.backward()
            torch.nn.utils.clip_grad_norm_(self.ir_model.parameters(), self.args.clip_grad)
            self.ir_optimizer.step()
        else:
            self.ir_scaler.scale(loss_adv).backward()
            self.ir_scaler.step(self.ir_optimizer)
            self.ir_scaler.update()

        return {
            'ir_loss_rl': float(loss_adv),
            'ir_loss_sketch': mean_sketch_loss,
            'ir_loss_lf': mean_lf_loss
        }

    def sql2text(self, batch):
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
        pred_batch_out = [x.split('TEXT:')[1].replace('\n', '').replace('TEXT :', '').replace('TEXT', '') for x in
                          decoded_out]
        for i, pred_out in enumerate(pred_batch_out):
            if len(pred_out) < 2:
                pred_out = 'What is this?'
            original_rows[i]['question'] = pred_out
            original_rows[i]['question_toks'] = tokenize_question(self.nlp_tokenizer, pred_out)
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
                # here we set assemble the predicted sketch actions as string
                original_row['rule_label'] = full_prediction
                original_row['sketch_result'] = " ".join(str(x) for x in results_all[1])
                original_row['model_result'] = full_prediction

                sql = _semql_to_sql(original_row, self.schema).replace('"', '')

                original_row['query'] = sql
                original_row['query_toks'] = tokenize(sql)
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
            reward = -1.0 if eval_results['exact'] == 0 else 1.0
            rewards.append(reward)
        return rewards

    def create_dummy_row(self, original_row):
        db_id = original_row['db_id']
        original_row['rule_label'] = self.dummy_queries[db_id]['rule_label']
        original_row['query'] = self.dummy_queries[db_id]['query']
        original_row['query_toks'] = self.dummy_queries[db_id]['query_toks']
