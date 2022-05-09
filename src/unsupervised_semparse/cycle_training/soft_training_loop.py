import torch, os, random, copy
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
from torch.nn import CrossEntropyLoss
from spacy.lang.en import English
import wandb
from collections import deque
from tqdm import tqdm


class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, data):
        """Save a transition"""
        self.memory.append(data)

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class SoftUpdateTrainer:

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

        self.target_ir_model = copy.deepcopy(ir_model)
        self.target_gpt2_model = copy.deepcopy(gpt2_model)

        self.gpt2_tokenizer = gpt2_tokenizer
        self.nlp_tokenizer = nlp.tokenizer
        self.args = args
        self.schema = schema
        self.device = device
        self.dummy_queries = dummy_queries
        self.db_value_finders = db_value_finders
        self.kmaps = build_foreign_key_map_from_json(os.path.join(args.data_dir, "original", 'tables.json'))
        self.db_names_to_schema = load_all_schema_data(os.path.join(args.data_dir, 'testsuite_databases'), list(schema.keys()))
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

        self.sql_baseline = []
        self.bleu_baseline = [0.3]
        self.text_memory = ReplayMemory(5000)
        self.sql_memory = ReplayMemory(5000)
        self.ir_tau = 0.001
        self.gpt2_tau = 0.01

    def train(self):
        num_train_steps = int((len(self.train_loader) * self.args.num_epochs))
        text_update, sql_update = 0, 0
        for step in tqdm(range(num_train_steps), desc="Training", total=num_train_steps):
            sample_ids = self.train_loader.sample_batch(self.args.batch_size)
            batch = [self.train_loader.dataset[sample_id] for sample_id in sample_ids]
            #batch = next(iter(self.train_loader))
            logs = {}
            if step % 2 == 0:
                text_update += 1
                fake_text_batch = self.sql2text(batch, skip_vals=True)
                cycled_sql_batch = self.text2sql(fake_text_batch)
                sql_rewards = self.reward_sql(fake_text_batch, cycled_sql_batch)
                sql_rewards_torch = torch.tensor(sql_rewards, dtype=torch.float, device=self.device)
                self.sql_baseline.extend(sql_rewards)
                logs['train/sql_rewards_torch'] = float(sql_rewards_torch.mean())
                for idx, sample_id in enumerate(sample_ids):
                    self.train_loader.update_sample(sample_id, sql_rewards[idx] == 1)
                    fake_text_batch[idx]['reward'] = sql_rewards[idx]
                    if fake_text_batch[idx].get('fail', False) or cycled_sql_batch[idx].get('fail', False):
                        continue
                    self.text_memory.push(fake_text_batch[idx])
                if text_update % self.args.update_every == 0:
                    logs = self.update_sql2text()
                if text_update % 10 == 0:
                    print(fake_text_batch[0]['query'], fake_text_batch[0]['question'], cycled_sql_batch[0]['query'], sql_rewards[0])
                #gpt_train_res = self.train_sql2text(fake_text_batch, sql_rewards_torch, sql_baseline)
            else:
                sql_update += 1
                fake_sql_batch = self.text2sql(batch)
                #cycled_loss = self.sql2text_loss(fake_sql_batch)
                #text_rewards_torch = 1 - cycled_loss
                #text_rewards = [float(x) for x in text_rewards_torch]
                cycled_text_batch = self.sql2text(fake_sql_batch, skip_vals=True)
                #text rewards are not very reliable, thus do a super-cycle
                super_cycled_sql_batch = self.text2sql(cycled_text_batch)
                text_rewards = self.reward_text(fake_sql_batch, cycled_text_batch)
                text_rewards_torch = torch.tensor(text_rewards, dtype=torch.float, device=self.device)
                sql_rewards = self.reward_sql(super_cycled_sql_batch, fake_sql_batch)
                sql_rewards_torch = torch.tensor(sql_rewards, dtype=torch.float, device=self.device)
                text_rewards_torch = text_rewards_torch*sql_rewards_torch
                self.bleu_baseline.extend(text_rewards)
                bleu_baseline = sum(self.bleu_baseline) / len(self.bleu_baseline)
                logs['train/text_rewards_torch'] = float(text_rewards_torch.mean())
                for idx, sample_id in enumerate(sample_ids):
                    self.train_loader.update_sample(sample_id, float(text_rewards[idx]) > bleu_baseline)
                    fake_sql_batch[idx]['reward'] = text_rewards[idx]
                    if fake_sql_batch[idx].get('fail', False) or cycled_text_batch[idx].get('fail', False) or super_cycled_sql_batch[idx].get('fail', False):
                        continue
                    self.sql_memory.push(fake_sql_batch[idx])
                if sql_update % self.args.update_every == 0:
                    logs = self.update_text2sql()
                if sql_update % 10 == 0:
                    print(fake_sql_batch[0]['question'], fake_sql_batch[0]['query'], cycled_text_batch[0]['question'], text_rewards[0])

            self.bleu_baseline = self.bleu_baseline[-100:]
            self.sql_baseline = self.sql_baseline[-100:]
            data_loader_logs = self.train_loader.get_logging_info()

            logs = {**logs, **data_loader_logs}
            if step % 100 == 0 and step != 0:
                eval_logs = self.evaluation()
                logs = {**logs, **eval_logs}

            wandb.log(logs)

    def update_sql2text(self):
        if len(self.text_memory) < self.args.batch_size:
            return {}
        batch = self.text_memory.sample(self.args.batch_size)
        rewards_batch = [x['reward'] for x in batch]
        rewards_batch_torch = torch.tensor(rewards_batch, dtype=torch.float, device=self.device)
        sql_baseline = sum(self.sql_baseline) / len(self.sql_baseline)
        logs = self.train_sql2text(batch, rewards_batch_torch, sql_baseline)
        logs['train/sql_baseline'] = float(sql_baseline)
        self.soft_update(self.gpt2_model, self.target_gpt2_model, self.gpt2_tau)
        return logs

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
            self.gpt2_scaler.unscale_(self.gpt2_optimizer)
            torch.nn.utils.clip_grad_norm_(self.gpt2_model.parameters(), self.args.clip_grad)
            self.gpt2_scaler.step(self.gpt2_optimizer)
            self.gpt2_scaler.update()

        return {
            'gpt2/loss': float(loss.mean()),
            'gpt2/loss_adv': float(loss_adv.mean()),
        }

    def sql2text_loss(self, batch):
        examples, original_rows, question_mask = self.sql2text_collator(batch, return_question_mask=True, is_eval=False)
        with torch.no_grad(), torch.cuda.amp.autocast():
            outputs = self.gpt2_model(**examples)
            logits = outputs.logits

            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = examples['labels'][..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss(reduction='none')
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            loss = loss.view(shift_logits.size(0), -1)
            loss_masked = (loss * question_mask[..., 1:]).sum(dim=1)
            mean_loss = loss_masked/(question_mask[..., 1:]).sum(dim=1)
        return mean_loss

    def update_text2sql(self):
        if len(self.sql_memory) < self.args.batch_size:
            return {}
        batch = self.sql_memory.sample(self.args.batch_size)
        rewards_batch = [x['reward'] for x in batch]
        rewards_batch_torch = torch.tensor(rewards_batch, dtype=torch.float, device=self.device)
        bleu_baseline = sum(self.bleu_baseline) / len(self.bleu_baseline)
        logs = self.train_text2sql(batch, rewards_batch_torch, bleu_baseline)
        logs['train/bleu_baseline'] = float(bleu_baseline)
        self.soft_update(self.ir_model, self.target_ir_model, self.ir_tau)
        return logs

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
            self.ir_scaler.unscale_(self.ir_optimizer)
            torch.nn.utils.clip_grad_norm_(self.ir_model.parameters(), self.args.clip_grad)
            self.ir_scaler.step(self.ir_optimizer)
            self.ir_scaler.update()

        return {
            'ir/loss_rl': float(loss_adv),
            'ir/loss_sketch': mean_sketch_loss,
            'ir/loss_lf': mean_lf_loss
        }

    def sql2text(self, batch, skip_vals=False):
        beam_size = self.args.beam_size
        encoded_batch, original_rows = self.sql2text_collator(batch, is_eval=True)
        with torch.no_grad(), torch.cuda.amp.autocast():
            generated_out = self.target_gpt2_model.generate(
                encoded_batch['input_ids'],
                attention_mask=encoded_batch['attention_mask'],
                max_length=encoded_batch['input_ids'].shape[1] + 24,
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
                original_rows[i]['fail'] = True
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
                results_all = self.target_ir_model.parse(example, beam_size=beam_size)
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
            if result < 0.2:
                result = 0.0
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
            partial_score = sum([v['acc'] for k, v in eval_results['partial'].items()]) / len(eval_results['partial'].items())
            reward = partial_score if partial_score == 1 else 0
            #reward = -1.0 if eval_results['exact'] == 0 else 1.0
            rewards.append(reward)
        return rewards

    def create_dummy_row(self, original_row):
        db_id = original_row['db_id']
        original_row['rule_label'] = self.dummy_queries[db_id]['rule_label']
        original_row['query'] = self.dummy_queries[db_id]['query']
        original_row['query_toks'] = self.dummy_queries[db_id]['query_toks']
        original_row['fail'] = True

    def evaluation(self):
        sketch_acc, acc, _, predictions = evaluate(self.ir_model,
                                                   self.dev_loader,
                                                   self.schema,
                                                   self.args.beam_size)
        return {
            'eval/sketch_acc': sketch_acc,
            'eval/acc': acc,
        }

    def soft_update(self, policy, target, tau):
        for target_param, param in zip(target.parameters(), policy.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - tau) + param.data * tau
            )