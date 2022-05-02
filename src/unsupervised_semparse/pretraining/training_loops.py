import os
import torch
import wandb
from tqdm import tqdm
from pytictoc import TicToc

from src.utils import save_model


def pretrain_loop(
        args,
        train_dataloader,
        dev_loader,
        data_collator,
        model,
        optimizer,
        scheduler,
        output_path

):
    global_step = 0
    best_acc = 0.0
    t = TicToc()
    for epoch in tqdm(range(int(args.num_epochs))):
        sketch_loss_weight = 1 if epoch < args.loss_epoch_threshold else args.sketch_loss_weight
        t.tic()

        pretrain_decoder_epoch(
            global_step,
            train_dataloader,
            data_collator,
            model,
            optimizer,
            args.clip_grad,
            sketch_loss_weight=sketch_loss_weight
        )

        sketch_acc, acc, _, predictions = pretrain_evaluate(model,
                                                            dev_loader,
                                                            data_collator,
                                                            args.beam_size
                                                            )

        if acc > best_acc:
            save_model(model, os.path.join(output_path))
            tqdm.write(
                "Accuracy of this epoch ({}) is higher then the so far best accuracy ({}). Save model.".format(acc,
                                                                                                               best_acc))
            best_acc = acc
        eval_results_string = "Epoch: {}    Sketch-Accuracy: {}     Accuracy: {}".format(epoch + 1, sketch_acc, acc)
        tqdm.write(eval_results_string)
        with open(os.path.join(output_path, "eval_results.log"), "a+", encoding='utf-8') as writer:
            writer.write(eval_results_string + "\n")

        wandb.log({"eval/Sketch-accuracy": sketch_acc, "eval/accuracy": acc})
        scheduler.step()


def pretrain_evaluate(model, dev_loader, data_collator, beam_size):
    sketch_correct, rule_label_correct, found_in_beams, not_all_values_found, total = 0, 0, 0, 0, 0
    predictions = []
    for batch in tqdm(dev_loader, desc="Evaluating"):
        examples, original_rows = data_collator(batch)
        for example, original_row in zip(examples, original_rows):
            with torch.no_grad():
                results_all = model.parse(example, beam_size=beam_size)
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

            # here we set assemble the predicted sketch actions as string
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

    print(
        f"in {found_in_beams} times we found the correct results in another beam (failing queries: {total - rule_label_correct})")

    return float(sketch_correct) / float(total), float(rule_label_correct) / float(total), float(
        not_all_values_found) / float(total), predictions


def pretrain_decoder_epoch(
        global_step,
        train_dataloader,
        data_collator,
        model,
        optimizer,
        clip_grad,
        sketch_loss_weight=1,
        lf_loss_weight=1
):
    tr_loss = 0.0
    model.zero_grad()
    model.train()

    for step, batch in enumerate(tqdm(train_dataloader, desc="Training")):
        examples, _ = data_collator(batch)

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
