import torch
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import matthews_corrcoef, f1_score, accuracy_score

from transformers import get_cosine_schedule_with_warmup

import wandb
from optimizer_cp import AdamW, SGD, Flora, RandomizedSGD, RandomizedGalore, RandomizedApollo

torch.set_printoptions(precision=4)


def centralized_glue(model, dataloader, val_dataloader, args):
    device = args.device
    model.to(device)
    if 'rso' in args.method:
        if args.optim == 'AdamW':
            optimizer = AdamW(model.named_parameters(), lr=args.lr, weight_decay=args.weight_decay,
                              scaling_factor=(args.lora_alpha / args.lora_r),
                              interval=args.interval,
                              optimizer_states=args.optimizer_states,
                              args=args)
        elif 'SGD' in args.optim:
            optimizer = SGD(model.named_parameters(), lr=args.lr, weight_decay=args.weight_decay,
                            momentum=0.9 if args.optim == 'SGDM' else 0,
                            scaling_factor=(args.lora_alpha / args.lora_r),
                            interval=args.interval,
                            optimizer_states=args.optimizer_states,
                            args=args)
    elif args.method == "primer":
        if args.optim == 'AdamW':
            optimizer = RandomizedGalore(model.named_parameters(), lr=args.lr, weight_decay=args.weight_decay,
                                         interval=args.interval,
                                         args=args)
        elif args.optim == 'Apollo':
            optimizer = RandomizedApollo(model.named_parameters(), lr=args.lr, weight_decay=args.weight_decay,
                                         interval=args.interval,
                                         args=args)
        elif 'SGD' in args.optim:
            optimizer = RandomizedSGD(model.named_parameters(), lr=args.lr,
                                      momentum=0.9 if args.optim == 'SGDM' else 0,
                                      weight_decay=args.weight_decay,
                                      args=args)
    elif args.method == "flora":
        optimizer = Flora(model.named_parameters(), lr=args.lr, interval=args.interval,
                          weight_decay=args.weight_decay,
                          scaling_factor=(args.lora_alpha / args.lora_r),
                          optimizer_states=args.optimizer_states,
                          args=args)
    else:
        if args.optim == 'AdamW':
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        elif 'SGD' in args.optim:
            optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
                                        momentum=0.9 if args.optim == 'SGDM' else 0)

    total_steps = len(dataloader) * args.epochs

    num_warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=total_steps
    )

    max_metric_1 = 0
    max_metric_2 = 0
    print('-' * 32, f'{args.method} training start !')
    epoch_loss = []
    for epoch in range(args.epochs):
        model.train()
        step = 0
        total_loss = 0
        for data in dataloader:
            loss = model(**{k: v.to(device) for k, v in data.items()}).loss
            loss.backward()

            optimizer.step(
                closure=lambda: model(**{k: v.to(device) for k, v in data.items()}).loss if args.method == 'primer'
                else None)
            scheduler.step()
            optimizer.zero_grad()

            total_loss += loss.item()
            step += 1
            if step % 20 == 0:
                print(f'Step :{step}, Loss: {total_loss / step}')
            if wandb.run:
                wandb.log({
                    'Step': step,
                    'Loss': loss.item()
                })
        epoch_loss.append(total_loss / step)
        if wandb.run:
            wandb.log({
                'Round': epoch,
                'Epoch_Loss': epoch_loss[-1]
            })

        max_metric_1, max_metric_2 = evaluate_model(
            model, val_dataloader, args, epoch, max_metric_1, max_metric_2
        )

    return model


def calculate_metrics(all_true_labels, all_predictions, task):
    if task == "cola":
        return accuracy_score(all_true_labels, all_predictions), matthews_corrcoef(
            all_true_labels, all_predictions
        )
    elif task in ["sst2", "qnli", "rte", "wnli", "mnli_matched", "mnli_mismatched"]:
        return accuracy_score(all_true_labels, all_predictions), None
    elif task in ["mrpc", "qqp"]:
        return f1_score(all_true_labels, all_predictions), accuracy_score(
            all_true_labels, all_predictions
        )
    elif task == "stsb":
        return (
            pearsonr(all_true_labels, all_predictions)[0],
            spearmanr(all_true_labels, all_predictions)[0],
        )
    else:
        raise ValueError(f"Unknown task: {task}")


def evaluate_model(model, dataloader, args, r, max_metric1, max_metric2):
    device = args.device

    model.eval()
    eval_loss = 0
    all_predictions = []
    all_true_labels = []

    for batch in dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():

            outputs = model(**batch)

            eval_loss += outputs.loss.detach().cpu().numpy()

            if args.task == "stsb":
                predictions = outputs.logits.squeeze().cpu().numpy()
            else:
                predictions = outputs.logits.argmax(dim=-1).cpu().numpy()
            all_predictions.extend(predictions)
            all_true_labels.extend(batch["labels"].cpu().numpy())

    eval_loss /= len(dataloader)

    # Calculate the metrics for the specific task
    metric1, metric2 = calculate_metrics(all_true_labels, all_predictions, args.task)

    if metric1 > max_metric1:
        max_metric1 = metric1

    if metric2 is not None and metric2 > max_metric2:
        max_metric2 = metric2

    print(f"{args.task} - Eval Loss: {eval_loss:.4f}, Metric 1: {metric1:.4f}")
    if metric2 is not None:
        print(f"{args.task} - Metric 2: {metric2:.4f}")
    print(f"{args.task} - Max Metric 1: {max_metric1:.4f}")
    if max_metric2 is not None:
        print(f"{args.task} - Max Metric 2: {max_metric2:.4f}")

    if wandb.run:
        wandb.log(
            {
                "round": r,
                f"eval_loss": eval_loss,
                f"metric1": metric1,
                f"metric2": metric2 if metric2 is not None else 0,
                f"max_metric1": max_metric1,
                f"max_metric2": max_metric2 if max_metric2 is not None else 0,
            }
        )

    return max_metric1, max_metric2
