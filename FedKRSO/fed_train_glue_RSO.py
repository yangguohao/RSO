import os
import argparse
import random
import time
import warnings
from copy import deepcopy

from fed_agg import *
from train_eval import *
from data_utils import *
from models import *
from transformers import RobertaConfig, AutoModelForSequenceClassification


warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser(description="Federated Learning with LoRA")

parser.add_argument("--task", type=str, default="cola", help="GLUE task to fine-tune on")
parser.add_argument("--model", type=str, default="roberta-base", help="Model name")
parser.add_argument("--warmup_ratio", type=float, default=0.05, help="Warmup ratio")
parser.add_argument("--max_seq_length", type=int, default=128, help="Maximum sequence length")
parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
parser.add_argument("--seed", type=int, default=42, help="Random seed")
parser.add_argument("--device", type=int, default=0, help="device")
parser.add_argument("--weight_decay", type=float, default=0.01)

parser.add_argument("--amp", action="store_true", help="automatic mixed precision")

parser.add_argument("--alpha", type=float, default=-1, )
parser.add_argument("--agg_type", type=str, default="normal", help="Type of method, normal, ffa, rso")
parser.add_argument("--num_clients", type=int, default=10, help="Number of total clients")
parser.add_argument("--k_selected", type=int, default=10, help="Number of selected clients")
parser.add_argument("--rounds", type=int, default=30, help="Number of rounds")
parser.add_argument("--local_steps", type=int, default=100, help="Number of local steps")

parser.add_argument("--kseed", type=int, default=-1, )
parser.add_argument("--interval", type=int, default=10, )

parser.add_argument("--lora_r", type=int, default=4, help="LoRA R value")
parser.add_argument("--lora_alpha", type=int, default=8, help="LoRA alpha value")
parser.add_argument("--lora_dropout", type=float, default=0., help="LoRA dropout value")
parser.add_argument("--rslora", action="store_true", help="Use RSLoRA")
parser.add_argument("--wandb", action="store_true", help="Use wandb to record training")

args = parser.parse_args()

if args.agg_type == 'normal':
    if args.lora_r == 0:
        args.method = 'fft'
    else:
        args.method = 'fedit'
elif args.agg_type == 'rso':
    args.method = f'{args.agg_type}-interval{args.interval}'
else:
    args.method = args.agg_type

if args.wandb:
    wandb.init(project=f"{args.task}_bs{args.batch_size}_clients{args.num_clients}_selected{args.k_selected}"
                       f"_local-steps{args.local_steps}{f'_noniid-{args.alpha}' if args.alpha>0 else ''}"
                       f"-{args.model}_test-round"
               ,
               config=args,
               name=f"{args.method}-lr{args.lr}{f'-kseed{args.kseed}' if args.kseed != -1 else ''}-seed{args.seed}-round{args.rounds}")

random.seed(args.seed)
np.random.seed(args.seed)
os.environ['PYTHONHASHSEED'] = str(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
torch.backends.cudnn.deterministic = True
args.device = torch.device(f'cuda:{args.device}')


def create_model(base_model, args):
    if args.agg_type == 'normal':
        client_model = create_peft_model(base_model, args)
    elif args.agg_type == "ffa":
        client_model = create_peft_FFA_model(base_model, args)
    else:
        client_model = create_RSO_model(base_model, args)
    return client_model


def federated_learning(task):
    train_data, val_data, test_data = load_and_preprocess_data(task, args)

    num_labels = len(set(train_data["labels"].numpy()))

    if args.task == "stsb":
        num_labels = 1

    if args.alpha <= 0:
        client_dataloaders = create_client_dataloaders(train_data, args)
    else:
        client_dataloaders = create_noniid_client_dataloaders(train_data, args, train_data["labels"].numpy())

    val_dataloader = create_dataloader(val_data, args)

    base_model = AutoModelForSequenceClassification.from_pretrained(
        args.model,
        num_labels=num_labels,
    )
    device = args.device
    max_metric_1 = 0
    max_metric_2 = 0

    index_list = [i for i in range(args.num_clients)]
    for r in range(args.rounds):
        if args.agg_type != 'rso':
            if r == 0:
                global_model = create_model(base_model, args)
                global_model.to(device)
            global_dict = deepcopy(global_model.state_dict())
        else:
            base_model.to(device)
            if args.kseed != -1:
                seed_list = [random.randint(1, 2*32-1) for _ in range(args.kseed)]
                args.seed_list = seed_list
            base_dict = deepcopy(base_model.state_dict())

        client_models = []
        print(f"Round {r + 1}/{args.rounds}")
        random.shuffle(index_list)

        train_loss = 0
        for i in index_list[:args.k_selected]:
            if client_dataloaders[i] is not None:
                if args.agg_type == 'rso':
                    base_model.load_state_dict(base_dict, strict=False)

                client_model = create_model(base_model, args)

                if args.agg_type != 'rso':
                    client_model.load_state_dict(global_dict, strict=False)

                client_model, loss = train_client(client_model, client_dataloaders[i], args)
                train_loss += loss
                if args.agg_type == 'rso':
                    client_models.append(deepcopy(client_model.merge_and_unload().state_dict()))
                else:
                    client_models.append(deepcopy(client_model.state_dict()))
        if wandb.run:
            wandb.log(
                {
                    "round": r,
                    f"train_loss": train_loss / len(index_list),
                }
            )
        if args.agg_type == "normal":
            global_model = aggregate_models_normal(global_model, client_models, args)
        elif args.agg_type == "ffa":
            global_model = aggregate_models_ffa(global_model, client_models)
        elif args.agg_type == 'rso':
            global_model = aggregate_models_RSO(base_model, client_models, args)
            base_model = global_model
        elif args.agg_type == 'flexlora':
            global_model = aggregate_models_flexlora(global_model, client_models, args)
        elif args.agg_type == 'flora':
            global_model = aggregate_models_flora(base_model, client_models, args)
            base_model = global_model
        else:
            raise Exception('agg_type is wrong')
        max_metric_1, max_metric_2 = evaluate_global_model(
            global_model, val_dataloader, args, r+1, max_metric_1, max_metric_2
        )


# Main execution
if __name__ == "__main__":
    start = time.time()
    task = args.task
    federated_learning(task)
    print(f'Total time: {time.time() - start} s')