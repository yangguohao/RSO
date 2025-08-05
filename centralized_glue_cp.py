import os
import argparse
import random
import warnings
import numpy as np
import wandb

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, RobertaConfig

from train_eval_cp import centralized_glue
from data_utils_cp import load_and_preprocess_data
from models_cp import create_RSO_model, create_peft_model

warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser(description="Federated Learning with LoRA")

parser.add_argument("--task", type=str, default="cola", help="GLUE task to fine-tune on")
parser.add_argument("--model", type=str, default="roberta-base", help="Model name")
parser.add_argument("--warmup_ratio", type=float, default=0.06, help="Warmup ratio")
parser.add_argument("--max_seq_length", type=int, default=128, help="Maximum sequence length")
parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
parser.add_argument("--lr", type=float, default=3e-5, help="Learning rate")
parser.add_argument("--seed", type=int, default=42, help="Random seed")
parser.add_argument("--device", type=int, default=0, help="device")
parser.add_argument("--weight_decay", type=float, default=0.01)
parser.add_argument("--epochs", type=int, default=30, help="Number of local steps")
parser.add_argument("--method", type=str, default="rso", help="fft, lora, rso")

parser.add_argument("--lora_r", type=int, default=16, help="LoRA R value")
parser.add_argument("--lora_alpha", type=int, default=16, help="LoRA alpha value")
parser.add_argument("--lora_dropout", type=float, default=0., help="LoRA dropout value")
parser.add_argument("--rslora", action="store_true", help="Use RSLoRA")
parser.add_argument("--wandb", action="store_true", help="Use wandb to record training")

parser.add_argument("--optim", type=str, default='SGD', help='SGD, SGDM, AdamW')
parser.add_argument("--interval", type=int, default=1, help="Interval change random projection matrix for rso method")
parser.add_argument("--optimizer_states", type=str, default='reset', help='reset, unchanged, transform, for momentum or Adam-like optimizer')
parser.add_argument("--proj_type", type=str, default='left', help='left, right, specifically for Apollo optimizer')

args = parser.parse_args()

if args.wandb:
    wandb.init(project=f"{args.task}_bs{args.batch_size}_epochs{args.local_epochs}_test", config=args)

random.seed(args.seed)
np.random.seed(args.seed)
os.environ['PYTHONHASHSEED'] = str(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
torch.backends.cudnn.deterministic = True
args.device = torch.device(f'cuda:{args.device}')


def centralized_learning(task):
    train_data, val_data, test_data = load_and_preprocess_data(task, args)

    num_labels = len(set(train_data["labels"].numpy()))

    if args.task == "stsb":
        num_labels = 1

    client_dataloaders = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, pin_memory=True,)

    val_dataloader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False, pin_memory=True)

    base_model = AutoModelForSequenceClassification.from_pretrained(
        args.model,
        num_labels=num_labels
        # config=RobertaConfig.from_pretrained(
        #     "roberta-base",
        #     attention_probs_dropout_prob=0.0,
        #     hidden_dropout_prob=0.0,
        #     classifier_dropout=0.0,  # 有的任务模型支持这个
        #     num_labels=num_labels,
        # )
    )

    base_model.to(args.device)
    if 'rso' in args.method:
        model = create_RSO_model(base_model, args)
    else:
        if args.method == 'fft':
            args.lora_r = args.lora_alpha = 0
        model = create_peft_model(base_model, args)

    # train_rso(model, client_dataloaders, val_dataloader, args)
    centralized_glue(model, client_dataloaders, val_dataloader, args)


# Main execution
if __name__ == "__main__":
    task = args.task
    centralized_learning(task)
