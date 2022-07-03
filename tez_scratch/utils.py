import argparse
import os
import random
import numpy as np
import torch

def seed_everything(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, required=False, default=0)
    parser.add_argument("--model", type=str, required=False, default="microsoft/deberta-base")
    parser.add_argument("--lr", type=float, required=False, default=3e-5)
    parser.add_argument("--output", type=str, default=".", required=False)
    parser.add_argument("--input", type=str, default="../input", required=False)
    parser.add_argument("--max_len", type=int, default=1024, required=False)
    parser.add_argument("--batch_size", type=int, default=2, required=False)
    parser.add_argument("--valid_batch_size", type=int, default=16, required=False)
    parser.add_argument("--epochs", type=int, default=5, required=False)
    parser.add_argument("--accumulation_steps", type=int, default=1, required=False)
    parser.add_argument("--predict", action="store_true", required=False)
    return parser.parse_args()
