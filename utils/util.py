import os
import math
import torch
import yaml
from itertools import repeat
from pathlib import Path


def load_yaml(fname):
    fname = Path(fname)
    with fname.open("rt") as file:
        config = yaml.safe_load(file)
    return config


def write_yaml(content, fname):
    fname = Path(fname)
    with fname.open("wt") as handle:
        yaml.dump(content, handle, indent=4, sort_keys=False)


def inf_loop(data_loader):
    """wrapper function for endless data loader."""
    for loader in repeat(data_loader):
        yield from loader


def init_wandb(
    wandb_lib,
    project,
    entity,
    api_key_file="./configs/api",
    dir="./saved",
    name=None,
    config=None,
):
    """
    Return a new W&B run to be used for logging purposes
    """
    assert os.path.exists(api_key_file), "The given W&B API key file does not exist"

    # Set environment API & DIR
    api_key_value = open(api_key_file, "r").read().strip()
    os.environ["WANDB_API_KEY"] = api_key_value
    os.environ["WANDB_DIR"] = dir

    # name: user_name in WandB
    return wandb_lib.init(project=project, entity=entity, name=name, config=config)


def get_lr(it, learning_rate, warmup_iters, lr_decay_iters, min_lr):
    # 1) Linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * it / warmup_iters

    # 2) If it > lr_decay_iters, return min lr
    if it > lr_decay_iters:
        return min_lr

    # 3) In between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1

    # Coef ranges [0,1]
    coef = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))

    return min_lr + coef * (learning_rate - min_lr)


def new_gelu(x):
    """GELU activation function"""

    return (
        0.5
        * x
        * (
            1.0
            + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0)))
        )
    )
