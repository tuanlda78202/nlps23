import os
import math
import torch
import yaml
from itertools import repeat
from pathlib import Path
from enum import Enum
from transformers import (
    AutoTokenizer, PreTrainedTokenizerBase,
)


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


class ExplicitEnum(str, Enum):
    """
    Enum with more explicit error message for missing values.
    """

    @classmethod
    def _missing_(cls, value):
        raise ValueError(
            f"{value} is not a valid {cls.__name__}, please select one of {list(cls._value2member_map_.keys())}"
        )


class PaddingStrategy(ExplicitEnum):
    """
    Possible values for the `padding` argument in [`PreTrainedTokenizerBase.__call__`]. Useful for tab-completion in an
    IDE.
    """

    LONGEST = "longest"
    MAX_LENGTH = "max_length"
    DO_NOT_PAD = "do_not_pad"


def get_tokenizer(tokenizer_name) -> PreTrainedTokenizerBase:
    if tokenizer_name == "bartpho-word":
        # add bos and eos at both end, bos || eos || pad || unk differ id, no <\n>
        tokenizer = AutoTokenizer.from_pretrained("vinai/bartpho-word")
    elif tokenizer_name == "bartpho-syllable":
        # add bos and eos at both end, bos || eos || pad || unk differ id, no <\n>
        tokenizer = AutoTokenizer.from_pretrained("vinai/bartpho-syllable")
    elif tokenizer_name == "gpt2":
        # don't add bos and eos at both end, bos || eos || unk same id, no pad, have <\n>>
        tokenizer = AutoTokenizer.from_pretrained("NlpHUST/gpt2-vietnamese")
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    elif tokenizer_name == "t5":
        # only add eos at the end, eos || unk || pad differ id, no bos, no <\n>
        tokenizer = AutoTokenizer.from_pretrained("VietAI/vit5-large-vietnews-summarization")
    else:
        assert Exception(f"Tokenizer {tokenizer_name} is not exist.")

    tokenizer.add_tokens(list({"<\n>"} - tokenizer.get_vocab().keys()))
    return tokenizer


