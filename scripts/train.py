import sys
import os
import warnings

sys.path.append(os.getcwd())
warnings.filterwarnings("ignore")

import argparse
import collections
import torch
import numpy as np
import trainer.dataloader as module_data
import model as module_arch
from configs.parse_config import ConfigParser
from trainer import GPT2Trainer

from model import GPTConfig

arch = {"n_layer": 12, "n_head": 12, "n_embd": 768, "dropout": 0.0, "bias": False}
print(GPTConfig(**arch))


SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True


def main(config):
    logger = config.get_logger("train")

    data_loader = config.init_obj("dataloader", module_data)
    train_dataloader = data_loader.dataset["train"]
    valid_dataloader = data_loader.dataset["val"]

    model = config.init_obj("arch", module_arch)
    logger.info(model)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    trainer = GPT2Trainer(
        model,
        config=config,
        device=device,
        data_loader=train_dataloader,
        valid_data_loader=valid_dataloader,
    )

    trainer.train()


if __name__ == "__main__":
    args = argparse.ArgumentParser(description="Vietnamese Poem Generator")

    args.add_argument(
        "-c",
        "--config",
        default="configs/gpt2/train_gpt2.yaml",
        type=str,
        help="config file path (default: None)",
    )

    args.add_argument(
        "-r",
        "--resume",
        default=None,
        type=str,
        help="path to latest checkpoint (default: None)",
    )

    args.add_argument(
        "-d",
        "--device",
        default="mps",
        type=str,
        help="indices of GPUs to enable (default: all)",
    )

    CustomArgs = collections.namedtuple("CustomArgs", "flags type target")

    options = [
        CustomArgs(
            ["--bs", "--batch_size"], type=int, target="data_loader;args;batch_size"
        ),
        CustomArgs(["--ep", "--epochs"], type=int, target="trainer;epochs"),
    ]

    config = ConfigParser.from_args(args, options)

    main(config)
