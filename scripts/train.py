import sys
import os
import warnings
import gc


sys.path.append(os.getcwd())
warnings.filterwarnings("ignore")

import argparse
import collections
import torch
import numpy as np
from utils import util

import all.data.dataset as module_data
import all.data.collate_fn as module_collator
import all.evaluation as module_eval

from configs.parse_config import ConfigParser
import transformers
from transformers import GPT2LMHeadModel
from transformers import AutoModelForSeq2SeqLM


SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True

model_name = "t5"

# model_architecture = "encoder_decoder"

def main(config):
    logger = config.get_logger("train")

    tokenizer_name = config['dataset']['args']['tokenizer_name']
    tokenizer = util.get_tokenizer(tokenizer_name=tokenizer_name)

    # data_loader = config.init_obj("dataloader", module_data)
    dataset = config.init_ftn("dataset", module_data)
    dataset = dataset(tokenizer=tokenizer, tokenizer_name=tokenizer_name)

    train_dataset = dataset.get_train_dataset()
    valid_dataset = dataset.get_valid_dataset()
    test_dataset = dataset.get_test_dataset()
    del dataset
    gc.collect()

    # model = config.init_obj("arch", module_arch)
    # logger.info(model)
    #
    # model = model.to(device)
    if model_name == "gpt2":
        model = GPT2LMHeadModel.from_pretrained('NlpHUST/gpt2-vietnamese').cuda()
        model.resize_token_embeddings(len(tokenizer))
    elif model_name == "t5":
        model = AutoModelForSeq2SeqLM.from_pretrained("VietAI/vit5-base")
        model.resize_token_embeddings(len(tokenizer))
    elif model_name == "hmm":
        model = ...
    elif model_name == "difflm":
        model = ...
    else:
        assert f"Model {model_name} is not exists, check your model type again."

    # get data collator
    data_collator = config.init_ftn("collator", module_collator)
    data_collator = data_collator(model=model, tokenizer=tokenizer)

    training_args = config.init_obj("training_arguments", transformers)

    trainer = getattr(transformers, config['trainer']['type'])(model=model,
                                                               args=training_args,
                                                               data_collator=data_collator,
                                                               train_dataset=train_dataset,
                                                               eval_dataset=valid_dataset,
                                                               tokenizer=tokenizer,
                                                               # compute_metrics=compute_metrics,
                                                               # optimizer=optimizer,
                                                               )

    trainer.train()

    test_samples = config.init_ftn("evaluation", module_eval)
    test_samples = test_samples(tokenizer=tokenizer, trainer=trainer)
    test_samples.generate(test_dataset=test_dataset)


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
        default="cuda",
        type=str,
        help="type of device",
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
