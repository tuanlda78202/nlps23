import sys
import os
import warnings
import gc
import utils

sys.path.append(os.getcwd())
warnings.filterwarnings("ignore")

import argparse
import collections
import torch
import numpy as np

from all.data.dataloader import VNPDataLoader
import all.data.dataset as module_data
import all.data.collate_fn as module_collator

import all.model as module_arch
from configs.parse_config import ConfigParser
from all.trainer import GPT2Trainer
from transformers import AutoTokenizer, TrainingArguments, Seq2SeqTrainingArguments, Trainer, Seq2SeqTrainer
from transformers import GPT2LMHeadModel, GPT2Config
from transformers import AutoModelForSeq2SeqLM


SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True

model_name = "gpt2"
# tokenizer_name = "t5"
# model_architecture = "encoder_decoder"

def main(config):
    logger = config.get_logger("train")

    tokenizer_name = config.dataset['args']['tokenizer_name']
    tokenizer = utils.get_tokenizer(tokenizer_name=tokenizer_name)

    # data_loader = config.init_obj("dataloader", module_data)
    dataset = config.init_ftn("dataset", module_data)
    dataset = dataset(tokenizer=tokenizer, tokenizer_name=tokenizer_name)
    #     VNPDataset(
    #         tokenizer,
    #         tokenizer_name,
    #         valid_size=0.1,
    #         test_size=0.1,
    #         model_architecture="decoder",
    #         max_title_length=-1,
    #         max_format_length=-1,
    #         max_sentence_length=-1,
    #         max_source_length=42,
    #         max_target_length=384,
    #         with_title=True,
    #         with_format=True,
    #         with_1st_sentence=False,
    #         with_2nd_sentence=False,
    #         is_augment=False,
    #         dataset_name="Libosa2707/vietnamese-poem",
    # )
    train_dataset = dataset.get_train_dataset()
    valid_dataset = dataset.get_valid_dataset()
    test_dataset = dataset.get_test_dataset()
    del dataset
    gc.collect()

    # # model = config.init_obj("arch", module_arch)
    # # logger.info(model)
    # #
    # # model = model.to(device)
    # if model_name == "gpt2":
    #     model = GPT2LMHeadModel.from_pretrained('NlpHUST/gpt2-vietnamese').cuda()
    #     model.resize_token_embeddings(len(tokenizer))
    # elif model_name == "t5":
    #     model = AutoModelForSeq2SeqLM.from_pretrained("VietAI/vit5-base")
    # elif model_name == "hmm":
    #     model = ...
    # elif model_name == "difflm":
    #     model = ...
    # else:
    #     assert f"Model {model_name} is not exists, check your model type again."
    #
    # # get data collator
    # data_collator = get_data_collator(model_architecture=model_name,
    #                                   tokenizer=tokenizer,
    #                                   model=model)
    # if model_architecture == "decoder":
    #     training_args = TrainingArguments(
    #         output_dir=args,
    #         overwrite_output_dir=True,
    #         num_train_epochs=20,
    #         per_device_train_batch_size=8,
    #         per_device_eval_batch_size=16,
    #         save_steps=1000,
    #         save_total_limit=2,
    #         warmup_steps=1000,
    #         logging_steps=100,
    #         report_to="wandb",
    #         )
    #
    #     trainer = Trainer(
    #         model=model,
    #         args=training_args,
    #         train_dataset=train_dataset,
    #         device=device,
    #         config=config,
    #         data_loader=train_dataloader,
    #         valid_dataloader=valid_dataloader,
    #     )
    #
    # elif model_architecture == "encoder_decoder":
    #     training_args = Seq2SeqTrainingArguments(
    #         output_dir=config.args,
    #         overwrite_output_dir=True,
    #         num_train_epochs=20,
    #         per_device_train_batch_size=8,
    #         per_device_eval_batch_size=16,
    #         save_steps=1000,
    #         save_total_limit=2,
    #         warmup_steps=1000,
    #         logging_steps=100,
    #         report_to="wandb",
    #     )
    #
    #     trainer = Seq2SeqTrainer(
    #         model,
    #         args,
    #         train_dataset=train_dataset,
    #         eval_dataset=valid_dataset,
    #         data_collator=data_collator,
    #         tokenizer=tokenizer,
    #         # compute_metrics=compute_metrics
    #     )
    #
    # trainer.train()


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
