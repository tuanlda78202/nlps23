from __future__ import print_function, division
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, DatasetDict
from transformers import (
    PreTrainedTokenizerBase,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
)


class VNPDataset(Dataset):
    """Vietnamese-Poem Dataset"""

    def __init__(self, dataset_name="Libosa2707/vietnamese-poem"):
        self.raw_dataset = load_dataset(dataset_name)
        self.tokenizer = AutoTokenizer.from_pretrained(
            "vinai/bartpho-word", use_fast=True
        )

        self.dataset = self.process_poem(self.raw_dataset)

        self.dataset = self.dataset.map(self.tokenization, batched=False, num_proc=4)

        self.dataset = self.dataset.remove_columns(["text", "token_type_ids"])

    def __len__(self):
        return len(self.raw_dataset)

    def __getitem__(self, idx):
        text = self.dataset["text"][idx]
        genre = self.dataset["genre"][idx]
        idx = np.array([idx])

        return {"idx": idx, "text": text, "genre": genre}

    def process_poem(self, ds):
        """Tokenize the dataset"""

        process = lambda example: {
            "text": PreTrainedTokenizerBase.clean_up_tokenization(
                example["content"].strip()
            ),
            "genre": example["genre"],
        }

        processed = ds.map(
            process,
            remove_columns=["id", "content", "title", "url", "genre"],
            num_proc=4,
        )

        return processed

    def tokenization(self, example):
        text = self.tokenizer.bos_token + example["text"] + self.tokenizer.eos_token

        tokenized_text = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )

        tokenized_text["genre"] = example["genre"]

        tokenized_text["input_ids"][
            tokenized_text["input_ids"] == self.tokenizer.pad_token_id
        ] = -100

        return tokenized_text


class VNPDataLoader(DataLoader):
    """Vietnamese-Poem Data Loader"""

    def __init__(
        self,
        dataset=None,
        batch_size=8,
        num_workers=4,
        shuffle=False,
    ):
        # Dataset
        self.ds = VNPDataset()

        self.data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.ds.tokenizer, mlm_probability=False
        )

        split_dataset = self.ds.dataset["train"].train_test_split(
            test_size=0.1, seed=42, shuffle=True
        )

        self.dataset = DatasetDict(
            {"val": split_dataset["test"], "train": split_dataset["train"]}
        )

        super().__init__(
            self.dataset,
            batch_size,
            num_workers=num_workers,
            shuffle=shuffle,
            collate_fn=self.data_collator,
        )
