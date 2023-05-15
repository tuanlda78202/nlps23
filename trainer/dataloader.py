import numpy as np
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

        self.process_dataset = self.process_poem(self.raw_dataset)

        self.token_dataset = self.process_dataset.map(
            self.tokenization, batched=False, num_proc=4
        )

    def __len__(self):
        return len(self.raw_dataset)

    def __getitem__(self, idx):
        text = self.token_dataset["train"]["text"][idx]
        genre = self.token_dataset["train"]["genre"][idx]
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
        self, dataset=None, batch_size=8, num_workers=4, shuffle=False, device=None
    ):
        raw_ds = VNPDataset()
        ds = raw_ds.token_dataset.remove_columns(["text", "token_type_ids"])

        self.data_collator = DataCollatorForLanguageModeling(
            tokenizer=raw_ds.tokenizer, mlm_probability=False
        )

        split_dataset = ds["train"].train_test_split(
            test_size=0.1, seed=42, shuffle=True
        )

        self.dataset = DatasetDict(
            {"val": split_dataset["test"], "train": split_dataset["train"]}
        )

        self.device = device

        self.init_kwargs = {
            "dataset": self.dataset,
            "batch_size": batch_size,
            "shuffle": shuffle,
            "num_workers": num_workers,
            "collate_fn": self.data_collator,
        }

        super().__init__(**self.init_kwargs)


dataloader = VNPDataLoader()
print(next(iter(dataloader)))
