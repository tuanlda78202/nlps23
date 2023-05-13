from __future__ import print_function, division
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import PreTrainedTokenizerBase


class VNPDataset(Dataset):
    def __init__(self, dataset_name="Libosa2707/vietnamese-poem"):
        self.raw_dataset = load_dataset(dataset_name)

        self.dataset = self.raw_dataset["train"].train_test_split(
            test_size=0.1, seed=42, shuffle=True
        )

        self.dataset["val"] = self.dataset.pop("test")

    def __len__(self):
        return len(self.raw_dataset)

    def __getitem__(self, idx):
        text = self.dataset["train"]["content"][idx]
        genre = self.dataset["train"]["genre"][idx]
        idx = np.array([idx])

        return {"idx": idx, "text": text, "genre": genre}

    def tokenized(self):
        """Tokenize the dataset"""

        def process(example):
            text = PreTrainedTokenizerBase.clean_up_tokenization(
                example["content"].strip()
            )

            return {"text": text, "genre": example["genre"]}

        token = self.dataset.map(
            process,
            remove_columns=["id", "content", "title", "url", "genre"],
            num_proc=4,
        )

        return token


class VNPDataLoader(DataLoader):
    """Vietnamese-Poem Data Loader"""

    def __init__(
        self, dataset=None, shuffle=True, batch_size=8, block_size=8, device="cpu"
    ):
        self.dataset = VNPDataset().tokenized()
        self.batch_size = batch_size
        self.block_size = block_size
        self.device = device

        super().__init__(self.dataset)

    def get_batch(self, split):
        data = self.dataset["train"] if split == "train" else self.dataset["val"]

        idx = torch.randint(len(data) - self.block_size, (self.batch_size,))

        input = torch.stack(
            [
                torch.from_numpy((data[i : i + self.block_size]).astype(np.int64))
                for i in idx
            ]
        )

        label = torch.stack(
            [
                torch.from_numpy(
                    (data[i + 1 : i + 1 + self.block_size]).astype(np.int64)
                )
                for i in idx
            ]
        )

        if self.device == "cuda":
            # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
            input = input.pin_memory().to(self.device, non_blocking=True)
            label = label.pin_memory().to(self.device, non_blocking=True)

        return input, label
