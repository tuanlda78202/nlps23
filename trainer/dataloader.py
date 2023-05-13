from __future__ import print_function, division
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import PreTrainedTokenizerBase


class VNPDataset(Dataset):
    """Vietnamese-Poem Dataset"""

    def __init__(self, dataset_name="Libosa2707/vietnamese-poem"):
        self.raw_dataset = load_dataset(dataset_name)

        self.raw_dataset = self.tokenized()

    def __len__(self):
        return len(self.raw_dataset)

    def __getitem__(self, idx):
        text = self.dataset["text"][idx]
        genre = self.dataset["genre"][idx]
        idx = np.array([idx])

        return {"idx": idx, "text": text, "genre": genre}

    def tokenized(self):
        """Tokenize the dataset"""

        process = lambda example: {
            "text": PreTrainedTokenizerBase.clean_up_tokenization(
                example["content"].strip()
            ),
            "genre": example["genre"],
        }

        token = self.raw_dataset.map(
            process,
            remove_columns=["id", "content", "title", "url", "genre"],
            num_proc=4,
        )

        return token


class VNPDataLoader(DataLoader):
    """Vietnamese-Poem Data Loader"""

    def __init__(
        self,
        dataset=None,
        batch_size=8,
        block_size=8,
        device="cpu",
        num_workers=0,
        shuffle=False,
    ):
        # Dataset
        self.dataset = VNPDataset()
        split_dataset = self.dataset.raw_dataset["train"].train_test_split(
            test_size=0.1, seed=42, shuffle=True
        )

        self.val_dataset = split_dataset.pop("test")
        self.train_dataset = split_dataset["train"]

        # Hyper-parameters
        self.batch_size = batch_size
        self.block_size = block_size
        self.device = device
        self.num_workers = num_workers
        self.shuffle = shuffle

        self.sampler = self.get_batch(split="train")
        self.valid_sampler = self.get_batch(split="val")

        super().__init__(self.dataset, batch_size, device, num_workers, shuffle)

    def get_batch(self, split):
        data = self.train_dataset if split == "train" else self.val_dataset

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

        return (input, label)


x = VNPDataLoader()
y = torch.from_numpy((x.train_dataset[0 : 0 + 256]).astype(np.int64))
print(y)
