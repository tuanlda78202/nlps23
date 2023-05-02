from __future__ import print_function, division
import numpy as np
from torch.utils.data import Dataset
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

        tokenized = self.dataset.map(
            process,
            remove_columns=["id", "content", "title", "url", "genre"],
            num_proc=2,
        )

        return tokenized

    def concat_bin(self):
        """
        Concatenate all the ids in each dataset into one large file

        Read the bin files later:
        m = np.memmap('train.bin', dtype=np.uint16, mode='r')
        """
        pass


class VNPDataLoader(DataLoader):
    """Vietnamese-Poem Data Loader"""

    def __init__(self, dataset=None, shuffle=True, batch_size=4, block_size=4):
        self.dataset = VNPDataset()
        self.batch_size = batch_size
        self.block_size = block_size

        super().__init__(self.dataset)
