# Save the vietnamese-poem dataset to binary file for training
import os
from tqdm import tqdm
import numpy as np
import tiktoken
from datasets import load_dataset

num_proc = 2

# Takes 123M in HuggingFace .cache dir, about 171k documents (171,183)
dataset = load_dataset("Libosa2707/vietnamese-poem")

split_dataset = dataset["train"].train_test_split(test_size=0.1, seed=42, shuffle=True)
split_dataset["val"] = split_dataset.pop("test")

# GPT2 BPE
enc = tiktoken.get_encoding("gpt2")


def process(example):
    ids = enc.encode_ordinary(
        example["content"]
    )  # encode_ordinary ignores any special tokens

    ids.append(enc.eot_token)  # add the end of text token, e.g. 50256 for gpt2 bpe

    out = {"ids": ids, "len": len(ids)}

    return out


# Tokenize the dataset
tokenized = split_dataset.map(
    process,
    remove_columns=["id", "content", "title", "url", "genre"],
    desc="Tokenizing the splits",
    num_proc=num_proc,
)

# Concatenate all the ids in each dataset into one large file
for split, dset in tokenized.items():
    arr_len = np.sum(dset["len"])
    filename = os.path.join(os.path.dirname(__file__), f"{split}.bin")
    dtype = np.uint16  # (can do since enc.max_token_value == 50256 is < 2**16)
    arr = np.memmap(filename, dtype=dtype, mode="w+", shape=(arr_len,))
    total_batches = 1024

    idx = 0
    for batch_idx in tqdm(range(total_batches), desc=f"writing {filename}"):
        # Batch together samples for faster write
        batch = dset.shard(
            num_shards=total_batches, index=batch_idx, contiguous=True
        ).with_format("numpy")
        arr_batch = np.concatenate(batch["ids"])
        # Write into mmap
        arr[idx : idx + len(arr_batch)] = arr_batch
        idx += len(arr_batch)
    arr.flush()

# Read the bin files later
# m = np.memmap('train.bin', dtype=np.uint16, mode='r')
