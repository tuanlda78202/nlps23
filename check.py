from all.data.dataloader import VNPDataLoader
from transformers import AutoTokenizer
from tqdm import tqdm

tokenizer = AutoTokenizer.from_pretrained("vinai/bartpho-word")
dataloader = VNPDataLoader(tokenizer=tokenizer)

tqdm_batch = tqdm(
    iterable=dataloader,
    desc="Epoch {}".format(10),
    total=len(dataloader),
    unit="it",
)

for batch_idx, loader in enumerate(tqdm_batch):
    print(batch_idx)
    print(loader)
    if batch_idx == 2:
        break
