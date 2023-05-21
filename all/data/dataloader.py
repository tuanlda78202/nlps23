from all.base.base_dataloader import VNPBaseDataLoader
from all.data.vnpdataset import VNPDataset
from transformers import DataCollatorForSeq2Seq


class VNPDataLoader(VNPBaseDataLoader):
    """Vietnamese-Poem Data Loader"""

    def __init__(
        self,
        tokenizer,
        batch_size=8,
        num_workers=4,
        shuffle=False,
        device=None,
        **kwargs
    ):
        self.dataset = VNPDataset(tokenizer=tokenizer, **kwargs)

        self.data_collator = DataCollatorForSeq2Seq(
            tokenizer=tokenizer,
        )

        self.device = device

        self.init_kwargs = {
            "dataset": self.dataset,
            "batch_size": batch_size,
            "shuffle": shuffle,
            "num_workers": num_workers,
            "data_collator": self.data_collator,
        }

        super().__init__(**self.init_kwargs)
