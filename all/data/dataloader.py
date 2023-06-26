from all.base.base_dataloader import VNPBaseDataLoader
from all.data.dataset import VNPDataset
from all.data import DataCollatorForSeq2Seq, DataCollatorForLanguageModeling


class VNPDataLoader(VNPBaseDataLoader):
    """Vietnamese-Poem Data Loader"""

    def __init__(
        self,
        tokenizer,
        tokenizer_name,
        model=None,
        model_architecture=None,
        batch_size=8,
        num_workers=4,
        shuffle=False,
        device=None,
        **kwargs
    ):
        self.dataset = VNPDataset(tokenizer=tokenizer, model_architecture=model_architecture,
                                  **kwargs)

        if model_architecture == "decoder":
            self.data_collator = DataCollatorForSeq2Seq(
                tokenizer=tokenizer,
                model=model,
            )
        elif model_architecture == "encoder_decoder":
            self.data_collator = DataCollatorForLanguageModeling(
                tokenizer=tokenizer,
                mlm=False,
            )
        else:
            raise Exception("Currently not support others architectures than decoder and encoder-decoder architecture")

        self.device = device

        self.init_kwargs = {
            "dataset": self.dataset,
            "batch_size": batch_size,
            "shuffle": shuffle,
            "num_workers": num_workers,
            "data_collator": self.data_collator,
        }

        super().__init__(**self.init_kwargs)
