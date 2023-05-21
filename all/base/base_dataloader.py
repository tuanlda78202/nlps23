from torch.utils.data import DataLoader


class VNPBaseDataLoader(DataLoader):
    """
    Base class for all data loaders
    """

    def __init__(
        self, dataset: VNPDataset, batch_size, shuffle, data_collator, num_workers=1
    ):
        self.shuffle = shuffle

        # Split Train & Valid & Test
        dataset.get_train_dataset()
        self.sampler = dataset.get_data
        dataset.get_valid_dataset()
        self.valid_sampler = dataset.get_data
        dataset.get_test_dataset()
        self.test_sampler = dataset.get_data

        self.init_kwargs = {
            "batch_size": batch_size,
            "shuffle": self.shuffle,
            "num_workers": num_workers,
            "collate_fn": data_collator,
        }
        super().__init__(dataset=self.sampler, **self.init_kwargs)

    def get_validation(self):
        if self.valid_sampler is None:
            return None
        else:
            return DataLoader(dataset=self.valid_sampler, **self.init_kwargs)

    def get_test(self):
        if self.valid_sampler is None:
            return None
        else:
            return DataLoader(dataset=self.test_sampler, **self.init_kwargs)
