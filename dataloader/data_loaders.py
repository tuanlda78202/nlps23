import sys, os

sys.path.append(os.getcwd())

from base.base_data_loader import BaseDataLoader


class VNPDataLoader(BaseDataLoader):
    """Vietnamese-Poem Data Loader"""

    def __init__(
        self, output_size, crop_size, batch_size, shuffle, validation_split, num_workers
    ):
        self.output_size = output_size
        self.crop_size = crop_size

        self.img_list, self.mask_list = mask_image_list()
        self.dataset = KNC_Dataset(
            self.img_list,
            self.mask_list,
            transform=transforms.Compose(
                [
                    Rescale(self.output_size),
                    RandomCrop(self.crop_size),
                    ToTensorLab(flag=0),
                ]
            ),
        )

        super().__init__(
            self.dataset, batch_size, shuffle, validation_split, num_workers
        )


# poor man's data loader
data_dir = os.path.join("data", dataset)
train_data = np.memmap(os.path.join(data_dir, "train.bin"), dtype=np.uint16, mode="r")
val_data = np.memmap(os.path.join(data_dir, "val.bin"), dtype=np.uint16, mode="r")


def get_batch(split):
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack(
        [torch.from_numpy((data[i : i + block_size]).astype(np.int64)) for i in ix]
    )
    y = torch.stack(
        [
            torch.from_numpy((data[i + 1 : i + 1 + block_size]).astype(np.int64))
            for i in ix
        ]
    )

    if device_type == "cuda":
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(
            device, non_blocking=True
        )
    else:
        x, y = x.to(device), y.to(device)
    return x, y
