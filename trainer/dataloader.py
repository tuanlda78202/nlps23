from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
from transformers import DataCollatorForLanguageModeling, DataCollatorForSeq2Seq
from transformers import PreTrainedTokenizerBase, AutoTokenizer
import underthesea


class VNPDataset(Dataset):
    """Vietnamese-Poem Dataset"""

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        valid_size=0.1,
        test_size=0.1,
        max_title_length=-1,
        max_format_length=-1,
        max_sentence_length=-1,
        max_source_length=42,
        max_target_length=384,
        with_title=True,
        with_format=True,
        with_1st_sentence=False,
        with_2nd_sentence=False,
        is_augment=False,
        dataset_name="Libosa2707/vietnamese-poem",
    ):
        assert (
            with_title or with_format or with_1st_sentence or with_2nd_sentence
        ), "Currently not support null prompt"

        self.dataset_name = dataset_name
        self.test_size = test_size
        self.valid_size = valid_size

        # prompt format
        self.with_title = with_title
        self.with_format = with_format
        self.with_1st_sentence = with_1st_sentence
        self.with_2nd_sentence = with_2nd_sentence
        self.is_augment = is_augment

        # load tokenizer
        self.tokenizer = tokenizer
        self.max_source_length = max_source_length

        # source sentence format: làm thơ với tiêu đề: <title> <eos> tuân theo thể thơ: <genre> <eos> hai câu đầu: <sentence_1&2>
        max_all = (
            len(self.tokenizer("làm thơ với tiêu đề:")["input_ids"])
            - 2
            + max_title_length
            + 1
            + len(self.tokenizer("tuân theo thể thơ:")["input_ids"])
            - 2
            + max_format_length
        )
        if with_2nd_sentence:
            max_all += (
                1
                + len(self.tokenizer("hai câu đầu:")["input_ids"])
                - 2
                + 2 * max_sentence_length
            )
        else:
            if with_1st_sentence:
                max_all += (
                    1
                    + len(self.tokenizer("câu đầu:")["input_ids"])
                    - 2
                    + max_sentence_length
                )

        self.max_target_length = (
            max_target_length if max_target_length > max_all else max_all
        )

        self.dataset = self.prepare_dataset()

    def __len__(self):
        return len(self.raw_dataset["train"])

    def __getitem__(self, idx):
        return self.get_data[idx]

    def get_train_dataset(self):
        self.get_data = self.dataset["train"]

    def get_test_dataset(self):
        self.get_data = self.dataset["test"]

    def get_valid_dataset(self):
        self.get_data = self.dataset["valid"]

    def prepare_dataset(self):
        # load raw dataset
        raw_dataset = load_dataset(self.dataset_name)

        # data normalization
        process_dataset = self.process_poem(raw_dataset)

        # data tokenization
        token_dataset = process_dataset.map(
            self.tokenization,
            batched=True,
            num_proc=2,
            remove_columns=["title", "genre", "text"],
        )

        dataset = token_dataset["train"].train_test_split(
            test_size=self.test_size, seed=42
        )
        dataset["train"], dataset["valid"] = (
            dataset["train"]
            .train_test_split(test_size=self.valid_size / (1 - self.test_size), seed=42)
            .values()
        )

        return dataset

    def process_poem(self, ds):
        """Strip and remove space"""

        def process_poem_text(examples):
            texts = [
                PreTrainedTokenizerBase.clean_up_tokenization(example.strip())
                for example in examples["content"]
            ]
            genres = [
                PreTrainedTokenizerBase.clean_up_tokenization(example.strip())
                if example is not None
                else self.tokenizer.unk_token
                for example in examples["genre"]
            ]
            titles = [
                " ".join(underthesea.word_tokenize(example.strip().lower()))
                if example is not None
                else self.tokenizer.unk_token
                for example in examples["title"]
            ]
            return {"text": texts, "genre": genres, "title": titles}

        dataset = ds.map(
            process_poem_text,
            batched=True,
            num_proc=4,
            load_from_cache_file=False,
            remove_columns=["id", "content", "title", "url", "genre"],
        )
        return dataset

    def tokenization(self, examples):
        texts = []
        labels = []
        if self.with_title:  # with title
            if self.with_format:  # with "thể thơ"
                if self.with_1st_sentence:
                    texts += [
                        "làm thơ với tiêu đề: "
                        + examples["title"][idx]
                        + self.tokenizer.eos_token
                        + "tuân theo thể thơ: "
                        + examples["genre"][idx]
                        + self.tokenizer.eos_token
                        + "câu đầu:"
                        + examples["text"][idx].split("\n")[0]
                        + "\n"
                        for idx in range(len(examples["text"]))
                    ]
                    labels += [
                        "\n".join(examples["text"][idx].split("\n")[1:])
                        for idx in range(len(examples["text"]))
                    ]
                if self.with_2nd_sentence:
                    texts += [
                        "làm thơ với tiêu đề: "
                        + examples["title"][idx]
                        + self.tokenizer.eos_token
                        + "tuân theo thể thơ: "
                        + examples["genre"][idx]
                        + self.tokenizer.eos_token
                        + "hai câu đầu: "
                        + "\n".join(examples["text"][idx].split("\n")[:2])
                        + "\n"
                        for idx in range(len(examples["text"]))
                    ]
                    labels += [
                        "\n".join(examples["text"][idx].split("\n")[2:])
                        for idx in range(len(examples["text"]))
                    ]

                texts += [
                    "làm thơ với tiêu đề: "
                    + examples["title"][idx]
                    + self.tokenizer.eos_token
                    + "tuân theo thể thơ: "
                    + examples["genre"][idx]
                    for idx in range(len(examples["text"]))
                ]
                labels += examples["text"]

                if self.is_augment:
                    texts += [
                        "tuân theo thể thơ: " + examples["genre"][idx]
                        for idx in range(len(examples["text"]))
                    ]
                    labels += examples["text"]
            else:  # without "thể thơ"
                if self.with_1st_sentence:
                    texts += [
                        "làm thơ với tiêu đề: "
                        + examples["title"][idx]
                        + self.tokenizer.eos_token
                        + "câu đầu:"
                        + examples["text"][idx].split("\n")[0]
                        + "\n"
                        for idx in range(len(examples["text"]))
                    ]
                    labels += [
                        "\n".join(examples["text"][idx].split("\n")[1:])
                        for idx in range(len(examples["text"]))
                    ]
                if self.with_2nd_sentence:
                    texts += [
                        "làm thơ với tiêu đề: "
                        + examples["title"][idx]
                        + self.tokenizer.eos_token
                        + "hai câu đầu:"
                        + "\n".join(examples["text"][idx].split("\n")[:2])
                        + "\n"
                        for idx in range(len(examples["text"]))
                    ]
                    labels += [
                        "\n".join(examples["text"][idx].split("\n")[2:])
                        for idx in range(len(examples["text"]))
                    ]

                texts += [
                    "làm thơ với tiêu đề:" + examples["title"][idx]
                    for idx in range(len(examples["text"]))
                ]
                labels += examples["text"]
        else:  # without title
            if self.with_format:  # with "thể thơ"
                if self.with_1st_sentence:
                    texts += [
                        "tuân theo thể thơ: "
                        + examples["genre"][idx]
                        + self.tokenizer.eos_token
                        + "câu đầu:"
                        + examples["text"][idx].split("\n")[0]
                        + "\n"
                        for idx in range(len(examples["text"]))
                    ]
                    labels += [
                        "\n".join(examples["text"][idx].split("\n")[1:])
                        for idx in range(len(examples["text"]))
                    ]
                if self.with_2nd_sentence:
                    texts += [
                        "tuân theo thể thơ: "
                        + examples["genre"][idx]
                        + self.tokenizer.eos_token
                        + "hai câu đầu:"
                        + "\n".join(examples["text"][idx].split("\n")[:2])
                        + "\n"
                        for idx in range(len(examples["text"]))
                    ]
                    labels += [
                        "\n".join(examples["text"][idx].split("\n")[2:])
                        for idx in range(len(examples["text"]))
                    ]

                texts += [
                    "tuân theo thể thơ: " + examples["genre"][idx]
                    for idx in range(len(examples["text"]))
                ]
                labels += examples["text"]
            else:  # without "thể thơ"
                if self.with_1st_sentence:
                    texts += [
                        "câu đầu:" + examples["text"][idx].split("\n")[0] + "\n"
                        for idx in range(len(examples["text"]))
                    ]
                    labels += [
                        "\n".join(examples["text"][idx].split("\n")[1:])
                        for idx in range(len(examples["text"]))
                    ]
                if self.with_2nd_sentence:
                    texts += [
                        "hai câu đầu:"
                        + "\n".join(examples["text"][idx].split("\n")[:2])
                        + "\n"
                        for idx in range(len(examples["text"]))
                    ]
                    labels += [
                        "\n".join(examples["text"][idx].split("\n")[2:])
                        for idx in range(len(examples["text"]))
                    ]

        model_inputs = self.tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=self.max_source_length,
            return_tensors="pt",
        )
        model_inputs["labels"] = self.tokenizer(
            labels,
            max_length=self.max_target_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )["input_ids"]

        return model_inputs


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
            "dataset": dataset,
            "batch_size": batch_size,
            "shuffle": self.shuffle,
            "num_workers": num_workers,
            "collate_fn": data_collator,
        }
        super().__init__(sampler=self.sampler, **self.init_kwargs)

    def get_validation(self):
        if self.valid_sampler is None:
            return None
        else:
            return DataLoader(sampler=self.valid_sampler, **self.init_kwargs)

    def get_test(self):
        if self.valid_sampler is None:
            return None
        else:
            return DataLoader(sampler=self.test_sampler, **self.init_kwargs)


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
