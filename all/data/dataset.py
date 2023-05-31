from datasets import load_dataset
from all.base.base_dataset import VNPBaseDataset
from transformers import PreTrainedTokenizerBase
import underthesea


class VNPDataset:
    """Vietnamese-Poem Dataset"""

    def __init__(
            self,
            tokenizer: PreTrainedTokenizerBase,
            tokenizer_name,
            model_architecture="decoder",
            dataset_name="Libosa2707/vietnamese-poem",
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

    ):
        '''
        title: Unfilled
        format: Filled  (should always be True)
        content: Filled
        Recommend combination for type of model:
        Gpt-2: with_title=True, with_format=True, with_1st_sentence=True, with_2nd_sentence=False, is_augment=True
        T5: with_title=True, with_format=True, with_1st_sentence=True/False, with_2nd_sentence=False/True, is_augment=True

        '''
        # assert (
        #     model_architecture=="decoder" and max_target_length is not None and max_target_length==max_source_length,
        # ), "Target length must be equal to source length"

        assert (
            model_architecture == "decoder" or model_architecture == "encoder_decoder"
        ), "Model architecture can only be either 'decoder' or 'encoder_decoder', check your syntax again."

        assert (
                with_title or with_format or with_1st_sentence or with_2nd_sentence
        ), "Currently not support null prompt"

        self.dataset_name = dataset_name
        self.test_size = test_size
        self.valid_size = valid_size
        self.model_architecture = model_architecture

        # prompt format
        self.with_title = with_title
        self.with_format = with_format
        self.with_1st_sentence = with_1st_sentence
        self.with_2nd_sentence = with_2nd_sentence
        self.is_augment = is_augment

        # load tokenizer
        self.tokenizer = tokenizer
        self.max_target_length = max_target_length

        # source sentence format:
        # || làm thơ với tiêu đề: <title> <eos> thể thơ: <genre> <eos> <sentence_1&2> ||
        # calculate maximum length of source sequence
        if tokenizer_name in ["bartpho-word", "bartpho-syllable"]:
            max_all = (
                    len(self.tokenizer("làm thơ với tiêu đề:")["input_ids"]) - 2
                    + max_title_length
                    + 1
                    + len(self.tokenizer("thể thơ:")["input_ids"]) - 2
                    + max_format_length
            )
        elif tokenizer_name == "gpt2":
            max_all = (
                    len(self.tokenizer("làm thơ với tiêu đề:")["input_ids"])
                    + max_title_length
                    + 1
                    + len(self.tokenizer("thể thơ:")["input_ids"])
                    + max_format_length
            )
        else:
            max_all = (
                    len(self.tokenizer("làm thơ với tiêu đề:")["input_ids"]) - 1
                    + max_title_length
                    + 1
                    + len(self.tokenizer("thể thơ:")["input_ids"]) - 1
                    + max_format_length
            )

        if self.model_architecture == "decoder":
            add_max = 0
            if with_1st_sentence:
                add_max = (
                        1
                        + max_sentence_length
                )
            if with_2nd_sentence:
                add_max = (
                        1
                        + 2 * max_sentence_length
                )
            max_all += add_max
        else:
            if self.is_augment:
                max_all += (
                        1
                        + 2 * max_sentence_length
                )
            else:
                add_max = 0
                if with_1st_sentence:
                    add_max = (
                            1
                            + max_sentence_length
                    )
                if with_2nd_sentence:
                    add_max = (
                            1
                            + 2 * max_sentence_length
                    )
                max_all += add_max

        self.max_source_length = (
            max_source_length if max_source_length > max_all else max_all
        )
        if model_architecture == "decoder":
            self.max_source_length = self.max_source_length + self.max_target_length

        self.dataset = self.prepare_dataset()

    def get_train_dataset(self):
        return VNPBaseDataset(self.dataset["train"])

    def get_test_dataset(self):
        return VNPBaseDataset(self.dataset["test"])

    def get_valid_dataset(self):
        return VNPBaseDataset(self.dataset["valid"])

    def prepare_dataset(self):
        # load raw dataset
        raw_dataset = load_dataset(self.dataset_name)

        # data normalization
        process_dataset = self.process_poem(raw_dataset)

        # data tokenization
        token_dataset = process_dataset.map(
            self.tokenization,
            batched=True,
            num_proc=4,
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
            load_from_cache_file=True,
            remove_columns=["id", "content", "title", "url", "genre"],
        )
        return dataset

    def tokenization(self, examples):
        texts = []
        labels = []
        if self.with_title:  # with title
            if self.with_format:  # with "thể thơ"
                if self.with_1st_sentence:                                  # tiêu đề: <tiêu đề> <eos> thể thơ: <thể thơ> <eos> <sent_1> \n
                    for idx in range(len(examples["text"])):                # || <sent_2...n>
                        if (examples["title"][idx] != self.tokenizer.unk_token
                                and examples["genre"][idx] != self.tokenizer.unk_token):
                            texts.append(
                                "làm thơ với tiêu đề: "
                                + examples["title"][idx]
                                + self.tokenizer.eos_token
                                + "thể thơ: "
                                + examples["genre"][idx]
                                + self.tokenizer.eos_token
                                + examples["text"][idx].split("\n")[0]
                                + "\n"
                            )
                            labels.append(
                                "\n".join(examples["text"][idx].split("\n")[1:])
                            )
                if self.with_2nd_sentence:                                  # tiêu đề: <tiêu đề> <eos> thể thơ: <thể thơ> <eos> <sent_1> \n <sent_2> \n
                    for idx in range(len(examples["text"])):                # || <sent_3...n>
                        if (examples["title"][idx] != self.tokenizer.unk_token
                                and examples["genre"][idx] != self.tokenizer.unk_token):
                            texts.append(
                                "làm thơ với tiêu đề: "
                                + examples["title"][idx]
                                + self.tokenizer.eos_token
                                + "thể thơ: "
                                + examples["genre"][idx]
                                + self.tokenizer.eos_token
                                + "\n".join(examples["text"][idx].split("\n")[:2])
                                + "\n"
                            )
                            labels.append(
                                "\n".join(examples["text"][idx].split("\n")[2:])
                            )

        else:  # without title
            if self.with_format:  # with "thể thơ"
                if self.with_1st_sentence:                                  # thể thơ: <thể thơ> <eos> <sent_1> \n
                    for idx in range(len(examples["text"])):                # || <sent_2...n>
                        if examples["genre"][idx] != self.tokenizer.unk_token:
                            texts.append(
                                "làm thơ với thể thơ: " + examples["genre"][idx]
                                + self.tokenizer.eos_token
                                + examples["text"][idx].split("\n")[0]
                                + "\n"
                            )
                            labels.append(
                                "\n".join(examples["text"][idx].split("\n")[1:])
                            )
                if self.with_2nd_sentence:                                  # thể thơ: <thể thơ> <eos> <sent_1> \n <sent_2> \n
                    for idx in range(len(examples["text"])):                # || <sent_3...n>
                        if examples["genre"][idx] != self.tokenizer.unk_token:
                            texts.append(
                                "làm thơ với thể thơ: " + examples["genre"][idx]
                                + self.tokenizer.eos_token
                                + "\n".join(examples["text"][idx].split("\n")[:2])
                                + "\n"
                            )
                            labels.append(
                                "\n".join(examples["text"][idx].split("\n")[2:])
                            )

        if self.is_augment:
            if self.model_architecture == "decoder":  # <sent_1> \n <sent_2> \n || <sent_3...n>
                for idx in range(len(examples["text"])):
                    texts.append(
                        "\n".join(examples["text"][idx].split("\n")[:2])
                        + "\n"
                    )
                    labels.append(
                        "\n".join(examples["text"][idx].split("\n")[2:])
                    )
            elif self.model_architecture == "encoder_decoder":
                if self.with_1st_sentence:                              # thể thơ: <thể thơ> <eos> <sent_1> \n <sent_2> \n  || <sent_2...n>
                    for idx in range(len(examples["text"])):
                        if examples["genre"][idx] != self.tokenizer.unk_token:
                            texts.append(
                                "làm thơ với thể thơ: " + examples["genre"][idx]
                                + self.tokenizer.eos_token
                                + "\n".join(examples["text"][idx].split("\n")[:2])
                                + "\n"
                            )
                            labels.append(
                                "\n".join(examples["text"][idx].split("\n")[2:])
                            )
                if self.with_2nd_sentence:                              # thể thơ: <thể thơ> <eos> <sent_1> \n || <sent_2...n>
                    for idx in range(len(examples["text"])):
                        if examples["genre"][idx] != self.tokenizer.unk_token:
                            texts.append(
                                "làm thơ với thể thơ: " + examples["genre"][idx]
                                + self.tokenizer.eos_token
                                + examples["text"][idx].split("\n")[0]
                                + "\n"
                            )
                            labels.append(
                                "\n".join(examples["text"][idx].split("\n")[1:])
                            )
        # in case batch has no data
        if len(texts) == 0:
            texts = ["\n".join(examples["text"][0].split("\n")[:2])
                     + "\n"]
            labels = ["\n".join(examples["text"][0].split("\n")[2:])
                      ]

        if self.model_architecture == "decoder":
            texts = [texts[idx] + labels[idx] for idx in range(len(labels))]

        model_inputs = self.tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=self.max_source_length,
            return_tensors="pt",
        )

        if self.model_architecture == "encoder_decoder":
            with self.tokenizer.as_target_tokenizer():
                model_inputs["labels"] = self.tokenizer(
                    labels,
                    max_length=self.max_target_length,
                    truncation=True,
                    padding="max_length",
                    return_tensors="pt",
                )["input_ids"]
        else:
            model_inputs["labels"] = model_inputs["input_ids"]

        return model_inputs
