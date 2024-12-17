import zstandard
import io, os
import jsonlines

# import simdjson as json
import json
import numpy as np
import torch
from typing import Optional, List
from transformers import AutoTokenizer, GPTNeoXForCausalLM
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

import lightning.pytorch as pl
from transformers import (
    AutoTokenizer,
    DataCollatorForLanguageModeling,
)
from datasets import load_dataset

MODEL_REGISTRY_PATH = os.environ.get("MODEL_REGISTRY_PATH")
LANGUAGE_DATA_PATH = os.environ.get("LANG_DATA_PATH")
GENERAL_DATA_PATH = os.environ.get("MR_DATA_PATH")


class IMDBDataset(Dataset):
    def __init__(
        self,
        max_length: int,
        dataset_name: str = "imdb",
        tokenizer_name_or_path: str = "pythia_all/pythia-70m",
        split: str = "train",
    ):
        self.dataset_name = dataset_name
        self.dataset = load_dataset(
            os.path.join(GENERAL_DATA_PATH, self.dataset_name), split=split
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            os.path.join(MODEL_REGISTRY_PATH, tokenizer_name_or_path),
            clean_up_tokenization_spaces=True,
        )
        self.max_length = max_length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        text = self.dataset[idx]["text"]

        tokens = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
        )

        return {
            "input_ids": tokens["input_ids"].squeeze(),
            "attention_mask": tokens["attention_mask"].squeeze(),
        }


class IMDbDataModule(pl.LightningDataModule):
    def __init__(
        self,
        dataset_name: str,
        tokenizer_name_or_path: str = "pythia_all/pythia-70m",
        batch_size: int = 8,
        max_length: int = 128,
    ):
        super().__init__()
        self.dataset_name = dataset_name
        self.tokenizer_name_or_path = tokenizer_name_or_path
        self.batch_size = batch_size
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(
            os.path.join(MODEL_REGISTRY_PATH, tokenizer_name_or_path)
        )
        self.data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer, mlm=False
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def setup(self, stage=None):
        self.train_dataset = IMDBDataset(
            max_length=self.max_length,
            dataset_name=self.dataset_name,
            tokenizer_name_or_path=self.tokenizer_name_or_path,
            split="train",
        )
        self.val_dataset = IMDBDataset(
            max_length=self.max_length,
            dataset_name=self.dataset_name,
            tokenizer_name_or_path=self.tokenizer_name_or_path,
            split="test",
        )
        self.predict_dataset = IMDBDataset(
            max_length=self.max_length,
            dataset_name=self.dataset_name,
            tokenizer_name_or_path=self.tokenizer_name_or_path,
            split="test",
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=self.data_collator,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            collate_fn=self.data_collator,
        )

    def predict_dataloader(self):
        return DataLoader(
            self.predict_dataset,
            batch_size=self.batch_size,
            collate_fn=self.data_collator,
        )


parser = lambda x: json.loads(x)


def json_parser(x):
    try:
        line = parser.parse(x).as_dict()
        return line
    except ValueError:
        return x


def load_samples(pile_file_ids=None, n_samples=-1):
    pile_folder = os.path.join(LANGUAGE_DATA_PATH, "ThePile/2020")
    if pile_file_ids is None:
        pile_file_ids = [f"{num:02d}" for num in range(30)]
    for pile_file_id in pile_file_ids:  # pile_file_ids = ['00', '01', ...]
        filepath = f"{pile_folder}/{pile_file_id}.jsonl.zst"
        with open(filepath, "rb") as f:
            cctx = zstandard.ZstdDecompressor()
            reader_stream = io.BufferedReader(cctx.stream_reader(f))
            reader = jsonlines.Reader(reader_stream, loads=json_parser)
            for i, item in enumerate(reader):
                if n_samples > 0 and i >= n_samples:  # DBG
                    break
                sample = dict()
                if isinstance(item, str):
                    sample["texts"] = item
                else:
                    text = item["text"]
                    if isinstance(text, list):
                        text = "\n".join(text)
                    sample["texts"] = text
                sample["sample_idx"] = i
                sample["file_id"] = pile_file_id
                yield sample


class PileDataset(Dataset):
    lengths = {
        "00": 7021438,
        "01": 7021384,
        "02": 7023291,
        "03": 7020402,
        "04": 7019557,
        "05": 7017234,
        "06": 7021785,
        "07": 7020726,
        "08": 7019770,
        "09": 7025621,
        "10": 7023114,
        "11": 7014945,
        "12": 7018367,
        "13": 7018349,
        "14": 7019127,
        "15": 7017379,
        "16": 7020002,
        "17": 7024882,
        "18": 7021780,
        "19": 7022886,
        "20": 7020334,
        "21": 7020143,
        "22": 7017674,
        "23": 7022543,
        "24": 7019788,
        "25": 7014418,
        "26": 7020539,
        "27": 7017840,
        "28": 7017988,
        "29": 7024422,
    }

    def __init__(
        self,
        max_length,
        tokenizer_name_or_path="pythia_all/pythia-70m",
        n_samples=-1,
        for_trainer=True,
        pile_file_ids=None,
    ):
        self.samples = load_samples(pile_file_ids=pile_file_ids, n_samples=n_samples)
        self.n_samples = n_samples
        self.pile_file_ids = pile_file_ids
        self.tokenizer = AutoTokenizer.from_pretrained(
            os.path.join(MODEL_REGISTRY_PATH, tokenizer_name_or_path),
            clean_up_tokenization_spaces=True,
        )
        self.for_trainer = for_trainer
        self.max_length = max_length

    def __len__(self):
        if self.n_samples > 0:
            return (
                self.n_samples * len(self.pile_file_ids)
                if isinstance(self.pile_file_ids, list)
                else self.n_samples * 30
            )
        else:
            total_length = 0
            if self.pile_file_ids:
                for file_id in self.file_ids:
                    total_length += PileDataset.lengths[file_id]
            else:
                total_length = sum(PileDataset.lengths.values())
            return total_length

    def __getitem__(self, idx):
        sample = next(self.samples)

        inputs = self.tokenizer(
            sample["texts"],
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
        )

        if self.for_trainer:
            inputs["input_ids"] = inputs["input_ids"][0, :]
            labels = inputs["input_ids"].clone()
            inputs["labels"] = labels
        else:
            inputs["input_ids"] = inputs["input_ids"]
            labels = inputs["input_ids"].clone()
            inputs["labels"] = labels

        return inputs


def custom_collate_fn(batch):
    input_ids = [item["input_ids"].squeeze(0) for item in batch]
    attention_mask = [item["attention_mask"].squeeze(0) for item in batch]
    labels = [item["labels"].squeeze(0) for item in batch]

    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
    attention_mask = pad_sequence(
        attention_mask, batch_first=True, padding_value=[0][-1]
    )
    labels = pad_sequence(labels, batch_first=True, padding_value=-100)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


class PileDataModule(pl.LightningDataModule):
    def __init__(
        self,
        dataset_name: str,
        batch_size: int,
        max_length: int,
        tokenizer_name_or_path: str = "pythia_all/pythia-70m",
        pile_file_ids: Optional[List[str]] = None,
        n_samples: int = -1,
        n_test_samples: int = 100,
        num_workers=12,
    ):
        super().__init__()
        self.dataset_name = dataset_name
        self.batch_size = batch_size
        self.max_length = max_length
        self.tokenizer_name_or_path = tokenizer_name_or_path
        self.pile_file_ids = pile_file_ids
        self.n_samples = n_samples
        self.n_test_samples = n_test_samples
        self.num_workers = num_workers

    def setup(self, stage=None):
        self.train_dataset = PileDataset(
            max_length=self.max_length,
            tokenizer_name_or_path=self.tokenizer_name_or_path,
            pile_file_ids=self.pile_file_ids,
            n_samples=self.n_samples,
        )
        self.val_dataset = PileDataset(
            max_length=self.max_length,
            tokenizer_name_or_path=self.tokenizer_name_or_path,
            pile_file_ids=self.pile_file_ids,
            n_samples=self.n_test_samples,
        )
        self.predict_dataset = PileDataset(
            max_length=self.max_length,
            tokenizer_name_or_path=self.tokenizer_name_or_path,
            pile_file_ids=self.pile_file_ids,
            n_samples=self.n_test_samples,
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=custom_collate_fn,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=custom_collate_fn,
        )

    def predict_dataloader(self):
        return DataLoader(
            self.predict_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=custom_collate_fn,
        )


import json
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
from collections import defaultdict
import random
import numpy as np
from typing import Dict, List, Optional, Iterator, Callable, Union, Tuple
import pandas as pd
import ast
import matplotlib.pyplot as plt
import torch
import pytorch_lightning as pl
from transformers import AutoTokenizer, DataCollatorWithPadding
from datasets import load_dataset

COUNTRIES = [
    "Nigeria",
    "Egypt",
    "India (Current national sample)",
    "China",
    "Japan",
    "Germany",
    "France",
    "Spain",
    "United States",
    "Canada",
    "Brazil",
    "Argentina",
    "Australia",
    "New Zealand",
]


def load_and_prepare_data(
    dataset_name: str, split: str, group_filter: List[str], cache_dir: str = None
):
    try:
        # Try to load the dataset from the Hugging Face Hub
        dataset = load_dataset(dataset_name, cache_dir=cache_dir)["train"]
    except Exception as e:
        print(f"Error loading dataset from Hugging Face Hub: {e}")
        print("Loading dataset from local storage...")
        dataset = load_dataset(
            "/home/uceesr4/Group-robust-preference-optimization/llm_global_opinions"
        )["train"]
    df = pd.DataFrame(dataset)
    df["qkey"] = df.index

    new_selections = []
    new_rows = []
    new_options = []
    for i in range(len(df)):
        if not df.loc[i, "question"] or not df.loc[i, "options"]:
            continue
        selections_str = "{" + df.loc[i, "selections"].split("{")[1].split("}")[0] + "}"
        selections_dict = ast.literal_eval(selections_str)
        if group_filter and not any(
            country in selections_dict for country in group_filter
        ):
            continue

        new_selections.append(selections_dict)
        new_rows.append(df.loc[i])
        parsed_options = ast.literal_eval(df.loc[i, "options"])
        new_options.append([str(opt) for opt in parsed_options])

    return pd.DataFrame(new_rows), new_selections, new_options


def process_data_frame(df, selections, group_filter, options):
    df["selections"] = selections
    df["options"] = options
    df["selections"] = df["selections"].apply(
        lambda x: [(k, v) for k, v in x.items()]
    )  # create country - selections tuples
    df = df.explode("selections", ignore_index=True)
    df[["group", "prob_y"]] = pd.DataFrame(df["selections"].tolist(), index=df.index)
    df = df[
        df["prob_y"].apply(lambda x: x is not None and len(x) > 0 and np.sum(x) > 0)
    ]
    return df[df["group"].isin(group_filter)]


def plot_questions_by_country(df, title_suffix=""):
    count = df["group"].value_counts()
    count.plot(kind="bar", figsize=(10, 6), color="skyblue")
    plt.title(f"Number of Questions by Country {title_suffix}")
    plt.xlabel("Country")
    plt.ylabel("Number of Questions")
    plt.xticks(rotation=45, ha="right")
    plt.savefig(f"questions_by_country{title_suffix}.png")
    plt.close()


def create_goqa_data(df, multi_pair=False, n_pairs=4):

    grouped = df.groupby("group")
    data = defaultdict(lambda: defaultdict(list))
    for group_name, group_data in grouped:
        for qkey, sub_group in group_data.groupby("qkey"):
            question = sub_group["question"].values[0]
            options = sub_group["options"].values[0]
            # prompt = f"Opinion of people in {group_name} on: {question}\nPlease select the best response:"
            prompt = f"{question}\nPlease select the best response:"
            letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"[: len(options)]
            for opt, letter in zip(options, letters):
                prompt += f"\n{letter}. {opt}"
            responses = [letter for letter in letters[: len(options)]]
            prob_y = torch.tensor(
                np.stack(sub_group["prob_y"].values), dtype=torch.float
            ).squeeze()
            ranks = torch.argsort(prob_y)
            pairs = [(ranks[i], ranks[j]) for i in range(len(ranks)) for j in range(i)]
            correct_response_index = ranks[-1]
            correct_response = responses[ranks[-1]]
            data[prompt]["prob_y"] = prob_y
            data[prompt]["sft_target"] = correct_response
            data[prompt]["responses"] = responses
            if multi_pair:
                data[prompt]["pairs"] = random.sample(pairs, min(n_pairs, len(pairs)))
            else:
                wrong_indices = [
                    i for i in range(len(options)) if i != correct_response_index
                ]
                if wrong_indices:
                    wrong_response_index = random.choice(wrong_indices)
                    data[prompt]["pairs"] = [
                        (correct_response_index, wrong_response_index)
                    ]
    # data = defaultdict(lambda: defaultdict(list))
    # for group_name, group_data in grouped:
    #     for qkey, sub_group in group_data.groupby("qkey"):
    #         question = sub_group["question"].values[0]
    #         options = sub_group["options"].values[0]
    #         prompt = f"Opinion of people in {group_name} on: {question}\nPlease select the best response:"
    #         letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"[: len(options)]
    #         for opt, letter in zip(options, letters):
    #             prompt += f"\n{letter}. {opt}"
    #         responses = [letter for letter in letters[: len(options)]]
    #         prob_y = torch.tensor(
    #             np.stack(sub_group["prob_y"].values), dtype=torch.float
    #         ).squeeze()
    #         ranks = torch.argsort(prob_y)
    #         pairs = [(ranks[i], ranks[j]) for i in range(len(ranks)) for j in range(i)]
    #         correct_response_index = ranks[-1]
    #         correct_response = responses[ranks[-1]]
    #         data[prompt]["sft_target"] = correct_response
    #         data[prompt]["responses"] = responses
    #         if multi_pair:
    #             data[prompt]["pairs"] = random.sample(pairs, min(n_pairs, len(pairs)))
    #         else:
    #             wrong_indices = [
    #                 i for i in range(len(options)) if i != correct_response_index
    #             ]
    #             if wrong_indices:
    #                 wrong_response_index = random.choice(wrong_indices)
    #                 data[prompt]["pairs"] = [
    #                     (correct_response_index, wrong_response_index)
    #                 ]
    return data


def get_goqa(
    split: str,
    train_frac: float = 0.8,
    group_id: int = None,
    multi_pair: bool = False,
    n_pairs: int = 4,
    silent: bool = False,
    cache_dir: str = None,
):
    if group_id is None:
        group_filter = COUNTRIES
    else:
        group_filter = [COUNTRIES[group_id]]
    df, selections, options = load_and_prepare_data(
        "Anthropic/llm_global_opinions", split, group_filter, cache_dir
    )
    df = process_data_frame(df, selections, group_filter, options)
    plot_questions_by_country(
        df, title_suffix=f" {split} with groups {' '.join(group_filter)}"
    )
    return df


def get_collate_fn(
    tokenizer,
) -> Callable[[List[Dict]], Dict[str, Union[List, torch.Tensor]]]:
    """Returns a collate function for the given tokenizer.

    The collate function takes a list of examples (dicts, where values are lists of
        ints [tokens] or strings [the original texts]) and returns a batch of examples,
        PyTorch tensors padded to the maximum length. Strings are passed through."""

    def collate_fn(batch):
        # first, pad everything to the same length
        padded_batch = {}
        for k in batch[0].keys():
            if (
                k.endswith("_input_ids")
                or k.endswith("_attention_mask")
                or k.endswith("_labels")
            ):
                if (
                    "prompt" in k
                ):  # adapted from https://stackoverflow.com/questions/73256206
                    to_pad = [torch.LongTensor(ex[k][::-1]) for ex in batch]
                else:
                    to_pad = [torch.LongTensor(ex[k]) for ex in batch]
                if k.endswith("_input_ids"):
                    padding_value = tokenizer.pad_token_id
                elif k.endswith("_labels"):
                    padding_value = -100
                elif k.endswith("_attention_mask"):
                    padding_value = 0
                else:
                    raise ValueError(f"Unexpected key in batch '{k}'")

                padded_batch[k] = pad_sequence(
                    to_pad, batch_first=True, padding_value=padding_value
                )
                if (
                    "prompt" in k
                ):  # for the prompt, flip back so padding is on left side
                    padded_batch[k] = padded_batch[k].flip(dims=[1])
            else:
                padded_batch[k] = [ex[k] for ex in batch]

        return padded_batch

    return collate_fn


class GlobalOpinionQADataset(Dataset):
    def __init__(self, data, tokenizer, max_length: int = 128):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        prompt = list(self.data.keys())[idx]
        item = self.data[prompt]
        tokens = self.tokenizer(
            prompt,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            "input_ids": tokens["input_ids"].squeeze(),
            "attention_mask": tokens["attention_mask"].squeeze(),
            "correct_label": item["sft_target"],
            "labels": item["prob_y"],
            "responses": item["responses"],
            "pairs": item["pairs"],
        }


class GlobalOpinionQADataModule(pl.LightningDataModule):
    def __init__(
        self,
        tokenizer_name: str,
        batch_size: int = 8,
        max_length: int = 128,
        countries: list = None,
        train_frac: float = 0.8,
    ):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.batch_size = batch_size
        self.max_length = max_length
        self.countries = countries
        self.train_frac = train_frac
        self.data_collator = get_collate_fn(tokenizer=self.tokenizer)

    def setup(self, stage=None):
        group_id = (
            None if self.countries is None else COUNTRIES.index(self.countries[0])
        )
        df = get_goqa(split="train", group_id=group_id)
        df_train = df.sample(frac=self.train_frac, random_state=42)
        df_test = df.drop(df_train.index)
        df_truetest = df_test.sample(frac=0.5, random_state=42)
        df_valtest = df_test.drop(df_truetest.index)

        train_data = create_goqa_data(df=df_train)
        val_data = create_goqa_data(df=df_valtest)
        test_data = create_goqa_data(df=df_truetest)

        self.train_dataset = GlobalOpinionQADataset(
            data=train_data, tokenizer=self.tokenizer, max_length=self.max_length
        )
        self.val_dataset = GlobalOpinionQADataset(
            data=val_data, tokenizer=self.tokenizer, max_length=self.max_length
        )
        self.test_dataset = GlobalOpinionQADataset(
            data=test_data, tokenizer=self.tokenizer, max_length=self.max_length
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=self.data_collator,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            collate_fn=self.data_collator,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            collate_fn=self.data_collator,
        )


# # import os
# # from torch.utils.data import Dataset, DataLoader
# # from transformers import AutoTokenizer, DataCollatorForLanguageModeling
# # from datasets import load_dataset
# # import pytorch_lightning as pl


# class GlobalOpinionQADataset(Dataset):

#     COUNTRIES = [
#         "Nigeria",
#         "Egypt",
#         "India (Current national sample)",
#         "China",
#         "Japan",
#         "Germany",
#         "France",
#         "Spain",
#         "United States",
#         "Canada",
#         "Brazil",
#         "Argentina",
#         "Australia",
#         "New Zealand",
#     ]

#     def __init__(self, tokenizer: str, max_length: int = 128, countries: list = None):
#         self.dataset = load_dataset("Anthropic/llm_global_opinions", split="train")
#         self.tokenizer = tokenizer
#         self.max_length = max_length
#         self.countries = (
#             countries if countries else list(self.dataset[0]["selections"].keys())
#         )

#     def __len__(self):
#         return len(self.dataset)

#     def __getitem__(self, idx):
#         text = self.dataset[idx]["question"]
#         labels = {
#             country: self.dataset[idx]["selections"][country]
#             for country in self.countries
#         }

#         tokens = self.tokenizer(
#             text,
#             padding="max_length",
#             truncation=True,
#             max_length=self.max_length,
#             return_tensors="pt",
#         )
#         return {
#             "input_ids": tokens["input_ids"].squeeze(),
#             "attention_mask": tokens["attention_mask"].squeeze(),
#         }


# class GlobalOpinionQADataModule(pl.LightningDataModule):
#     def __init__(self, tokenizer_name: str, batch_size: int = 8, max_length: int = 128):
#         super().__init__()
#         self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
#         self.tokenizer.pad_token = self.tokenizer.eos_token
#         self.batch_size = batch_size
#         self.max_length = max_length
#         self.data_collator = DataCollatorForLanguageModeling(
#             tokenizer=self.tokenizer,
#             mlm=False,
#         )

#     def setup(self, stage=None):
#         self.dataset = GlobalOpinionQADataset(
#             tokenizer=self.tokenizer,
#             max_length=self.max_length,
#         )

#     def train_dataloader(self):
#         return DataLoader(
#             self.dataset,
#             batch_size=self.batch_size,
#             shuffle=True,
#             collate_fn=self.data_collator,
#         )

#     def val_dataloader(self):
#         return DataLoader(
#             self.dataset,
#             batch_size=self.batch_size,
#             collate_fn=self.data_collator,
#         )

#     def test_dataloader(self):
#         return DataLoader(
#             self.dataset,
#             batch_size=self.batch_size,
#             collate_fn=self.data_collator,
#         )
