from torch.utils.data import Dataset
import os
import json
import numpy as np
import torch
from utils_prompt import build


def load_dataset_std(dataset):
    with open("data/" + dataset + "/train.json", "r") as rf:
        train_data = json.load(rf)
    with open("data/" + dataset + "/val.json", "r") as rf:
        val_data = json.load(rf)
    with open("data/" + dataset + "/test.json", "r") as rf:
        test_data = json.load(rf)
    return train_data, val_data, test_data


class DatasetStd(Dataset):
    """
    Creating a custom dataset for reading the dataset and
    loading it into the dataloader to pass it to the
    neural network for finetuning the model

    """

    def __init__(self, data, tokenizer, source_len, target_len, args):
        self.tokenizer = tokenizer
        self.data = data
        self.source_len = source_len
        self.summ_len = target_len
        self.target_text = []
        self.source_text = []

        for i in data:
            prompt, target = build(i, args.prompt_format)
            self.target_text.append(target)
            self.source_text.append(prompt)

    def __len__(self):
        return len(self.target_text)

    def __getitem__(self, index):
        source_text = str(self.source_text[index])
        target_text = str(self.target_text[index])

        # cleaning data so as to ensure data is in string type
        source_text = " ".join(source_text.split())
        target_text = " ".join(target_text.split())

        source = self.tokenizer.batch_encode_plus(
            [source_text],
            max_length=self.source_len,
            pad_to_max_length=True,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        target = self.tokenizer.batch_encode_plus(
            [target_text],
            max_length=self.summ_len,
            pad_to_max_length=True,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        source_ids = source["input_ids"].squeeze()
        source_mask = source["attention_mask"].squeeze()
        target_ids = target["input_ids"].squeeze().tolist()

        return {
            "input_ids": source_ids,
            "attention_mask": source_mask,
            "labels": target_ids,
        }
