import os
import json
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer
from datasets import Dataset
from balancing import balance_datasets
from tokenizing import tokenize_dataset
from NM_Trainer import NM_Trainer

#TODO: do we really need this class? Should it be here or at another file?
class NM_Dataset:
    def __init__(self, dataset: str=None) -> None:
        if dataset is not None:
            self.load_dataset(dataset)

    def load_dataset(self, path: str):
        """Loads a .hf file into a Dataset variable."""
        self.dataset = Dataset.load_from_disk(path)

    def save_dataset(self, path: str):
        """Saves the dataset as a .hf file."""
        self.dataset.save_to_disk(path)

    def load_dataset_from_json(self, path: str):
        """Loads a .json file into a Dataset variable.

        Made for compatibility with older code.
        """
        with open(path, 'r', encoding='utf8') as jfile:
            data = json.load(jfile)
        corpus = [doc for doc in data]
        self.dataset = Dataset.from_list(corpus)


def run_test(
        dataset: dict,
        label_names,
        metric,
        balance=True,
        stride=0,
        tokenizer=None,
        test_size=0.2,
        random_state=42,
        balancing_upper_limit=0.75,
        balancing_range=0.20,
        entities_names=None,
        ):
    #dataset
    treino, teste = train_test_split(dataset,
                                    test_size=test_size,
                                    random_state=random_state)

    #balanceamento
    if balance:
        balance_datasets(treino, teste,
                        upper_limit=balancing_upper_limit,
                        balancing_range=balancing_range,
                        names_list=entities_names)

    #tokenização
    if not tokenizer:
        tokenizer = AutoTokenizer.from_pretrained('neuralmind/bert-base-portuguese-cased',
                                                    do_lower_case=False)
    treino = tokenize_dataset(treino, tokenizer,
                                stride=stride)
    teste = tokenize_dataset(teste, tokenizer,
                                stride=stride)

    #treino
    trainer = NM_Trainer(treino, teste,
                        label_names=label_names,
                        metric=metric)
    trainer.train()

    return trainer.return_metrics()
