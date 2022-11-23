import os
import json
import numpy as np
import tensorflow as tf
import matplotlib
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, TrainingArguments, Trainer
from transformers import AutoModelForTokenClassification, DataCollatorForTokenClassification
from datasets import Dataset, load_dataset, load_metric


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

class NM_Model:
    class NM_Trainer:
        pass
    pass
