import itertools
import pathlib

import numpy as np
from datasets import load_metric
from ray.tune.schedulers import PopulationBasedTraining
from ray.tune.logger import DEFAULT_LOGGERS
from ray import tune
from ray.tune.integration.wandb import WandbLoggerCallback
from transformers import AutoTokenizer
import wandb

from pipeline import run_pbt
from src.utils import json2dict

if __name__ == "__main__":
    wandb.login()
    wandb_config = {
        "project": "PBT_Optimization_Project",
        "entity": "chinagab",
        "api_key": "7d7deda5ab99137996e34e47dc688b1d6b4d179c",
        "log_config": True
    }

    current_dir = str(pathlib.Path(__file__).parent.resolve()) + "/"
    data_path = current_dir + "NM_dataset.json"
    dataset = [doc for doc in json2dict(data_path)]
    tokenizer = AutoTokenizer.from_pretrained('neuralmind/bert-base-portuguese-cased', do_lower_case=False)
    metric = load_metric("seqeval")
    entities_names = ['COMECO RECORTE', 'CABECALHO', 'SUBCABECALHO']
    label_names={
        0: 'O',
        1: 'B-CABECALHO',
        2: 'I-CABECALHO',
        3: 'B-SUBCABECALHO',
        4: 'I-SUBCABECALHO',
        5: 'B-COMECO_RECORTE',
        6: 'I-COMECO_RECORTE',
    }

    def my_objective(metrics):
        return metrics["eval_f1"]

    pbt_scheduler = PopulationBasedTraining(
        time_attr='training_iteration',
        metric=my_objective,
        mode='max',
        perturbation_interval=600.0,
        hyperparam_mutations={
            "learning_rate": tune.loguniform(6e-6, 1e-3),
            "num_train_epochs": tune.choice(range(5, 15)),
            # "seed": tune.choice(range(1, 41)),
            "per_device_train_batch_size": tune.choice([4, 8, 16]),
        })

    run_pbt(pbt_scheduler=pbt_scheduler,
            dataset=dataset,
            label_names=label_names,
            metric=metric,
            tokenizer=tokenizer,
            entities_names=entities_names,
            use_wandb = True,
            wandb_config=wandb_config,)
