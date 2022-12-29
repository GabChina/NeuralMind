import itertools
import pathlib

import numpy as np
from datasets import load_metric
from transformers import AutoTokenizer
import wandb

from pipeline import test_with_checkpoints
from src.utils import json2dict

if __name__ == "__main__":
    wandb.login()
    wandb_config = {
            "project": "SWNM",
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

    test_hyperparameters = {
        #"balance": [True, False],
        "balancing_range": [round(i,2) for i in np.arange(0.3, 0.4, 0.05)],
        "balancing_upper_limit": [round(i,2) for i in np.arange(0.65, 0.9, 0.1)],
        "test_size": [round(i,2) for i in np.arange(0.1, 0.4, 0.05)],
        "stride": [0, 128, 256],
    }

    params_used = [k for k in test_hyperparameters]
    params_list = list(itertools.product(*(test_hyperparameters.values())))
    params_list = [{k:v for k,v in zip(params_used, p)} for p in params_list]

    test_with_checkpoints(params_list=params_list,
                            output_name='SWNM-dataset',
                            dataset=dataset,
                            label_names=label_names,
                            metric=metric,
                            entities_names=entities_names,
                            output_dir=current_dir+"checkpoints/",
                            step=0.05,
                            use_wandb=True,
                            wandb_config=wandb_config,)
