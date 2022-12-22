import os
import time

from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer

from src.balancing import balance_datasets
from src.NM_Trainer import NM_Trainer
from src.tokenizing import tokenize_dataset
from src.utils import dict2json


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
                        metric=metric,
                        tokenizer=tokenizer)
    trainer.train()

    return trainer.return_metrics()


def test_with_checkpoints(params_list,
                        output_name,
                        dataset: dict,
                        label_names,
                        metric,
                        entities_names=None,
                        output_dir='checkpoints/',
                        step=0.1):
    step = round(len(params_list)*step)
    checkpoints = [x for x in range(step, len(params_list)-step, step)]
    test_results = {}
    run = 0
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for parameters in params_list:
        result = run_test(
            dataset=dataset,
            label_names=label_names,
            metric=metric,
            entities_names=entities_names,
            **parameters
        )
        run += 1
        test_results[f'run{run}'] = {
            'parameters': parameters,
            'result': result,
        }
        if run in checkpoints:
            timestr = time.strftime("%Y%m%d-%H%M%S")
            fname = f"{output_dir}{output_name}_{timestr}_run{run}.json"
            dict2json(test_results, fname, sort_keys=False, indent=2)

    timestr = time.strftime("%Y%m%d-%H%M%S")
    dict2json(test_results, f"{output_dir}{output_name}_{timestr}_final.json',
                sort_keys=False, indent=2)

    return test_results
