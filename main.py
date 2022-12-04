from pipeline import run_test
from utils import json2dict
from transformers import AutoTokenizer
from datasets import load_metric

if __name__ == "__main__":
    data_path = "NM_dataset.json"
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
        'balance': True,
        'stride': 0,
        'tokenizer': tokenizer,
        'test_size': 0.2,
        'random_state': 42,
        'balancing_upper_limit': 0.75,
        'balancing_range': 0.20,
    }

    test_results = run_test(
        dataset=dataset,
        label_names=label_names,
        metric=metric,
        entities_names=entities_names,
        **test_hyperparameters
    )

    #TODO: Add some way to save the results in a folder
