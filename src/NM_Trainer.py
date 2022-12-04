import numpy as np
from transformers import AutoTokenizer, TrainingArguments, Trainer
from transformers import AutoModelForTokenClassification, DataCollatorForTokenClassification
from utils import json2dict

class NM_Trainer():
    def __init__(self, data_path: str, metric, label_names: str,
                entities_names: str = None, tokenizer = None) -> None:
        self.dataset = [doc for doc in json2dict(data_path)]
        self.metric = metric
        self.label_names = label_names
        self.entities_names = entities_names
        self.tokenizer = tokenizer
        if tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained('neuralmind/bert-base-portuguese-cased',
                                                            do_lower_case=False)

    def return_metrics(self, trainer, teste) -> dict:
        predictions, labels, _ = trainer.predict(teste)
        predictions = np.argmax(predictions, axis=2)

        # Remove ignored index (special tokens)
        true_predictions = [
            [self.label_names[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [self.label_names[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]

        return self.metric.compute(predictions=true_predictions, references=true_labels)

    def get_trainer(self, treino, teste, tokenizer):
        def model_init():
            return AutoModelForTokenClassification.from_pretrained("neuralmind/bert-base-portuguese-cased", num_labels=7)

        def compute_metrics(p):
            predictions, labels = p
            predictions = np.argmax(predictions, axis=2)

            # Remove ignored index (special tokens)
            true_predictions = [
                [self.label_names[p] for (p, l) in zip(prediction, label) if l != -100]
                for prediction, label in zip(predictions, labels)
            ]
            true_labels = [
                [self.label_names[l] for (p, l) in zip(prediction, label) if l != -100]
                for prediction, label in zip(predictions, labels)
            ]

            results = self.metric.compute(predictions=true_predictions, references=true_labels)

            return {
                    "precision": results["overall_precision"],
                    "recall": results["overall_recall"],
                    "f1": results["overall_f1"],
                    #"accuracy": results["overall_accuracy"],
                    }

        data_collator = DataCollatorForTokenClassification(tokenizer)
        hyperparameters={
            'learning_rate': 4.076831342095183e-05,
            'num_train_epochs': 3,
            'per_device_train_batch_size': 4
        }
        batch_size = hyperparameters['per_device_train_batch_size']
        logging_steps = len(treino) // batch_size
        epochs = hyperparameters['num_train_epochs']
        training_args = TrainingArguments(
            output_dir = "results",
            num_train_epochs = epochs,
            per_device_train_batch_size = batch_size,
            per_device_eval_batch_size = batch_size,
            evaluation_strategy = "epoch",
            metric_for_best_model = "f1",
            disable_tqdm = False,
            logging_steps = logging_steps,
            gradient_accumulation_steps = 2,
            eval_accumulation_steps = 2,
            learning_rate = hyperparameters['learning_rate'],
        )
        trainer = Trainer(
            model_init=model_init,
            args=training_args,
            train_dataset=treino,
            eval_dataset=teste,
            data_collator=data_collator,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics
        )

        return trainer
