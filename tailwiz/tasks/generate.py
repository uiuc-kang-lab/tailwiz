import evaluate
import os
os.environ['TRANSFORMERS_CACHE'] = 'cache'  # Must be before transformers import.
import pandas as pd
import torch
import transformers
from sklearn.model_selection import train_test_split

from .task import Task

class GenerateTask(Task):
    def __init__(self, train, val, test):
        (
            self.train_dataset,
            self.val_dataset,
            self.val_references,
            self.test_dataset
        ) = self._load_data(train, val, test)
        self.model = self._load_model()

    def _load_data(self, train, val, test):
        self.tokenizer = transformers.T5Tokenizer.from_pretrained('google/flan-t5-base', cache_dir='cache/flan-t5')
        
        train_tokens = None
        if train is not None:
            train_tokens = self.tokenizer(train.prompt.tolist(), return_tensors='pt', padding=True)
            train_labels = self.tokenizer(train.label.tolist(), return_tensors='pt', padding=True)
            train_tokens['labels'] = train_labels['input_ids']
        
        val_tokens = None
        if val is not None:
            val_tokens = self.tokenizer(val.prompt.tolist(), return_tensors='pt', padding=True)
            val_labels = self.tokenizer(val.label.tolist(), return_tensors='pt', padding=True)
            val_tokens['labels'] = val_labels['input_ids']

        test_tokens = self.tokenizer(test.prompt.tolist(), return_tensors='pt', padding=True)

        return train_tokens, val_tokens, (val.label.tolist() if val is not None else None), test_tokens

    def _load_model(self):
        return transformers.T5ForConditionalGeneration.from_pretrained('google/flan-t5-base', cache_dir='cache/flan-t5')

    def train(self):
        class FT5Dataset(torch.utils.data.Dataset):
            def __init__(self, data):
                self.data = data
            
            def __len__(self):
                return len(self.data['input_ids'])
            
            def __getitem__(self, index):
                return {k: self.data[k][index] for k in self.data.keys()}

        args = transformers.Seq2SeqTrainingArguments(
            'cache/flan-t5',
            num_train_epochs=10,
            evaluation_strategy='epoch',
            save_strategy='epoch',
            load_best_model_at_end=True,
            predict_with_generate=True,
            metric_for_best_model='eval_loss',
            logging_steps=1,
        )
        trainer = transformers.Seq2SeqTrainer(
            model=self.model,
            args=args,
            train_dataset=FT5Dataset(self.train_dataset),
            eval_dataset=FT5Dataset(self.val_dataset) if self.val_dataset is not None else FT5Dataset(self.train_dataset),  # Requires eval dataset.
        )
        trainer.train()

    def evaluate(self):
        # Remove 'labels' key from validation dataset for prediction.
        dummy_val = self.val_dataset.copy()
        del dummy_val['labels']
        outputs = self.model.to(dummy_val['input_ids'].device).generate(**dummy_val)
        predictions = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

        metrics_funcs = [
            evaluate.load('exact_match'),
            evaluate.load('rouge')
        ]
        metrics = {}
        for metric in metrics_funcs:
            metrics.update(metric.compute(predictions=predictions, references=self.val_references))
        for k in metrics.keys():
            metrics[k] = metrics[k].item()
        return metrics

    def predict(self):
        outputs = self.model.to(self.test_dataset['input_ids'].device).generate(**self.test_dataset)
        return self.tokenizer.batch_decode(outputs, skip_special_tokens=True)


def generate(to_generate, labeled_examples=None, output_metrics=False):
    assert isinstance(to_generate, pd.DataFrame), 'Make sure you are passing in pandas DataFrames.'
    assert 'prompt' in to_generate.columns, \
        'Make sure the prompt column in your pandas DataFrame is named "prompt".'
    if labeled_examples is not None:
        assert isinstance(labeled_examples, pd.DataFrame), 'Make sure you are passing in pandas DataFrames.'
        assert 'prompt' in labeled_examples.columns and 'label' in labeled_examples.columns, \
            'Make sure the prompt column in your pandas DataFrame is named "prompt" and the label column is named "label".'
    if output_metrics:
        assert labeled_examples is not None, 'In order to output an estimate of performance with output_metrics, labeled_examples must be provided.'

    if labeled_examples is None:
        generate_task_out = GenerateTask(None, None, to_generate)
        pred_results = generate_task_out.predict()
    else:
        assert len(labeled_examples) >= 2, 'At least 2 rows of prelabeled data must be given.'
        train, val = train_test_split(labeled_examples, test_size=0.2)
        generate_task_out = GenerateTask(train, val, to_generate)
        generate_task_out.train()
        pred_results = generate_task_out.predict()
    
    results = to_generate.copy()
    results['label_from_tailwiz'] = pred_results

    if output_metrics:
        metrics = generate_task_out.evaluate()

    return (results, metrics) if output_metrics else results
