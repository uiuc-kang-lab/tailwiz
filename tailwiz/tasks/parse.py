import os
os.environ['TRANSFORMERS_CACHE'] = 'cache'  # Must be before transformers import.
import pandas as pd
import torch
import transformers
import evaluate
from sklearn.model_selection import train_test_split
import transformers

from .task import Task

class ParsingTask(Task):
    def __init__(self, train, val, test):
        self.tokenizer = transformers.BertTokenizer.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
        (
            self.train_dataset,
            self.val_dataset,
            self.test_dataset
        ) = self._load_data(train, val, test)
        self.model = self._load_model()

    def _load_data(self, train, val, test):
        # Tokenize.
        train_tokens = None if train is None else self.tokenizer(train.prompt.tolist(), train.context.tolist(), train.label.tolist(), padding=True, return_tensors='pt')
        
        val_tokens = None if val is None else self.tokenizer(val.prompt.tolist(), val.context.tolist(), val.label.tolist(), padding=True, return_tensors='pt')

        test_tokens = self.tokenizer(test.prompt.tolist(), test.context.tolist(), padding=True, return_tensors='pt')

        # Helper function that finds the start and end positions of the label within the context.
        # By default sets the start and end positions to the first occurrence of the label.
        def get_target_start_end_pos(tokens, contexts, labels):
            target_start_pos = []
            target_end_pos = []
            for i, (text_tokenized, label_tokenized, context, label) in enumerate(zip(tokens['input_ids'], tokens['labels'], contexts, labels)):
                if label_tokenized[0] == 101 and label_tokenized[1] == 102:
                    target_start_pos.append(0)
                    target_end_pos.append(0)
                else:
                    label_tokenized_nopad = label_tokenized[label_tokenized.nonzero()].squeeze()
                    label_tokenized_nospectoks = label_tokenized_nopad[1:-1]

                    for j in range(len(text_tokenized) - len(label_tokenized_nospectoks)):  # Search from start for label
                        if (text_tokenized[j:j+len(label_tokenized_nospectoks)] == label_tokenized_nospectoks).all():
                            target_start_pos.append(j)
                            target_end_pos.append(j+len(label_tokenized_nospectoks))
                            break
                if len(target_start_pos) != i + 1:
                    raise ValueError(f'''We could not find the label:
                        
                        {label}

in the context:
                        
                        {context}

Either: (A) only include labels where the label is extracted exactly from the context as whole words, or (B) use the generate()
function, which does not require the labels do be found in the context.''')
            target_start_pos = torch.tensor(target_start_pos)
            target_end_pos = torch.tensor(target_end_pos)
            return target_start_pos, target_end_pos

        if train_tokens is not None:
            train_target_start_pos, train_target_end_pos = get_target_start_end_pos(train_tokens, train.context.tolist(), train.label.tolist())
            train_tokens['start_positions'] = train_target_start_pos
            train_tokens['end_positions'] =  train_target_end_pos
            train_tokens = {k:v for k, v in train_tokens.items() if k != 'labels'}

        if val_tokens is not None:
            val_target_start_pos, val_target_end_pos = get_target_start_end_pos(val_tokens, val.context.tolist(), val.label.tolist())
            val_tokens['start_positions'] = val_target_start_pos
            val_tokens['end_positions'] =  val_target_end_pos
            val_tokens = {k:v for k, v in val_tokens.items() if k != 'labels'}

        return train_tokens, val_tokens, test_tokens

    def _load_model(self):
        return transformers.BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')

    def _decode_pos2strs(self, start_pos, end_pos, input_ids):
        strs = []
        for i, (j, k) in enumerate(zip(start_pos, end_pos)):
            pred_tokens = input_ids[i, j:k+1]
            strs.append(self.tokenizer.decode(pred_tokens, skip_special_tokens=True))
        return strs

    def train(self):
        class BLUWWMFSDataset(torch.utils.data.Dataset):
            def __init__(self, data):
                self.data = data
            
            def __len__(self):
                return len(self.data['input_ids'])
            
            def __getitem__(self, index):
                return {k: self.data[k][index] for k in self.data.keys()}

        args = transformers.TrainingArguments(
            'cache/bert-qa',
            num_train_epochs=10,
            evaluation_strategy='epoch',
            save_strategy='epoch',
            load_best_model_at_end=True,
            metric_for_best_model='eval_loss',
            logging_steps=1,
        )
        trainer = transformers.Trainer(
            model=self.model,
            args=args,
            train_dataset=BLUWWMFSDataset(self.train_dataset),
            eval_dataset=BLUWWMFSDataset(self.val_dataset)
        )
        trainer.train()

    def evaluate(self):
        # Remove 'labels' key from validation dataset for prediction.
        outputs = self.model.to(self.val_dataset['input_ids'].device)(**self.val_dataset)

        predictions = self._decode_pos2strs(outputs['start_logits'].argmax(1), outputs['end_logits'].argmax(1), self.val_dataset['input_ids'])
        references = self._decode_pos2strs(self.val_dataset['start_positions'], self.val_dataset['end_positions'], self.val_dataset['input_ids'])

        metrics_funcs = [
            evaluate.load('exact_match'),
            evaluate.load('rouge')
        ]
        metrics = {}
        for metric in metrics_funcs:
            metrics.update(metric.compute(predictions=predictions, references=references))
        for k in metrics.keys():
            metrics[k] = metrics[k].item()
        return metrics

    def predict(self):
        outputs = self.model.to(self.test_dataset['input_ids'].device)(**self.test_dataset)
        return self._decode_pos2strs(outputs['start_logits'].argmax(1), outputs['end_logits'].argmax(1), self.test_dataset['input_ids'])


def parse(to_parse, labeled_examples=None, output_metrics=False):
    assert isinstance(to_parse, pd.DataFrame), 'Make sure you are passing in pandas DataFrames.'
    assert 'prompt' in to_parse.columns and 'context' in to_parse.columns, \
        'Make sure the prompt column in your pandas DataFrame is named "prompt" and the context column is named "context".'

    if labeled_examples is not None:
        assert isinstance(labeled_examples, pd.DataFrame), 'Make sure you are passing in pandas DataFrames.'
        assert 'prompt' in labeled_examples.columns and 'context' in labeled_examples.columns and 'label' in labeled_examples.columns, \
            'Make sure the prompt column in your pandas DataFrame is named "prompt", the context column is named "context", and the label column is named "label".'
    if output_metrics:
        assert labeled_examples is not None, 'In order to output an estimate of performance with output_metrics, labeled_examples must be provided.'

    if labeled_examples is None:
        parse_task_out = ParsingTask(None, None, to_parse)
        pred_results = parse_task_out.predict()
    else:
        assert len(labeled_examples) >= 2, 'At least 2 rows of prelabeled data must be given.'
        train, val = train_test_split(labeled_examples, test_size=0.2)
        parse_task_out = ParsingTask(train, val, to_parse)
        parse_task_out.train()
        pred_results = parse_task_out.predict()
    
    results = to_parse.copy()
    results['label_from_tailwiz'] = pred_results

    if output_metrics:
        metrics = parse_task_out.evaluate()

    return (results, metrics) if output_metrics else results
