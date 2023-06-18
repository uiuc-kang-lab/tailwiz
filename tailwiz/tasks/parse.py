import os
os.environ['TRANSFORMERS_CACHE'] = 'cache'  # Must be before transformers import.
import pandas as pd
import torch
import tqdm
import transformers
import evaluate
from sklearn.model_selection import train_test_split

from .task import Task

class ParsingTask(Task):
    def __init__(self, train, val, test, **override_train_args):
        self.num_steps = 2 if train is None else 3
        self.tokenizer = transformers.BertTokenizer.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
        (
            self.train_dataset,
            self.val_dataset,
            self.test_dataset
        ) = self._load_data(train, val, test)
        self.model = self._load_model()
        self.override_train_args = override_train_args

    def _load_data(self, train, val, test):
        print(f'\n(1/{self.num_steps}) PROCESSING DATA...\n')

        text = test.prompt.tolist() + test.context.tolist()
        if train is not None:
            text += train.prompt.tolist() + train.context.tolist() + train.label.tolist()
        if val is not None:
            text += val.prompt.tolist() + val.context.tolist() + val.label.tolist()
        if len(max(text, key=len).split()) > 400:
            print('''
***

WARNING
At least one of your texts is long so it may have been truncated.
In general, this is okay. If you wish to use all your data, we
suggest you split your long texts into multiple lines. Try to remain
under 400 words per text.

***
''')
        # Tokenize.
        with tqdm.tqdm(total=5) as pbar:
            train_tokens = None if train is None else self.tokenizer(train.prompt.tolist(), train.context.tolist(), train.label.tolist(), padding=True, truncation=True, return_tensors='pt')
            pbar.update()
            
            val_tokens = None if val is None else self.tokenizer(val.prompt.tolist(), val.context.tolist(), val.label.tolist(), padding=True, truncation=True, return_tensors='pt')
            pbar.update()

            test_tokens = self.tokenizer(test.prompt.tolist(), test.context.tolist(), padding=True, truncation=True, return_tensors='pt')
            pbar.update()

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

because either the context was truncated due to its length and the
label appears later in the context, or because the label is not found
at all in the context.

To resolve, either: (A) only include labels where the label is extracted
exactly from the context as whole words, or (B) use the generate() function,
which does not require the labels do be found in the context.''')
                target_start_pos = torch.tensor(target_start_pos)
                target_end_pos = torch.tensor(target_end_pos)
                return target_start_pos, target_end_pos

            if train_tokens is not None:
                train_target_start_pos, train_target_end_pos = get_target_start_end_pos(train_tokens, train.context.tolist(), train.label.tolist())
                train_tokens['start_positions'] = train_target_start_pos
                train_tokens['end_positions'] =  train_target_end_pos
                train_tokens = {k:v for k, v in train_tokens.items() if k != 'labels'}
            pbar.update()

            if val_tokens is not None:
                val_target_start_pos, val_target_end_pos = get_target_start_end_pos(val_tokens, val.context.tolist(), val.label.tolist())
                val_tokens['start_positions'] = val_target_start_pos
                val_tokens['end_positions'] =  val_target_end_pos
                val_tokens = {k:v for k, v in val_tokens.items() if k != 'labels'}
            pbar.update()

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
        print(f'\n(2/{self.num_steps}) LEARNING...\n')
        class BLUWWMFSDataset(torch.utils.data.Dataset):
            def __init__(self, data):
                self.data = data
            
            def __len__(self):
                return len(self.data['input_ids'])
            
            def __getitem__(self, index):
                return {k: self.data[k][index] for k in self.data.keys()}

        args_dict = {
            'output_dir': 'cache/bert-qa',
            'num_train_epochs': 7,
            'evaluation_strategy': 'epoch',
            'save_strategy': 'no',
            'metric_for_best_model': 'eval_loss',
            'logging_steps': 1,
            'no_cuda': (not torch.cuda.is_available()),
            'use_mps_device': (torch.backends.mps.is_available() and not torch.cuda.is_available()),
        }
        args_dict.update(self.override_train_args)
        args = transformers.TrainingArguments(**args_dict)
        trainer = transformers.Trainer(
            model=self.model,
            args=args,
            train_dataset=BLUWWMFSDataset(self.train_dataset),
            eval_dataset=BLUWWMFSDataset(self.val_dataset)
        )
        trainer.train()

    def evaluate(self):
        print('\nGETTING METRICS...\n')
        # Remove 'labels' key from validation dataset for prediction.
        outputs = self.model.to(self.val_dataset['input_ids'].device)(**self.val_dataset)

        predictions = self._decode_pos2strs(outputs['start_logits'].argmax(1), outputs['end_logits'].argmax(1), self.val_dataset['input_ids'])
        references = self._decode_pos2strs(self.val_dataset['start_positions'], self.val_dataset['end_positions'], self.val_dataset['input_ids'])

        predictions = ['unlisted' if x == '' else x for x in predictions]
        references = ['unlisted' if x == '' else x for x in references]

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
        print(f'\n({self.num_steps}/{self.num_steps}) CREATING TAILWIZ LABELS...\n')
        self.model = self.model.to(self.test_dataset['input_ids'].device)
        out_predictions = []
        for i in tqdm.tqdm(range(self.test_dataset['input_ids'].shape[0])):
            outputs = self.model(
                input_ids=self.test_dataset['input_ids'][i].unsqueeze(0),
                token_type_ids=self.test_dataset['token_type_ids'][i].unsqueeze(0),
                attention_mask=self.test_dataset['attention_mask'][i].unsqueeze(0))
            out_predictions.extend(self._decode_pos2strs(outputs['start_logits'].argmax(1), outputs['end_logits'].argmax(1), self.test_dataset['input_ids'][i].unsqueeze(0)))
        return out_predictions
        # return self._decode_pos2strs(outputs['start_logits'].argmax(1), outputs['end_logits'].argmax(1), self.test_dataset['input_ids'])


def parse(to_parse, labeled_examples=None, output_metrics=False, **override_train_args):
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
        parse_task_out = ParsingTask(None, None, to_parse, **override_train_args)
        pred_results = parse_task_out.predict()
    else:
        assert len(labeled_examples) >= 2, 'At least 2 rows of prelabeled data must be given.'
        train, val = train_test_split(labeled_examples, test_size=0.2)
        parse_task_out = ParsingTask(train, val, to_parse, **override_train_args)
        parse_task_out.train()
        pred_results = parse_task_out.predict()
    
    results = to_parse.copy()
    results['tailwiz_label'] = pred_results

    if output_metrics:
        metrics = parse_task_out.evaluate()

    print('\nDONE')
    return (results, metrics) if output_metrics else results
