import numpy as np
import os
os.environ['TRANSFORMERS_CACHE'] = 'cache'  # Must be before transformers import.
import pandas as pd
import sklearn
from sklearn import cluster, linear_model, metrics, multiclass
from sklearn.model_selection import train_test_split
import torch
import tqdm
import transformers

from . import utils
from .task import Task


class ClassificationTask(Task):
    def __init__(self, train, val, test):
        print(f'\n(1/3) PROCESSING DATA...\n')
        (
            self.train_embeds,
            self.train_labels,
            self.val_embeds,
            self.val_labels,
            self.test_embeds
        ) = self._load_data(train, val, test)

        self.model = self._load_model()
    
    def _load_data(self, train, val, test):
        text = []
        if train is not None:
            text += train.text.tolist()
        if val is not None:
            text += val.text.tolist()
        text += test.text.tolist()  # Must embed togeter to match sequence length.

        if torch.cuda.is_available():
            device = 'cuda'
        else:
            device = 'cpu'

        tokenizer = transformers.BertTokenizer.from_pretrained('bert-base-uncased')
        embed_model = transformers.BertModel.from_pretrained('bert-base-uncased', output_hidden_states=True).to(device)
        embed_model.eval()

        bert_max_length = 512
        stride = 256
        is_overlapped = False

        embeds = []
        for t in tqdm.tqdm(text):
            token_ids = tokenizer.encode(t, return_tensors='pt')

            # Truncate and overlap input sequences.
            input_tensors = []
            start = 0
            while start < token_ids.shape[1]: # 0th dim is batch.
                if start != 0:
                    is_overlapped = True  # Record when we overlap long texts to output warning.
                end = min(start + bert_max_length, token_ids.shape[1])
                input_tensors.append((token_ids[:, start:end]))
                start += stride

            # Embed.
            with torch.no_grad():
                outputs = [embed_model(input_tensor.to(device)) for input_tensor in input_tensors]  
                hidden_states = torch.stack([torch.concat(output.hidden_states, 0).mean(1) for output in outputs], 0).mean(0).cpu() # Mean over seq len (1) and over wrapped sequences (0).
                embeds.append(hidden_states)
        embeds = torch.stack(embeds, 0)
        embeds = embeds.view(embeds.shape[0], -1)

        if is_overlapped:
            print('''
***

WARNING
At least one of your texts is long so it may have been overlapped.
In general, this is okay. If you wish to use all your data, we
suggest you split your long texts into multiple lines. Try to remain
under 300 words per text.

***
''')
        train_embeds = embeds[:len(train)] if train is not None else []
        train_labels = train.label.tolist() if train is not None else []
        val_embeds = embeds[len(train_embeds):len(train_embeds) + len(val)] if val is not None else []
        val_labels = val.label.tolist() if val is not None else []
        test_embeds = embeds[len(train_embeds) + len(val_embeds):]

        self.classes = list(set(train_labels + val_labels))

        # Binarize labels.
        def binarize_labels(labels, classes):
            binarized_labels = sklearn.preprocessing.label_binarize(labels, classes=classes).tolist()
            if len(binarized_labels[0]) == 1:  # Quirk of label_binarize: binary cases must be expanded from a single column vector.
                binarized_labels = [[1, 0] if x == [0] else [0, 1] for x in binarized_labels]
            return binarized_labels
        if len(train_labels) > 0:
            train_labels = binarize_labels(train_labels, self.classes)
        if len(val_labels) > 0:
            val_labels = binarize_labels(val_labels, self.classes)

        return train_embeds, train_labels, val_embeds, val_labels, test_embeds

    def _load_model(self):
        return multiclass.OneVsRestClassifier(linear_model.LogisticRegression(random_state=0, max_iter=1000))
    
    def train(self):
        print(f'\n(2/3) LEARNING...\n')
        self.model.fit(self.train_embeds, self.train_labels)

    def evaluate(self):
        if len(self.val_embeds) == 0:
            return None

        val_preds = self.model.predict(self.val_embeds)
        val_probs = self.model.predict_proba(self.val_embeds)

        accs = {}
        precs = {}
        recs = {}
        f1s = {}
        aurocs = {}
        auprs = {}

        # Get one-vs-all classification metrics for each class.
        print('\nGETTING METRICS...\n')
        for i, class_name in tqdm.tqdm(enumerate(self.classes)):
            class_bin = [0 for _ in range(len(self.classes))]
            class_bin[i] = 1

            labels_bin = [1 if j == class_bin else 0 for j in self.val_labels]
            preds_bin = [1 if j == class_bin else 0 for j in val_preds.tolist()]

            accs[class_name] = metrics.accuracy_score(labels_bin, preds_bin).item()
            precs[class_name] = metrics.precision_score(labels_bin, preds_bin, zero_division=0).item()
            recs[class_name] = metrics.recall_score(labels_bin, preds_bin, zero_division=0).item()
            f1s[class_name] = metrics.f1_score(labels_bin, preds_bin, zero_division=0).item()

            fpr, tpr, _ = metrics.roc_curve(labels_bin, val_probs[:,i])
            aurocs[class_name] = metrics.auc(fpr, tpr).item()

            precision, recall, _ = metrics.precision_recall_curve(labels_bin, val_probs[:,i])
            auprs[class_name] = metrics.auc(recall, precision).item()

        return {
            'acc': accs,
            'prec': precs,
            'rec': recs,
            'f1': f1s,
            'auroc': aurocs,
            'aupr': auprs,
        }
    
    def predict(self):
        print(f'\n(3/3) CREATING TAILWIZ LABELS...\n')
        out_predictions = []
        for i in tqdm.tqdm(range(self.test_embeds.shape[0])):
            prediction = self.model.predict(self.test_embeds[i].unsqueeze(0))
            class_i = np.argmax(prediction[0])
            out_predictions.append(self.classes[class_i])
        return out_predictions


class KMeansClassificationTask(ClassificationTask):
    def __init__(self, test):
        print(f'\n(1/2) PROCESSING DATA...\n')
        (_, _, _, _, self.test_embeds) = self._load_data(None, None, test)
        self.model = self._load_model()
    
    def _load_data(self, train, val, test):
        return super()._load_data(train, val, test)

    def _load_model(self):
        return cluster.KMeans(n_clusters=2, random_state=0, max_iter=1000)
    
    def train(self):
        self.model.fit(self.test_embeds)
    
    def evaluate(self):
        # KMeans task is default task when no train/val data is provided.
        # Thus, it should not have an evaluate function.
        pass
    
    def predict(self):
        print(f'\n(2/2) CREATING TAILWIZ LABELS...\n')
        out_predictions = []
        for i in tqdm.tqdm(range(self.test_embeds.shape[0])):
            prediction = self.model.predict(self.test_embeds[i].unsqueeze(0))
            out_predictions.append(prediction[0])
        return out_predictions  # self.model.predict(self.test_embeds)


def classify(to_classify, labeled_examples=None, output_metrics=False):
    assert isinstance(to_classify, pd.DataFrame), 'Make sure you are passing in pandas DataFrames.'
    assert 'text' in to_classify.columns, 'Make sure the text column in your pandas DataFrame is named "text".'
    if labeled_examples is not None:
        assert isinstance(labeled_examples, pd.DataFrame), 'Make sure you are passing in pandas DataFrames.'
        assert 'text' in labeled_examples.columns and 'label' in labeled_examples.columns, \
            'Make sure the text column in your pandas DataFrame is named "text" and the label column is named "label"'
        assert len(labeled_examples) >= 3, 'labeled_examples has too few examples. At least 3 are required.'
        assert len(labeled_examples.label.unique()) >= 2, 'labeled_examples contains examples from just one class. Examples from at least 2 classes are required.'

    if output_metrics:
        assert labeled_examples is not None, 'In order to output an estimate of performance with output_metrics, labeled_examples must be provided.'

    # Perform KMeans if no training data is given.
    if labeled_examples is None:
        task = KMeansClassificationTask(to_classify)
        task.train()
        pred_results = task.predict()
        results = to_classify.copy()
        results['tailwiz_label'] = pred_results
        return results

    # Try 10 times to get a proper split. Sometimes, a split will cause all training examples to be
    # in the same class, which will error.
    classify_task_out = None
    split_attempt = 0
    while split_attempt < 10:
        train, val = sklearn.model_selection.train_test_split(labeled_examples, test_size=0.2, random_state=0)
        num_unique_classes_in_train = len(train.label.unique())
        if num_unique_classes_in_train < 2:
            split_attempt += 1
            continue
        else:
            classify_task_out = ClassificationTask(train, val, to_classify)
            classify_task_out.train()
            break
    
    if classify_task_out is None:
        raise ValueError('''The provided labeled_examples examples were not diverse enough to estimate performance.
        Try balancing your labeled_examples examples by adding more examples of each class.''')
    
    pred_results = classify_task_out.predict()
    results = to_classify.copy()
    results['tailwiz_label'] = pred_results

    if output_metrics:
        metrics_out = classify_task_out.evaluate()

    print('\nDONE')
    return (results, metrics_out) if output_metrics else results
