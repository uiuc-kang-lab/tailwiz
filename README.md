# Text Labeling AI Wizard (tailwiz)

`tailwiz` is an AI-powered tool for labeling text. It has three main capabilties: classifying text (`tailwiz.classify`), parsing text given context and prompts (`tailwiz.parse`), and generating text given prompts (`tailwiz.generate`).

## Quickstart

Install `tailwiz` by entering into command line:

```
python -m pip install tailwiz
```
Then run the following in a Python environment for a quick example of text classification:

```python
import tailwiz
import pandas as pd

# Create a pandas DataFrame of pre-labeled text. Notice the 'label'
# column contains 'mean' or 'nice' as labels for each text.
prelabeled_text = pd.DataFrame(
    [
        ['You make me vomit', 'mean'],
        ['Love you lots', 'nice'],
        ['You are the best', 'nice'],
    ],
    columns=['text', 'label'],
)

# Create a pandas DataFrame of text to be labeled. Notice that this
# DataFrame does not have a 'label' column. The labels here will be
# created by tailwiz.
text_to_label = pd.DataFrame(
    ['Have a great day', 'I hate you'],
    columns=['text'],
)

# Classify text_to_label using prelabeled_text as reference data.
results = tailwiz.classify(
    text_to_label=text_to_label,
    prelabeled_text=prelabeled_text,
)

# Note how the results are a copy of text_to_label with a new column
# populated with AI-generated labels.
print(results)
```

## Installation

Install `tailwiz` through `pip`:

```
python -m pip install tailwiz
```

## Usage

In this section, we outline the three main functions of `tailwiz` and provide examples.


### <code>tailwiz.classify<i>(text_to_label, prelabeled_text=None, output_metrics=False)</i></code>

Given text, classify the text.
#### Parameters:
- `text_to_label` : _pandas.DataFrame_ with a column named `'text'` (`str`). Text to be classified.
- `prelabeled_text` : _pandas.DataFrame_ with columns named `'text'` (`str`) and `'label'` (`str`, `int`), _default None_. Pre-labeled text to enhance the performance of the classification task. The classified text is in the `'text'` column and the text's labels are in the `'label'` column.
- `output_metrics` : _bool, default False_. Whether to output `performance_estimate` together with results in a tuple.

#### Returns:
- `results` : _pandas.DataFrame_. A copy of `text_to_label` with a new column, `'label_from_tailwiz'`, containing classification results.
- `performance_estimate` : _Dict[str, float]_. Dictionary of metric name to metric value mappings. Included together with results in a tuple if `output_metrics` is True. Uses prelabeled_text to give an estimate of the accuracy of the classification. One vs. all metrics are given for multiclass classification.

#### Example:

```python
import tailwiz
import pandas as pd

prelabeled_text = pd.DataFrame(
    [
        ['You make me vomit', 'mean'],
        ['Love you lots', 'nice'],
        ['You are the best', 'nice'],
    ],
    columns=['text', 'label'],
)
text_to_label = pd.DataFrame(
    ['Have a great day', 'I hate you'],
    columns=['text'],
)
results = tailwiz.classify(
    text_to_label=text_to_label,
    prelabeled_text=prelabeled_text,
)
print(results)
```

### <code>tailwiz.parse<i>(text_to_label, prelabeled_text=None, output_metrics=False)</i></code>

Given a prompt and a context, parse the answer from the context.
#### Parameters:
- `text_to_label` : _pandas.DataFrame_ with columns named `'context'` (`str`) and `'prompt'` (`str`). Labels will be parsed directly from contexts in `'context'` according to the prompts in `'prompt'`.
- `prelabeled_text` : _pandas.DataFrame_ with columns named `'context'` (`str`), `'prompt'` (`str`), and `'label'` (`str`), _default None_. Pre-labeled text to enhance the performance of the parsing task. The labels in `'label'` must be extracted *exactly* from the contexts in `'context'` (as whole words) according to the prompts in `'prompt'`.
- `output_metrics` : _bool, default False_. Whether to output `performance_estimate` together with results in a tuple.

#### Returns:
- `results` : _pandas.DataFrame_. A copy of `text_to_label` with a new column, `'label_from_tailwiz'`, containing parsed results.
- `performance_estimate` : _Dict[str, float]_. Dictionary of metric name to metric value mappings. Included together with results in a tuple if `output_metrics` is True. Uses prelabeled_text to give an estimate of the accuracy of the parsing job.

#### Example:
```python
import tailwiz
import pandas as pd

prelabeled_text = pd.DataFrame(
    [
        ['Extract the money.', 'He owed me $100', '$100'],
        ['Extract the money.', '¥5000 bills are common', '¥5000'],
        ['Extract the money.', 'Eggs rose to €5 this week', '€5'],
    ],
    columns=['prompt', 'context', 'label'],
)
text_to_label = pd.DataFrame(
    [['Extract the money.', 'Try to save at least £10']],
    columns=['prompt', 'context'],
)
results = tailwiz.parse(
    text_to_label=text_to_label,
    prelabeled_text=prelabeled_text,
)
print(results)
```


### <code>tailwiz.generate<i>(text_to_label, prelabeled_text=None, output_metrics=False)</i></code>

Given a prompt, generate an answer.
#### Parameters:
- `text_to_label` : _pandas.DataFrame_ with a column named `'prompt'` (`str`). Prompts according to which labels will generated.
- `prelabeled_text` : _pandas.DataFrame_ with columns named `'prompt'` (`str`) and `'label'` (`str`), _default None_. Pre-labeled text to enhance the performance of the parsing task. The labels in `'label'` should be responses to the prompts in `'prompt'`.
- `output_metrics` : _bool, default False_. Whether to output `performance_estimate` together with results in a tuple.

#### Returns:
- `results` : _pandas.DataFrame_. A copy of `text_to_label` with a new column, `'label_from_tailwiz'`, containing generated results.
- `performance_estimate` : _Dict[str, float]_. Dictionary of metric name to metric value mappings. Included together with results in a tuple if `output_metrics` is True. Uses prelabeled_text to give an estimate of the accuracy of the text generation job.

#### Example:
```python
import tailwiz
import pandas as pd

prelabeled_text = pd.DataFrame(
    [
        ['Label this sentence as "positive" or "negative": I love puppies!', 'positive'],
        ['Label this sentence as "positive" or "negative": I do not like you at all.', 'negative'],
        ['Label this sentence as "positive" or "negative": Love you lots.', 'positive'],
    ],
    columns=['prompt', 'label']
)
text_to_label = pd.DataFrame(
    ['Label this sentence as "positive" or "negative": I am crying my eyes out.'],
    columns=['prompt']
)
results = tailwiz.generate(
    text_to_label=text_to_label,
    prelabeled_text=prelabeled_text,
)
print(results)
```

## Templates (Notebooks)

Use these Jupyter Notebook examples as templates to help load your data and run any of the three `tailwiz` functions:
- For an example of `tailwiz.classify`, see [`examples/classify.ipynb`](https://github.com/timothydai/tailwiz/blob/main/examples/classify.ipynb)
- For an example of `tailwiz.parse`, see [`examples/parse.ipynb`](https://github.com/timothydai/tailwiz/blob/main/examples/parse.ipynb)
- For an example of `tailwiz.generate`, see [`examples/generate.ipynb`](https://github.com/timothydai/tailwiz/blob/main/examples/generate.ipynb)
