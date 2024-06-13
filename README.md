# tailwiz

`tailwiz` is an AI-powered tool for analyzing text. It has three main capabilties: classifying text (`tailwiz.classify`), parsing text given context and prompts (`tailwiz.parse`), and generating text given prompts (`tailwiz.generate`).

## Quickstart

Install `tailwiz` by entering into command line:

```
python -m pip install --upgrade tailwiz
```
Then run the following in a Python environment for a quick example of text classification:

```python
import tailwiz
import pandas as pd

# Create a pandas DataFrame of labeled text. The 'label'
# column contains 'mean' or 'nice' as labels for each text.
labeled_examples = pd.DataFrame(
    [
        ['You make me vomit', 'mean'],
        ['Love you lots', 'nice'],
        ['You are the best', 'nice'],
    ],
    columns=['text', 'label'],
)

# Create a pandas DataFrame of text to be classified by tailwiz.
# This DataFrame does not have a 'label' column. The labels here
# will be created by tailwiz.
to_classify = pd.DataFrame(
    ['Have a great day', 'I hate you'],
    columns=['text'],
)

# Classify text using labeled_examples as reference data.
results = tailwiz.classify(
    to_classify,
    labeled_examples=labeled_examples,
)

# The results are a copy of text with a new column populated
# with AI-generated labels.
print(results)
```

## Installation

Install `tailwiz` through `pip` by entering the following into command line:

```
python -m pip install --upgrade tailwiz
```

## Usage

In this section, we outline the three main functions of `tailwiz` and provide examples.


### <code>tailwiz.classify<i>(to_classify, labeled_examples, output_metrics=False, data_split_seed=None)</i></code>

Given text, classify the text.
#### Parameters:
- `to_classify` : _pandas.DataFrame_ with a column named `'text'` (`str`). Text to be classified.
- `labeled_examples` : _pandas.DataFrame_ with columns named `'text'` (`str`) and `'label'` (`str`, `int`). Labeled examples to enhance the performance of the classification task. The classified text is in the `'text'` column and the text's labels are in the `'label'` column.
- `output_metrics` : _bool, default False_. Whether to output `performance_estimate` together with results in a tuple.
- `data_split_seed` : _int, default None_. Controls the shuffling of `labeled_examples` for internal training and evaluation of language models. Setting `data_split_seed` to be an integer ensures reproducible results.

Any additional keyword arguments will override `tailwiz.classify`'s training arguments, specifically scikit-learn's [`LogisticRegression`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html) parameters.

#### Returns:
- `results` : _pandas.DataFrame_. A copy of `to_classify` with a new column, `'tailwiz_label'`, containing classification results.
- `performance_estimate` : _Dict[str, float]_. Dictionary of metric name to metric value mappings. Included together with results in a tuple if `output_metrics` is True. Uses `labeled_examples` to give an estimate of the accuracy of the classification.

#### Example:

```python
import tailwiz
import pandas as pd

df_to_classify = pd.DataFrame(
    ['Have a great day', 'I hate you'],
    columns=['text'],
)
df_labeled_examples = pd.DataFrame(
    [
        ['You make me vomit', 'mean'],
        ['Love you lots', 'nice'],
        ['You are the best', 'nice'],
    ],
    columns=['text', 'label'],
)
results = tailwiz.classify(
    to_classify=df_to_classify,
    labeled_examples=df_labeled_examples,
)
print(results)
```

### <code>tailwiz.parse<i>(to_parse, labeled_examples=None, output_metrics=False, data_split_seed=None)</i></code>

Given a prompt and a context, parse the answer from the context.
#### Parameters:
- `to_parse` : _pandas.DataFrame_ with columns named `'context'` (`str`) and `'prompt'` (`str`). Labels will be parsed directly from contexts in `'context'` according to the prompts in `'prompt'`.
- `labeled_examples` : _pandas.DataFrame_ with columns named `'context'` (`str`), `'prompt'` (`str`), and `'label'` (`str`), _default None_. Labeled examples to enhance the performance of the parsing task. The labels in `'label'` must be extracted *exactly* from the contexts in `'context'` (as whole words) according to the prompts in `'prompt'`.
- `output_metrics` : _bool, default False_. Whether to output `performance_estimate` together with results in a tuple.
- `data_split_seed` : _int, default None_. Controls the shuffling of `labeled_examples` for internal training and evaluation of language models. Setting `data_split_seed` to be an integer ensures reproducible results.

Any additional keyword arguments will override `tailwiz.parse`'s training arguments, specifically Hugging Face's [`TrainingArguments`](https://huggingface.co/docs/transformers/v4.30.0/en/main_classes/trainer#transformers.TrainingArguments) parameters.

#### Returns:
- `results` : _pandas.DataFrame_. A copy of `to_parse` with a new column, `'tailwiz_label'`, containing parsed results.
- `performance_estimate` : _Dict[str, float]_. Dictionary of metric name to metric value mappings. Included together with results in a tuple if `output_metrics` is True. Uses `labeled_examples` to give an estimate of the accuracy of the parsing job.

#### Example:
```python
import tailwiz
import pandas as pd

df_to_parse = pd.DataFrame(
    [['Extract the money.', 'Try to save at least £10']],
    columns=['prompt', 'context'],
)
df_labeled_examples = pd.DataFrame(
    [
        ['Extract the money.', 'He owed me $100', '$100'],
        ['Extract the money.', '¥5000 bills are common', '¥5000'],
        ['Extract the money.', 'Eggs rose to €5 this week', '€5'],
    ],
    columns=['prompt', 'context', 'label'],
)
results = tailwiz.parse(
    to_parse=df_to_parse,
    labeled_examples=df_labeled_examples,
)
print(results)
```


### <code>tailwiz.generate<i>(to_generate, labeled_examples=None, output_metrics=False, data_split_seed=None)</i></code>

Given a prompt, generate an answer.
#### Parameters:
- `to_generate` : _pandas.DataFrame_ with a column named `'prompt'` (`str`). Prompts according to which labels will generated.
- `labeled_examples` : _pandas.DataFrame_ with columns named `'prompt'` (`str`) and `'label'` (`str`), _default None_. Labeled examples to enhance the performance of the parsing task. The labels in `'label'` should be responses to the prompts in `'prompt'`.
- `output_metrics` : _bool, default False_. Whether to output `performance_estimate` together with results in a tuple.
- `data_split_seed` : _int, default None_. Controls the shuffling of `labeled_examples` for internal training and evaluation of language models. Setting `data_split_seed` to be an integer ensures reproducible results.

Any additional keyword arguments will override `tailwiz.generate`'s training arguments, specifically Hugging Face's [`Seq2SeqTrainingArguments`](https://huggingface.co/docs/transformers/main_classes/trainer#transformers.Seq2SeqTrainingArguments) parameters.

#### Returns:
- `results` : _pandas.DataFrame_. A copy of `to_generate` with a new column, `'tailwiz_label'`, containing generated results.
- `performance_estimate` : _Dict[str, float]_. Dictionary of metric name to metric value mappings. Included together with results in a tuple if `output_metrics` is True. Uses `labeled_examples` to give an estimate of the accuracy of the text generation job.

#### Example:
```python
import tailwiz
import pandas as pd

df_to_generate = pd.DataFrame(
    ['Label this sentence as "positive" or "negative": I am crying my eyes out.'],
    columns=['prompt']
)
df_labeled_examples = pd.DataFrame(
    [
        ['Label this sentence as "positive" or "negative": I love puppies!', 'positive'],
        ['Label this sentence as "positive" or "negative": I do not like you at all.', 'negative'],
        ['Label this sentence as "positive" or "negative": Love you lots.', 'positive'],
    ],
    columns=['prompt', 'label']
)
results = tailwiz.generate(
    to_generate=df_to_generate,
    labeled_examples=df_labeled_examples,
)
print(results)
```

## Templates (Notebooks)

Use these Jupyter Notebook examples as templates to help load your data and run any of the three `tailwiz` functions:
- For an example of `tailwiz.classify`, see [`examples/classify.ipynb`](https://github.com/timothydai/tailwiz/blob/main/examples/classify.ipynb)
- For an example of `tailwiz.parse`, see [`examples/parse.ipynb`](https://github.com/timothydai/tailwiz/blob/main/examples/parse.ipynb)
- For an example of `tailwiz.generate`, see [`examples/generate.ipynb`](https://github.com/timothydai/tailwiz/blob/main/examples/generate.ipynb)


## Contact
Please contact Daniel Kang (ddkang [at] g.illinois.edu) and Timothy Dai (timdai [at] stanford.edu) if you decide to use `tailwiz`. 
