import tailwiz
import pandas as pd

# Create a pandas DataFrame of labeled text. Notice the 'label'
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
# Notice that this DataFrame does not have a 'label' column.
# The labels here will be created by tailwiz.
text = pd.DataFrame(
    ['Have a great day', 'I hate you'],
    columns=['text'],
)

# Classify text using labeled_examples as reference data.
results = tailwiz.classify(
    text,
    labeled_examples=labeled_examples,
)

# Note how the results are a copy of text with a new column
# populated with AI-generated labels.
print(results)

import tailwiz
import pandas as pd

labeled_examples = pd.DataFrame(
    [
        ['You make me vomit', 'mean'],
        ['Love you lots', 'nice'],
        ['You are the best', 'nice'],
    ],
    columns=['text', 'label'],
)
text = pd.DataFrame(
    ['Have a great day', 'I hate you'],
    columns=['text'],
)
results = tailwiz.classify(
    text,
    labeled_examples=labeled_examples,
)
print(results)

import tailwiz
import pandas as pd

labeled_examples = pd.DataFrame(
    [
        ['Extract the money.', 'He owed me $100', '$100'],
        ['Extract the money.', '¥5000 bills are common', '¥5000'],
        ['Extract the money.', 'Eggs rose to €5 this week', '€5'],
    ],
    columns=['prompt', 'context', 'label'],
)
text = pd.DataFrame(
    [['Extract the money.', 'Try to save at least £10']],
    columns=['prompt', 'context'],
)
results = tailwiz.parse(
    text,
    labeled_examples=labeled_examples,
)
print(results)

import tailwiz
import pandas as pd

labeled_examples = pd.DataFrame(
    [
        ['Label this sentence as "positive" or "negative": I love puppies!', 'positive'],
        ['Label this sentence as "positive" or "negative": I do not like you at all.', 'negative'],
        ['Label this sentence as "positive" or "negative": Love you lots.', 'positive'],
    ],
    columns=['prompt', 'label']
)
text = pd.DataFrame(
    ['Label this sentence as "positive" or "negative": I am crying my eyes out.'],
    columns=['prompt']
)
results = tailwiz.generate(
    text,
    labeled_examples=labeled_examples,
)
print(results)