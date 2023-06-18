import tailwiz
import pandas as pd


def test_generate_no_training_data():
    results = tailwiz.generate(pd.DataFrame(['Is this sentence Happy or Sad? I am crying my eyes out.'], columns=['prompt']), None, False)
    assert len(results) == 1
    assert type(results.tailwiz_label.iloc[0]) == str


def test_generate():
    results = tailwiz.generate(
        pd.DataFrame(['Is this sentence Happy or Sad? I am crying my eyes out.'], columns=['prompt']),
        pd.DataFrame([
            ['Is this sentence Happy or Sad? I love puppies!', 'Happy'],
            ['Is this sentence Happy or Sad? I do not like you at all.', 'Sad'],
        ], columns=['prompt', 'label']), False)
    assert len(results) == 1
    assert type(results.tailwiz_label.iloc[0]) == str


def test_generate_long():
    results = tailwiz.generate(
        pd.DataFrame(['Is this sentence Happy or Sad? I am crying my eyes out.' * 1000], columns=['prompt']),
        pd.DataFrame([
            ['Is this sentence Happy or Sad? I love puppies!', 'Happy'],
            ['Is this sentence Happy or Sad? I do not like you at all.', 'Sad'],
        ], columns=['prompt', 'label']), False)
    assert len(results) == 1
    assert type(results.tailwiz_label.iloc[0]) == str

    
def test_generate_stress():
    results = tailwiz.generate(
        pd.DataFrame(['Is this sentence Happy or Sad? I am crying my eyes out.'], columns=['prompt']),
        pd.DataFrame([
            ['Is this sentence Happy or Sad? I love puppies!', 'Happy'],
        ] * 200, columns=['prompt', 'label']), False)
    assert len(results) == 1
    assert type(results.tailwiz_label.iloc[0]) == str


def test_generate_with_metrics():
    results, metrics = tailwiz.generate(
        pd.DataFrame(['Is this sentence Happy or Sad? I am crying my eyes out.'], columns=['prompt']),
        pd.DataFrame([
            ['Is this sentence Happy or Sad? I love puppies!', 'Happy'],
            ['Is this sentence Happy or Sad? I do not like you at all.', 'Sad'],
        ], columns=['prompt', 'label']), True)
    assert len(results) == 1
    assert type(results.tailwiz_label.iloc[0]) == str
    assert metrics is not None
    assert 'rouge1' in metrics
    assert type(metrics['rouge1']) == float


def test_generate_override_train_args():
    results = tailwiz.generate(
        pd.DataFrame(['Is this sentence Happy or Sad? I am crying my eyes out.'], columns=['prompt']),
        pd.DataFrame([
            ['Is this sentence Happy or Sad? I love puppies!', 'Happy'],
            ['Is this sentence Happy or Sad? I do not like you at all.', 'Sad'],
        ], columns=['prompt', 'label']),
        output_dir='cache/changed_dir_generate')
    assert len(results) == 1
    assert type(results.tailwiz_label.iloc[0]) == str
