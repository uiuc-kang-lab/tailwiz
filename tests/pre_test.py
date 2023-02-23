import tailwiz


def test_list_tasks():
    assert len(tailwiz.list_tasks().split('\n')) == 3
