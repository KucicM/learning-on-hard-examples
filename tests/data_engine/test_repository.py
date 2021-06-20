from data_engine.database.repository import LossRepository
import torch

DB_HOST = ":memory:"


def test_db_creation():
    repo = LossRepository(DB_HOST)
    ret = repo._connection.execute("SELECT name FROM sqlite_master;").fetchall()
    assert 1 == len(ret)


def test_db_insert():
    repo = LossRepository(DB_HOST)
    indices = [1, 2, 3, 4, 5]
    losses = [0.2, 1, 0.9, 2, 8]

    repo.insert_or_replace(zip(indices, losses))

    actual = repo._connection.execute("SELECT idx, loss FROM idx_loss;").fetchall()
    expected = [(i, l) for i, l in zip(indices, losses)]
    assert expected == actual


def test_db_update():
    repo = LossRepository(DB_HOST)
    indices = [1, 2, 3, 4, 5]
    losses = [0.2, 1, 0.9, 2, 8]

    repo.insert_or_replace(zip(indices, losses))

    update_indices = [3, 5]
    update_losses = [9, 3]

    repo.insert_or_replace(zip(update_indices, update_losses))

    actual = repo._connection.execute("SELECT idx, loss FROM idx_loss;").fetchall()
    expected = [(i, l) for i, l in zip(indices, [0.2, 1, 9, 2, 3])]
    assert expected == actual


def test_full_db_update():
    repo = LossRepository(DB_HOST)
    indices = list(range(10_000))
    losses = list(range(10_000))

    repo.insert_or_replace(zip(indices, losses))
    repo.insert_or_replace(zip(indices, losses))

    actual = repo._connection.execute("SELECT idx, loss FROM idx_loss;").fetchall()
    expected = [(i, l) for i, l in zip(indices, losses)]
    assert expected == actual


def test_selective_backpopr():
    torch.random.manual_seed(42)

    repo = LossRepository(DB_HOST)
    indices = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    losses = [3, 5, 2, 1, 9, 7, 4, 6, 8, 100]

    repo.insert_or_replace(zip(indices, losses))
    expected = [10, 9, 8, 2, 6]  # importance sampling

    actual = repo.selective_backprop(0.5)
    assert actual == expected


def test_robust_selective_backprop():
    torch.random.manual_seed(42)

    repo = LossRepository(DB_HOST)
    indices = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    losses = [3, 5, 2, 1, 9, 7, 4, 6, 8, 100]

    repo.insert_or_replace(zip(indices, losses))
    expected = [9, 8, 2, 6, 5]  # importance sampling

    actual = repo.robust_selective_backporp(cutoff=0.5, std_num=1)
    assert actual == expected

