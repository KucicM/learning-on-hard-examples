from data_engine.database.provider import HighLossIdProvider
import numpy as np


def test_q_inserts():
    provider = HighLossIdProvider()
    index_generator = range(50)
    new_generator = provider.add_index_to_queue(index_generator)

    for i, _ in enumerate(new_generator, start=1):
        assert i == len(provider._index_queue)

    for expected_value in index_generator:
        assert expected_value == provider._index_queue.popleft()


def test_insert_into_db():
    provider = HighLossIdProvider(":memory:")
    connection = provider._repository._connection

    # populate queue
    for _ in provider.add_index_to_queue(range(5)):
        pass

    losses = [3, 1, 5, 3, 0]
    provider.update_losses(losses)

    actual = connection.execute("SELECT idx, loss FROM idx_loss;").fetchall()
    expected = [(i, l) for i, l in zip(range(5), losses)]
    assert actual == expected


def test_insert_followed_by_update():
    provider = HighLossIdProvider(":memory:")
    connection = provider._repository._connection
    ex_num = 20

    # populate queue
    for _ in provider.add_index_to_queue(range(ex_num)):
        pass

    losses = np.random.rand(ex_num).tolist()
    provider.update_losses(losses)

    # new batch
    for _ in provider.add_index_to_queue(range(ex_num)):
        pass

    losses = np.random.rand(ex_num).tolist()
    provider.update_losses(losses)

    actual = connection.execute("SELECT idx, loss FROM idx_loss;").fetchall()
    expected = [(i, l) for i, l in zip(range(ex_num), losses)]
    assert actual == expected
