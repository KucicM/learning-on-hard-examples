from high_cost_data_engine.repository import HighCostRepository


def test_high_cost_repository_init():
    repo = HighCostRepository()
    ret = repo.connection.execute("SELECT name FROM sqlite_master;").fetchall()
    assert 2 == len(ret)


def test_insert_new_indices_and_classes():
    repo = HighCostRepository()
    targets = [5, 5, 2, 1, 7]
    indices = list(range(len(targets)))

    repo.insert_new_indices_and_classes(indices, targets)

    actual = repo.connection.execute("SELECT * FROM class_cost;").fetchall()
    expected = [(i, t, None) for i, t in zip(indices, targets)]
    assert expected == actual


def test_update_all_costs():
    repo = HighCostRepository()

    targets = [1, 1, 2, 2, 3]
    costs = [0.2, 0.1, 0.3, 0.5, 0.9]
    indices = list(range(len(targets)))

    repo.insert_new_indices_and_classes(indices, targets)
    repo.update_costs(indices, costs)

    expected = [(i, t, c) for i, t, c in zip(indices, targets, costs)]
    actual = repo.connection.execute("SELECT * FROM class_cost;").fetchall()
    assert expected == actual


def test_update_some_costs():
    repo = HighCostRepository()

    targets = [1, 1, 2, 2, 3]
    indices = list(range(len(targets)))
    repo.insert_new_indices_and_classes(indices, targets)

    indices = [0, 1]
    costs = [0.2, 0.1]
    repo.update_costs(indices, costs)

    expected_indices = range(len(targets))
    expected_costs = [0.2, 0.1, None, None, None]
    expected = [(i, t, c) for i, t, c in zip(expected_indices, targets, expected_costs)]
    actual = repo.connection.execute("SELECT * FROM class_cost;").fetchall()
    assert expected == actual


def test_get_top_k_percent_indices():
    repo = HighCostRepository()

    targets = [1, 2, 3]
    indices = list(range(len(targets)))
    repo.insert_new_indices_and_classes(indices, targets)

    costs = [3, 1, 2]
    repo.update_costs(indices, costs)

    expected_indices = [0]
    actual_indices = repo.get_top_k_percent_indices(1 / 3)
    assert expected_indices == actual_indices

    expected_indices = [0, 2]
    actual_indices = repo.get_top_k_percent_indices(2 / 3)
    assert expected_indices == actual_indices

    expected_indices = [0, 2, 1]
    actual_indices = repo.get_top_k_percent_indices(1)
    assert expected_indices == actual_indices


def test_get_top_k_percent_indices_per_class():
    repo = HighCostRepository()

    targets = [1, 2, 3, 1, 2, 3, 1, 2]
    indices = list(range(len(targets)))
    repo.insert_new_indices_and_classes(indices, targets)

    costs = [3, 2, 1, 1, 4, 3, 9, 10]
    repo.update_costs(indices, costs)

    expected_indices = [6, 7, 5]
    actual_indices = repo.get_top_k_percent_indices_per_class(1 / 3)
    assert expected_indices == actual_indices

    expected_indices = [0, 6, 4, 7, 5]
    actual_indices = repo.get_top_k_percent_indices_per_class(2 / 3)
    assert expected_indices == actual_indices

    expected_indices = [0, 3, 6, 1, 4, 7, 2, 5]
    actual_indices = repo.get_top_k_percent_indices_per_class(1)
    assert expected_indices == actual_indices


def test_get_top_k_percent_indices_without_outliers():
    repo = HighCostRepository()

    targets = [1, 1, 1, 1, 2, 2, 2, 2]
    indices = list(range(len(targets)))
    repo.insert_new_indices_and_classes(indices, targets)

    costs = [1, 2, 13, 8, 9, 4, 3, 7]
    repo.update_costs(indices, costs)

    expected_indices = [5]
    actual_indices = repo.get_top_k_percent_indices_without_outliers(1 / 8, 0)
    assert expected_indices == actual_indices

    expected_indices = [4]
    actual_indices = repo.get_top_k_percent_indices_without_outliers(1 / 8, 1)
    assert expected_indices == actual_indices

    expected_indices = [2]
    actual_indices = repo.get_top_k_percent_indices_without_outliers(1 / 8, 2)
    assert expected_indices == actual_indices

    expected_indices = [5, 6]
    actual_indices = repo.get_top_k_percent_indices_without_outliers(1 / 4, 0)
    assert expected_indices == actual_indices

    expected_indices = [4, 3]
    actual_indices = repo.get_top_k_percent_indices_without_outliers(1 / 4, 1)
    assert expected_indices == actual_indices

    expected_indices = [2, 4]
    actual_indices = repo.get_top_k_percent_indices_without_outliers(1 / 4, 2)
    assert expected_indices == actual_indices

    expected_indices = [5, 6, 1, 0]
    actual_indices = repo.get_top_k_percent_indices_without_outliers(1 / 2, 0)
    assert expected_indices == actual_indices

    expected_indices = [4, 3, 7, 5]
    actual_indices = repo.get_top_k_percent_indices_without_outliers(1 / 2, 1)
    assert expected_indices == actual_indices

    expected_indices = [2, 4, 3, 7]
    actual_indices = repo.get_top_k_percent_indices_without_outliers(1 / 2, 2)
    assert expected_indices == actual_indices

    expected_indices = [5, 6, 1, 0]
    actual_indices = repo.get_top_k_percent_indices_without_outliers(1, 0)
    assert expected_indices == actual_indices

    expected_indices = [4, 3, 7, 5, 6, 1, 0]
    actual_indices = repo.get_top_k_percent_indices_without_outliers(1, 1)
    assert expected_indices == actual_indices

    expected_indices = [2, 4, 3, 7, 5, 6, 1, 0]
    actual_indices = repo.get_top_k_percent_indices_without_outliers(1, 2)
    assert expected_indices == actual_indices


def test_get_top_k_percent_indices_per_class_without_outliers():
    repo = HighCostRepository()

    targets = [1, 1, 1, 1, 2, 2, 2, 2]
    indices = list(range(len(targets)))
    repo.insert_new_indices_and_classes(indices, targets)

    costs = [1, 2, 13, 8, 9, 4, 3, 7]
    repo.update_costs(indices, costs)

    expected_indices = [1, 5]
    actual_indices = repo.get_top_k_percent_indices_per_class_without_outliers(1 / 4, 0)
    assert expected_indices == actual_indices

    expected_indices = [3, 7]
    actual_indices = repo.get_top_k_percent_indices_per_class_without_outliers(1 / 4, 1)
    assert expected_indices == actual_indices

    expected_indices = [2, 4]
    actual_indices = repo.get_top_k_percent_indices_per_class_without_outliers(1 / 4, 2)
    assert expected_indices == actual_indices

    expected_indices = [1, 5]
    actual_indices = repo.get_top_k_percent_indices_per_class_without_outliers(1 / 2, 0)
    assert expected_indices == actual_indices

    expected_indices = [3, 1, 7, 5]
    actual_indices = repo.get_top_k_percent_indices_per_class_without_outliers(1 / 2, 1)
    assert expected_indices == actual_indices

    expected_indices = [2, 3, 4, 7]
    actual_indices = repo.get_top_k_percent_indices_per_class_without_outliers(1 / 2, 2)
    assert expected_indices == actual_indices

    expected_indices = [1, 0, 5, 6]
    actual_indices = repo.get_top_k_percent_indices_per_class_without_outliers(3 / 4, 0)
    assert expected_indices == actual_indices

    expected_indices = [3, 1, 7, 5]
    actual_indices = repo.get_top_k_percent_indices_per_class_without_outliers(3 / 4, 1)
    assert expected_indices == actual_indices

    expected_indices = [2, 3, 1, 4, 7, 5]
    actual_indices = repo.get_top_k_percent_indices_per_class_without_outliers(3 / 4, 2)
    assert expected_indices == actual_indices

    expected_indices = [1, 0, 5, 6]
    actual_indices = repo.get_top_k_percent_indices_per_class_without_outliers(1, 0)
    assert expected_indices == actual_indices

    expected_indices = [3, 1, 0, 7, 5, 6]
    actual_indices = repo.get_top_k_percent_indices_per_class_without_outliers(1, 1)
    assert expected_indices == actual_indices

    expected_indices = [2, 3, 1, 0, 4, 7, 5, 6]
    actual_indices = repo.get_top_k_percent_indices_per_class_without_outliers(1, 2)
    assert expected_indices == actual_indices
