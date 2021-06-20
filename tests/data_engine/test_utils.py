import torch
import pytest
from data_engine import utils


def test_random_shuffle_portion_of_1d_tensor_0():
    torch.random.manual_seed(42)
    init_array = torch.Tensor([5, 4, 3, 2, 1, 1, 2, 3, 4, 5])
    result = utils.random_shuffle_portion_of_1d_tensor(init_array, 0)
    assert torch.all(init_array == result)


def test_random_shuffle_portion_of_1d_tensor_50():
    torch.random.manual_seed(42)
    init = [5, 4, 3, 2, 1, 1, 2, 3, 4, 5]
    tensor = torch.Tensor(init)
    utils.random_shuffle_portion_of_1d_tensor(tensor, 0.5)
    assert torch.sum(torch.Tensor(init) == tensor) != 10


def test_single_item_output():
    @utils.convert_list_of_tuples_to_list
    def get_single_value():
        return 1

    with pytest.raises(TypeError):
        get_single_value()


def test_single_value_in_list():
    @utils.convert_list_of_tuples_to_list
    def get_single_value_in_list():
        return [1]

    with pytest.raises(TypeError):
        get_single_value_in_list()


def test_list_of_tuple_with_single_value_inside():
    @utils.convert_list_of_tuples_to_list
    def get_list_of_tuple_with_single_value():
        return [(1,)]

    values = get_list_of_tuple_with_single_value()
    assert [1] == values


def test_list_of_tuples_with_single_value_inside():
    @utils.convert_list_of_tuples_to_list
    def get_list_of_tuples_with_single_value_inside():
        return [(1,), (2,), (3,)]

    values = get_list_of_tuples_with_single_value_inside()
    assert [1, 2, 3] == values


def test_list_of_tuples_with_multiple_values_inside():
    @utils.convert_list_of_tuples_to_list
    def get_list_of_tuples_with_multiple_values_inside():
        return [(1, 2), (3, 4), (5, 6)]

    values = get_list_of_tuples_with_multiple_values_inside()
    assert [1, 2, 3, 4, 5, 6] == values


def test_load_yml_non_existing_file():
    with pytest.raises(FileNotFoundError):
        utils.load_yml("NON_EXISTING_PATH.yml")


def test_load_yml_none_file_path():
    assert None is utils.load_yml(None)


def test_load_yml():
    import tempfile
    import os

    file = tempfile.NamedTemporaryFile(delete=False)
    yml = b"""
    test: 1
    nested:
      test1: 1
      test2: 2
      test3:
        test: 1
        test2: test
    """
    file.write(yml)
    file.close()
    config = utils.load_yml(file.name)
    expected_dir = {"test": 1, "nested": {"test1": 1, "test2": 2, "test3": {"test": 1, "test2": "test"}}}
    assert expected_dir == config
    os.unlink(file.name)
    assert not os.path.exists(file.name)
