from high_cost_data_engine.config import Dataset, DataLoader


def test_dataset_config():
    config = {
        "dataset": {
            "data_path": "test_path",
            "normalize": {"mean": [1, 2, 3], "std": [4, 5, 6]},
            "use_selective_backprop": True
        }
    }

    cfg = Dataset(config)
    assert cfg.normalization_values["mean"] == [1, 2, 3]
    assert cfg.normalization_values["std"] == [4, 5, 6]
    assert cfg.data_path == "test_path"
    assert cfg.use_selective_backprop


def test_dataset_config_2():
    config = {
        "dataset": {
            "data_path": "test_path",
            "normalize": {"mean": [1, 2, 3], "std": [4, 5, 6]},
            "use_selective_backprop": False
        }
    }

    cfg = Dataset(config)
    assert cfg.normalization_values["mean"] == [1, 2, 3]
    assert cfg.normalization_values["std"] == [4, 5, 6]
    assert cfg.data_path == "test_path"
    assert not cfg.use_selective_backprop


def test_dataset_config_default_selective_backprop():
    config = {
        "dataset": {
            "data_path": "test_path",
            "normalize": {"mean": [1, 2, 3], "std": [4, 5, 6]},
        }
    }

    cfg = Dataset(config)
    assert cfg.normalization_values["mean"] == [1, 2, 3]
    assert cfg.normalization_values["std"] == [4, 5, 6]
    assert cfg.data_path == "test_path"
    assert not cfg.use_selective_backprop


def test_dataloader_config():
    train_config = {"batch_size": 100}
    test_config = {"batch_size": 200}
    val_config = {"batch_size": 300}
    config = {
        "dataloaders": {
            "train": train_config,
            "test": test_config,
            "val": val_config
        }
    }

    cfg = DataLoader(config)

    assert cfg.train_params == train_config
    assert cfg.test_params == test_config
    assert cfg.val_params == val_config
    assert cfg.train_batch_size == 100


def test_dataloader_config_default_val():
    train_config = {"batch_size": 100}
    test_config = {"batch_size": 200}
    config = {
        "dataloaders": {
            "train": train_config,
            "test": test_config,
        }
    }

    cfg = DataLoader(config)

    assert cfg.train_params == train_config
    assert cfg.test_params == test_config
    assert cfg.val_params is None
    assert cfg.train_batch_size == 100
