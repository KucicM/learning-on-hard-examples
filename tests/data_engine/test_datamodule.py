from data_engine.datamodule import HighLossDataModule
from data_engine.sampler import HighLossSampler
from functools import partial

mockHighLossSampler = HighLossSampler("", None, None, None)
mockPartialDataLoader = partial(lambda x: x)


def test_prepare_data():
    events = []

    class MockDataSet:
        def __call__(self, train, download):
            assert download
            events.append(train)

    datamodule = HighLossDataModule(
        MockDataSet(),
        mockHighLossSampler,
        mockPartialDataLoader,
        mockPartialDataLoader,
        mockPartialDataLoader
    )

    datamodule.prepare_data()

    assert [True, False] == events


def test_setup():
    events = []

    class MockDataSet:
        def __call__(self, train, transform):
            events.append((train, transform))

        @property
        def targets(self):
            return list(range(10))

    datamodule = HighLossDataModule(
        MockDataSet(),
        mockHighLossSampler,
        mockPartialDataLoader,
        mockPartialDataLoader,
        mockPartialDataLoader
    )

    datamodule.setup("fit")

    assert [(True, None), (False, None)] == events


def test_train_dataloader():
    class MockDataSet:
        def __call__(self, train, transform):
            if train:
                return "TRAIN"
            else:
                return "NOT-TRAIN"

    datamodule = HighLossDataModule(
        MockDataSet(),
        mockHighLossSampler,
        train_dataloader=partial(lambda dataset: dataset),
        val_dataloader=mockPartialDataLoader,
        test_dataloader=mockPartialDataLoader
    )

    datamodule.setup("fit")
    train_dataloader = datamodule.train_dataloader()
    assert train_dataloader == "TRAIN"


def test_test_dataloader():
    class MockDataSet:
        def __call__(self, train, transform):
            if train:
                return "TRAIN"
            else:
                return "NOT-TRAIN"

    datamodule = HighLossDataModule(
        MockDataSet(),
        mockHighLossSampler,
        train_dataloader=mockPartialDataLoader,
        val_dataloader=mockPartialDataLoader,
        test_dataloader=partial(lambda dataset: dataset)
    )

    datamodule.setup("test")
    test_dataloader = datamodule.test_dataloader()
    assert test_dataloader == "NOT-TRAIN"


def test_val_dataloader_traditional_learning():
    class MockDataSet:
        def __call__(self, train, transform):
            if train:
                return "TRAIN"
            else:
                return "NOT-TRAIN"

    datamodule = HighLossDataModule(
        MockDataSet(),
        mockHighLossSampler,
        train_dataloader=mockPartialDataLoader,
        val_dataloader=None,
        test_dataloader=partial(lambda dataset: dataset)
    )

    datamodule.setup("fit")
    val_dataloader = datamodule.val_dataloader()
    assert val_dataloader == "NOT-TRAIN"


def test_val_dataloader():
    class MockDataSet:
        def __call__(self, train, transform):
            if train:
                return "TRAIN"
            else:
                return "NOT-TRAIN"

    datamodule = HighLossDataModule(
        MockDataSet(),
        mockHighLossSampler,
        train_dataloader=mockPartialDataLoader,
        val_dataloader=partial(lambda dataset: dataset),
        test_dataloader=partial(lambda dataset: dataset)
    )

    datamodule.setup("fit")
    val_dataloader = datamodule.val_dataloader()
    assert val_dataloader == ["NOT-TRAIN", "TRAIN"]
