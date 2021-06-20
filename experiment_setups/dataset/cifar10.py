from data_engine.datamodule import HighLossDataModule
from data_engine.sampler import HighLossSampler
from experiment_setups.dataset.cutout import Cutout
from torchvision import transforms
from torchvision.datasets import CIFAR10
from functools import partial
from data_engine import utils


class ShuffledDataset:
    root = "data/cifar10"

    def __init__(self, dataset, proportion: float):
        self._dataset = dataset
        self._proportion = proportion

    def __call__(self, train: bool, download: bool = False, transform=None):
        if download:
            return self._dataset(root=self.root, train=train, download=True)

        dataset = self._dataset(root=self.root, train=train, download=False, transform=transform)
        self._shuffle_targets(dataset)

        return dataset

    def _shuffle_targets(self, dataset):
        targets = dataset.targets
        utils.random_shuffle_portion_of_1d_tensor(targets, self._proportion)


class CIFAR10Impl(HighLossDataModule):
    MEAN = [0.4914117647058824, 0.48215686274509806, 0.44654901960784316]
    STD = [0.24701960784313726, 0.2434901960784314, 0.2615686274509804]

    def __init__(
            self,
            shuffle_proportion: float,
            sampler: HighLossSampler,
            train_dataloader: partial,
            val_dataloader: partial,
            test_dataloader: partial
    ):
        dataset = ShuffledDataset(CIFAR10, shuffle_proportion)
        super().__init__(dataset, sampler, train_dataloader, val_dataloader, test_dataloader)

    @property
    def train_transforms(self) -> transforms.Compose:
        return transforms.Compose([
            transforms.Pad(padding=4, fill=0, padding_mode="reflect"),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.MEAN, std=self.STD),
            transforms.RandomCrop(size=32),
            transforms.RandomHorizontalFlip(),
            Cutout(size=8)
        ])

    @property
    def test_transforms(self) -> transforms.Compose:
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=self.MEAN, std=self.STD)
        ])
