import random
from typing import Tuple

import torch
from torch.utils.data import DataLoader, Sampler
from torchvision import transforms
from torchvision.datasets import CIFAR10

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


class Cutout:
    """
    Simplified Cutout regularization as proposed by DeVries and Taylor (2017), https://arxiv.org/pdf/1708.04552.pdf.
    """

    def __init__(self, size=8):
        assert size > 0
        self.size = size

    def __call__(self, tensor):
        _, height, width = tensor.shape
        x, y = self._get_random_xy(height, width)

        cutout = torch.ones(height, width)
        cutout[x:x+self.size, y:y+self.size] = 0

        return tensor * cutout

    def _get_random_xy(self, height: int, width: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = torch.randint(low=0, high=(height - self.size), size=(1,))
        y = torch.randint(low=0, high=(width - self.size), size=(1,))
        return x, y


class HighLossSampler(Sampler):
    def __init__(self, dataset):
        self.len = int(len(dataset) * 1)
        self.losses = [5] * len(dataset)
        self.ids = []
        self.id = 0

    def update(self, losses):
        for loss in losses.cpu():
            self.losses[self.ids[self.id]] = loss.item()
            self.id += 1

    def __len__(self):
        return self.len

    def __iter__(self):
        self.id = 0
        self.ids = list(range(self.len))
        self.ids.sort(key=lambda i: self.losses[i], reverse=True)
        random.shuffle(self.ids[:self.len])
        return map(int, self.ids[:self.len])


class HighLossDataloader(DataLoader):
    def __init__(self, dataset, sampler, **kwargs):
        super().__init__(dataset, sampler=sampler, **kwargs)
        self._sampler: HighLossSampler = sampler

    def update(self, losses):
        self._sampler.update(losses)


def get_dataloaders(batch_size):
    train_dataset = _train_dataset()
    train_dataloader = HighLossDataloader(
        train_dataset,
        sampler=HighLossSampler(train_dataset),
        batch_size=batch_size,
        num_workers=3,
        pin_memory=True,
        drop_last=True
    )

    test_dataloader = DataLoader(
        _test_dataset(),
        batch_size=batch_size,
        num_workers=3,
        pin_memory=True,
        drop_last=False
    )
    return train_dataloader, test_dataloader


def _train_dataset():
    transform = transforms.Compose([
        transforms.Pad(padding=4, fill=0, padding_mode="reflect"),
        transforms.ToTensor(),
        transforms.Normalize(mean=CIFAR10_MEAN, std=CIFAR10_STD),
        transforms.RandomCrop(size=32),
        transforms.RandomHorizontalFlip(),
        Cutout(size=8)
    ])
    return CIFAR10("data/", train=True, download=True, transform=transform)


def _test_dataset():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=CIFAR10_MEAN, std=CIFAR10_STD)
    ])
    return CIFAR10("data/", train=False, download=True, transform=transform)
