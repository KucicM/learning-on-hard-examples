from experiment_setups.dataset.cutout import Cutout
import torch


def test_cutout():
    img = torch.ones((3, 10, 10))
    size = 4
    expected_difference = (size**2) * 3

    cutout = Cutout(size)
    processed = cutout(img)

    assert torch.sum(img - processed) == expected_difference
