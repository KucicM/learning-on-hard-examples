import time

import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader
from torch import nn

import model
import data

torch.backends.cudnn.benchmark = True


def run():
    net = model.Resnet9().cuda()
    bs = 512

    dataloader = DataLoader(
        data.get_dataset(),
        batch_size=bs,
        shuffle=True,
        num_workers=12,
        pin_memory=True
    )

    optimizer = model.StepOptimizer(
        net.parameters(),
        optimizer=optim.SGD,
        weight_decay=0.0005 * bs,
        momentum=0.9,
        nesterov=True,
        lr=lambda step: (np.interp([step / len(dataloader)], [0, 5, 0], [0, .4, 0]) / bs)[0]
    )

    loss_fn = nn.CrossEntropyLoss().cuda()

    start = time.monotonic()
    for epoch in range(24):
        epoch_start = time.monotonic()

        for i, (x, y) in enumerate(dataloader):
            x, y = x.cuda(), y.cuda()
            for param in net.parameters():
                param.grad = None

            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
                logits = net.forward(x)
                loss = loss_fn(logits, y)
            loss.backward()

            optimizer.step()

        print(f"{epoch=} took: {time.monotonic() - epoch_start:.2f}s")
    print(f"total time: {time.monotonic() - start:.2f}s")


if __name__ == "__main__":
    run()
