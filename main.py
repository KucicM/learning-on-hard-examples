import time

import numpy as np
from torch import optim
from torch.utils.data import DataLoader
from torch import nn

import model
import data


def run():
    net = model.Resnet9()
    net.to("cuda")
    bs = 512

    dataloader = DataLoader(
        data.get_dataset(),
        batch_size=bs,
        shuffle=True,
        num_workers=1,
        pin_memory=True
    )

    optimizer = optim.SGD(
        net.parameters(),
        lr=1,
        weight_decay=0.0005 * bs,
        momentum=0.9,
        nesterov=True
    )

    def calculate_lr():  # TODO not done
        batch_count = len(dataloader)

        def inner(step: int) -> float:
            scale = step / batch_count
            lr = np.interp([scale], [0, 5, 0], [0, .4, 0])
            return (lr / bs)[0]
        return inner

    scheduler = optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=calculate_lr(), )

    loss_fn = nn.CrossEntropyLoss()

    start = time.monotonic()
    for epoch in range(24):
        epoch_start = time.monotonic()

        running_loss = 0
        for i, (x, y) in enumerate(dataloader):
            x, y = x.to("cuda"), y.to("cuda")
            optimizer.zero_grad()

            logits = net.forward(x)
            loss = loss_fn(logits, y)
            loss.backward()

            optimizer.step()
            scheduler.step()

            running_loss += loss.item()
            if i % 100 == 99:
                print(f"loss {running_loss / 100:.3f}")
                running_loss = 0
        print(f"{epoch=} took: {time.monotonic() - epoch_start:.2f}s")
    print(f"total time: {time.monotonic() - start:.2f}s")


if __name__ == "__main__":
    run()
