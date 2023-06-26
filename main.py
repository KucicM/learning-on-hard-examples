import time

from torch import optim
from torch.utils.data import DataLoader
from torch import nn

import model
import data


def run():
    net = model.Resnet9()
    net.to("cuda")

    optimizer = optim.Adam(net.parameters())

    dataloader = DataLoader(
        data.get_dataset(),
        batch_size=512,
        shuffle=True,
        num_workers=1,
        pin_memory=True
    )
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
            running_loss += loss.item()
            if i % 100 == 99:
                print(f"loss {running_loss / 100:.3f}")
                running_loss = 0
        print(f"{epoch=} took: {time.monotonic() - epoch_start:.2f}s")
    print(f"total time: {time.monotonic() - start:.2f}s")


if __name__ == "__main__":
    run()
