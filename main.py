import time

import torch
from torch import nn

import model
import data

torch.backends.cudnn.benchmark = True


def run():
    batch_size, epochs = 512, 24
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type != "cpu" else torch.float32

    train_dataloader, test_dataloader = data.get_dataloaders(batch_size)

    net = model.Resnet9().to(device).to(dtype)

    optimizer = model.get_optimizer(
        weights=net.parameters(),
        epochs=epochs,
        batches=len(train_dataloader),
        batch_size=batch_size
    )

    loss_fn = nn.CrossEntropyLoss(reduction="none").to(device)

    start = time.monotonic()
    for epoch in range(epochs):
        epoch_start = time.monotonic()

        running_loss = 0
        for x, y in train_dataloader:
            x, y = x.to(device).to(dtype), y.to(device).long()
            for param in net.parameters():
                param.grad = None

            logits = net.forward(x)
            loss = loss_fn(logits, y)
            loss.sum().backward()

            running_loss += loss.mean().item()

            optimizer.step()

        for x, y in train_dataloader:
            x, y = x.to(device).to(dtype), y.to(device).long()
            with torch.no_grad():
                logits = net.forward(x)
                loss = loss_fn(logits, y)
            train_dataloader.update(loss)

        epoch_time = time.monotonic() - epoch_start
        accuracy = eval(net, test_dataloader, device, dtype)
        avg_loss = running_loss / len(train_dataloader)
        print(f"{epoch=} {epoch_time=:.2f}s {accuracy=:.2f}% {avg_loss=:.3f}")
    print(f"total time: {time.monotonic() - start:.2f}s")


@torch.no_grad()
def eval(net, dataloader, device, dtype):
    correct = 0
    for x, y in dataloader:
        x, y = x.to(device).to(dtype), y.to(device).long()
        logits = net.forward(x)

        yp = logits.data.max(1, keepdim=True)[1]
        correct += yp.eq(y.data.view_as(yp)).sum().cpu()
    return 100 * correct.item() / len(dataloader.dataset)


if __name__ == "__main__":
    run()
