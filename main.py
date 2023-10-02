from GPRI2Net import GPRI2Net
import torch
from torch.utils.data import DataLoader
from Dataset import PicDataSet
import numpy as np


def train(dataloader, model, loss_fn, optimizer, device):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test(dataloader, model, loss_fn, device):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y)
    print(f"Test Error: \n Avg loss: {test_loss:>8f} \n")


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    dataset = np.array([
        [np.random.rand(1, 128, 128) for i in range(5)],
        [np.random.rand(1, 128, 128) for i in range(5)]
    ])
    dataset = torch.tensor(dataset)
    print(dataset.shape)
    ds = PicDataSet(dataset, dataset, 3, 3)
    train_dataloader = DataLoader(ds, batch_size=2)
    test_dataloader = DataLoader(ds, batch_size=2)

    model = GPRI2Net()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
