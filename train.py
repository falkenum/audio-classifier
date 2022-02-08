from math import ceil, floor
import torch
from torch.utils.data import random_split, DataLoader
import pickle
import common

class AudioClassifierModule(torch.nn.Module):
    def __init__(self, out_features) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(common.FEATURE_SIZE, out_features)
    
    def forward(self, x):
        return self.linear(x)



with open(common.DATAPATH, "rb") as f:
    dataset = pickle.load(f)

model = AudioClassifierModule(dataset.labels.shape[0])

learning_rate = 1e-3
batch_size = 32
epochs = 20

data_lengths = (floor(len(dataset) * 0.9), ceil(len(dataset) * 0.1))
train_data, test_data = random_split(dataset, data_lengths)
train_dataloader = DataLoader(train_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

loss_fn = torch.nn.NLLLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(train_dataloader, model, loss_fn, optimizer)
    test_loop(test_dataloader, model, loss_fn)
print("Done!")

