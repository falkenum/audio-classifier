from math import ceil, floor, nan
from numpy import count_nonzero
import torch
from torch.utils.data import random_split, DataLoader, SequentialSampler
import pickle
from common import *
from db import AudioDatabase
from matplotlib import pyplot as plt
import matplotlib
import numpy as np
matplotlib.use("WebAgg")
db = AudioDatabase()

num_sounds = 1000
chunk_size = 160000
num_feature_labels = db.get_num_birds()
num_feature_data = 1
learning_rate = 1e-3
batch_size = 20
epochs = 20
dataset_type = BirdsDataset

train_sounds, test_sounds = (floor(num_sounds * 0.9), ceil(num_sounds * 0.1))
train_data = dataset_type(train_sounds, chunk_size, shuffle=True)
test_data = dataset_type(test_sounds, chunk_size, shuffle=True)

model = AudioClassifierModule(num_feature_data, num_feature_labels, chunk_size).cuda(0)


train_dataloader = DataLoader(train_data, batch_size=batch_size, drop_last=True)
test_dataloader = DataLoader(test_data, batch_size=batch_size, drop_last=True)

def train_loop(dataloader, model, loss_fn, optimizer):
    # losses = []
    # size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)
        # losses.append(loss.item())

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss = loss.item()
            current = batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}]")
    
def test_loop(dataloader, model, loss_fn):
    size = 0
    num_batches = 0
    test_loss, correct = 0, 0
    losses = []

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            losses.append(loss_fn(pred, y).item())
            test_loss += losses[-1]

            pred = torch.argmax(pred, 1)
            for row_idx, pred_val in enumerate(pred):
                correct += y[row_idx] == pred_val

            size += len(X)
            num_batches += 1

    avg_loss = test_loss / num_batches
    accuracy = correct / size
    print(f"Predicition accuracy: {(100*accuracy):>0.1f}% ({correct}/{size}")
    print(f"Avg loss: {avg_loss:>0.4f}")
    print()
    # plt.plot(losses)
    # plt.show()


loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=5e-5)
# optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=5e-4)

for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(train_dataloader, model, loss_fn, optimizer)
    test_loop(test_dataloader, model, loss_fn)
print("Done!")
# plt.plot(range(10))
# plt.show()
with open(MODEL_PATH, "wb") as f:
    pickle.dump(model, f)
