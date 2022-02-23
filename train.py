from math import ceil, floor, nan
from numpy import count_nonzero
import torch
from torch import nn
from torch.utils.data import random_split, DataLoader, SequentialSampler
import pickle
from common import *
from db import AudioDatabase
from matplotlib import pyplot as plt
import matplotlib
import numpy as np
matplotlib.use("WebAgg")
db = AudioDatabase()
import sys

num_sounds = 2000
num_feature_labels = db.get_num_birds()
learning_rate = 1e-2
batch_size = 20
epochs = 100
dataset_type = BirdsDataset

train_sounds, test_sounds = (floor(num_sounds * 0.9), ceil(num_sounds * 0.1))
train_data = dataset_type(train_sounds, batch_size, shuffle=True)
test_data = dataset_type(test_sounds, batch_size, shuffle=True)

model = BirdModel(n_input=1, n_output=num_feature_labels).cuda(0)

def train_loop(dataloader, model, loss_fn, optimizer):
    for batch, (X, y) in enumerate(dataloader):
        pred = model(X)
        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 10 == 0:
            current = batch * len(X)
            print(f"loss: {loss.item():>7f}  [{current:>5d}]", flush=True)
    
def test_loop(dataloader, model, loss_fn):
    size = 0
    num_batches = 0
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()

            pred_indices = torch.argmax(pred, dim=1)
            for row_idx, pred_idx in enumerate(pred_indices):
                correct += y[row_idx].item() == pred_idx.item()

            size += len(X)
            num_batches += 1

    avg_loss = test_loss / num_batches
    accuracy = correct / size
    print(f"Predicition accuracy: {(100*accuracy):>0.1f}% ({correct}/{size})")
    print(f"Avg loss: {avg_loss:>0.4f}\n", flush=True)


loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
# optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=5e-4)

for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(train_data, model, loss_fn, optimizer)
    test_loop(test_data, model, loss_fn)
    # print("ENTER to continue")
    # sys.stdin.readline()
print("Done!")
# plt.plot(range(10))
# plt.show()
with open(MODEL_PATH, "wb") as f:
    pickle.dump(model, f)
