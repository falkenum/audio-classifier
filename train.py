from math import ceil, floor
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

max_sounds = 2000
learning_rate = 2e-3
batch_size = 1
epochs = 20
SOUND_DIR = BIRD_SOUNDS_DIR
MODE = "conv"

sounds = AudioDataset(SOUND_DIR, max_sounds, shuffle=True, max_classes=10)

train_sounds, test_sounds = (floor(len(sounds) * 0.8), ceil(len(sounds) * 0.2))
train_data, test_data = random_split(sounds, (train_sounds, test_sounds))

model = None

if MODE == "conv":
    model = ConvModel(output_labels=sounds.num_output_features()).cuda(0)

train_data = DataLoader(train_data, batch_size=batch_size)
test_data = DataLoader(test_data, batch_size=batch_size)

def train_loop(dataset, model, loss_fn, optimizer):

    for batch, (X, y) in enumerate(dataset):
        pred = model(X)
        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 10 == 0:
            current = batch * len(X)
            print(f"loss: {loss.item():>7f}  [{current:>5d}]", flush=True)

    # plt.plot(losses)
    # plt.show()
    
def test_loop(dataset, model, loss_fn):
    size = 0
    num_batches = 0
    test_loss, correct = 0, 0

    test_losses = []

    with torch.no_grad():
        for X, y in dataset:
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
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=5e-4)
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
