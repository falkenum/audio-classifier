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

num_sounds = 280
learning_rate = 1e-3
batch_size = 20
epochs = 50
SOUND_DIR = CAT_DOG_SOUNDS_DIR
MODE = "conv"


# TODO ensure no overlap between train and test
sounds = AudioDataset(SOUND_DIR, num_sounds)
train_sounds, test_sounds = (floor(num_sounds * 0.9), ceil(num_sounds * 0.1))
train_data, test_data = random_split(sounds, (train_sounds, test_sounds))

model = None

if MODE == "conv":
    model = ConvModel(output_labels=sounds.num_output_features()).cuda(0)

class AudioChunker(IterableDataset):
    def __init__(self, inner_dataset) -> None:
        self.inner_data_list = list(inner_dataset)
        self.inner_data_queue = deque()
        self.chunk_queue = deque()
        self.next_label = None

    def __iter__(self):
        self.inner_data_queue = deque()
        self.inner_data_queue.extend(self.inner_data_list)
        return self
    
    def __next__(self):
        if len(self.chunk_queue) == 0:
            if len(self.inner_data_queue) == 0:
                raise StopIteration
            sound, label = self.inner_data_queue.pop()
            self.next_label = label

            data_chunks = torch.split(sound, model.conv_chunk_width, dim=1)
            # don't include short chunk
            self.chunk_queue.extend(data_chunks[:-1])
        
        return self.chunk_queue.pop(), self.next_label

if MODE == "conv":
    train_data = DataLoader(AudioChunker(train_data), batch_size)
    test_data = DataLoader(AudioChunker(test_data), batch_size)

def train_loop(dataset, model, loss_fn, optimizer):
    # chunk_flattener = nn.Flatten(2, 3)

    for batch, (X, y) in enumerate(dataset):
        pred = model(X)
        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 10 == 0:
            current = batch * len(X)
            print(f"loss: {loss.item():>7f}  [{current:>5d}]", flush=True)
    
def test_loop(dataset, model, loss_fn):
    size = 0
    num_batches = 0
    test_loss, correct = 0, 0

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
