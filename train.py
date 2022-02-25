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

num_sounds = 200
learning_rate = 2e-3
# chunks_per_batch = 5
sounds_per_batch = 5
epochs = 20
SOUND_DIR = BIRD_SOUNDS_DIR
MODE = "conv"

# sorted = False
# if MODE == "convlstm":
#     sorted = True

sounds = AudioDataset(SOUND_DIR, num_sounds, shuffle=True, sorted=True)
train_sounds, test_sounds = (floor(num_sounds * 0.9), ceil(num_sounds * 0.1))
train_data, test_data = random_split(sounds, (train_sounds, test_sounds))

model = None

if MODE == "conv":
    model = ConvModel(output_labels=sounds.num_output_features()).cuda(0)
elif MODE == "convlstm":
    model = ConvLSTMModel(output_labels=sounds.num_output_features()).cuda(0)

class AudioLoader:
    def __init__(self, inner_dataset) -> None:
        self.inner_dataset = inner_dataset
        self.chunk_queue = None
        self.next_labels = None
        self.next_sound_idx = None

    def __iter__(self):
        self.chunk_queue = deque()
        self.next_sound_idx = 0
        return self
    
    def __next__(self):
        if len(self.chunk_queue) == 0:

            sound_batch_list = []
            sound_label_list = []
            for _ in range(sounds_per_batch):
                if self.next_sound_idx == len(self.inner_dataset):
                    raise StopIteration

                sound, label = self.inner_dataset[self.next_sound_idx]
                self.next_sound_idx += 1
                sound_label_list.append(label)
                # only first channel for now
                sound_batch_list.append(sound[0])

            sound_batch = torch.nn.utils.rnn.pad_sequence(sound_batch_list, batch_first=True)

            win_length = FFT_SIZE // 2
            hop_length = win_length // 2
            window = torch.hann_window(win_length)

            # pad such that the output chunks are a multiple of the chunk size
            sound_batch = torch.stft(sound_batch, n_fft=FFT_SIZE, win_length=win_length, 
                hop_length=hop_length, window=window, normalized=True, return_complex=True).abs()

            output_len = sound_batch.shape[2]
            output_pad = model.conv_chunk_width - output_len % model.conv_chunk_width
            sound_batch = nn.functional.pad(sound_batch, (0, output_pad), value=0.0)

            data_chunks = torch.split(sound_batch, model.conv_chunk_width, dim=2)

            # there's a short chunk at the end that we don't want, TODO fix this
            self.chunk_queue.extend(data_chunks)
            self.next_labels = torch.tensor(sound_label_list)

        # adding channel dim back in
        next_chunk = self.chunk_queue.pop()[:, None, :, :]
        return next_chunk.cuda(0), self.next_labels.cuda(0)
            
train_data = AudioLoader(train_data)
test_data = AudioLoader(test_data)

def train_loop(dataset, model, loss_fn, optimizer):
    losses=[]

    for batch, (X, y) in enumerate(dataset):
        pred = model(X)
        loss = loss_fn(pred, y)
        losses.append(loss.cpu().detach().numpy())

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
