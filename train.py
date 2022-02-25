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
chunks_per_batch = 40
sounds_per_batch = 10
epochs = 20
SOUND_DIR = BIRD_SOUNDS_DIR
MODE = "conv"

sorted = False
if MODE == "convlstm":
    sorted = True

sounds = AudioDataset(SOUND_DIR, num_sounds, shuffle=True, sorted=sorted)
train_sounds, test_sounds = (floor(num_sounds * 0.9), ceil(num_sounds * 0.1))
train_data, test_data = random_split(sounds, (train_sounds, test_sounds))

model = None

if MODE == "conv":
    model = ConvModel(output_labels=sounds.num_output_features()).cuda(0)
elif MODE == "convlstm":
    model = ConvLSTMModel(output_labels=sounds.num_output_features()).cuda(0)

class AudioChunker(IterableDataset):
    def __init__(self, inner_dataset) -> None:
        self.inner_dataset = inner_dataset
        self.chunk_queue = deque()
        self.next_label = None
        self.next_sound_idx = None
        self.spectrogram = torchaudio.transforms.Spectrogram(n_fft=FFT_SIZE, win_length=FFT_SIZE//2+1, center=False)
        self.to_db = torchaudio.transforms.AmplitudeToDB()

    def __iter__(self):
        self.chunk_queue = deque()
        self.next_sound_idx = 0
        # self.inner_data_queue.extend(self.inner_data_list)
        return self
    
    def __next__(self):
        if len(self.chunk_queue) == 0:
            if self.next_sound_idx == len(self.inner_dataset):
                raise StopIteration
            sound, label = self.inner_dataset[self.next_sound_idx]
            self.next_label = label
            self.next_sound_idx += 1

            sound = self.to_db(self.spectrogram(sound))
            data_chunks = torch.split(sound, model.conv_chunk_width, dim=2)

            short_chunk = data_chunks[-1]
            if short_chunk.shape[2] < model.conv_chunk_width:
                new_dim = short_chunk.shape[0], short_chunk.shape[1], model.conv_chunk_width - short_chunk.shape[2]
                new_values = torch.full(new_dim, -100.0)

                # pad the short chunk
                data_chunks_list = []
                data_chunks_list.extend(data_chunks[:-1])
                data_chunks_list.append(torch.cat((short_chunk, new_values), dim=2))
                data_chunks = tuple(data_chunks_list)

            self.chunk_queue.extend(data_chunks)
        
        return self.chunk_queue.pop().cuda(0), self.next_label.cuda(0)

class ConvLSTMLoader:
    def __init__(self, inner_dataset) -> None:
        self.inner_dataset = inner_dataset
        self.chunk_queue = None
        self.next_labels = None
        self.next_sound_idx = None
        self.spectrogram = torchaudio.transforms.Spectrogram(n_fft=FFT_SIZE, win_length=FFT_SIZE//2+1)
        self.to_db = torchaudio.transforms.AmplitudeToDB()

    def __iter__(self):
        self.chunk_queue = deque()
        self.next_sound_idx = 0
        return self
    
    def __next__(self):
        new_sounds = False
        if len(self.chunk_queue) == 0:
            new_sounds = True
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
            sound_batch = self.to_db(self.spectrogram(sound_batch))

            data_chunks = torch.split(sound_batch, model.conv_chunk_width, dim=2)

            short_chunk = data_chunks[-1]
            if short_chunk.shape[2] < model.conv_chunk_width:
                new_dim = short_chunk.shape[0], short_chunk.shape[1], model.conv_chunk_width - short_chunk.shape[2]
                new_values = torch.full(new_dim, -100.0)

                # pad the short chunk
                data_chunks_list = []
                data_chunks_list.extend(data_chunks[:-1])
                data_chunks_list.append(torch.cat((short_chunk, new_values), dim=2))
                data_chunks = tuple(data_chunks_list)

            self.chunk_queue.extend(data_chunks)

            self.next_labels = torch.tensor(sound_label_list)
        
        return self.chunk_queue.pop().cuda(0), self.next_labels.cuda(0)
            

if MODE == "conv":
    train_data = DataLoader(AudioChunker(train_data), chunks_per_batch)
    test_data = DataLoader(AudioChunker(test_data), chunks_per_batch)
elif MODE == "convlstm":
    train_data = ConvLSTMLoader(train_data)
    test_data = ConvLSTMLoader(test_data)

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
