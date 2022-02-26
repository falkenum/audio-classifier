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

max_sounds = 200
learning_rate = 2e-3
conv_chunks_per_batch = 40
lstm_sounds_per_batch = 3
epochs = 20
SOUND_DIR = BIRD_SOUNDS_DIR
MODE = "lstm"

sounds = AudioDataset(SOUND_DIR, max_sounds, shuffle=True, max_classes=5)

train_sounds, test_sounds = (floor(len(sounds) * 0.9), ceil(len(sounds) * 0.1))
train_data, test_data = random_split(sounds, (train_sounds, test_sounds))

model = None

if MODE == "conv":
    model = ConvModel(output_labels=sounds.num_output_features()).cuda(0)
elif MODE == "lstm":
    model = LSTMModel(output_labels=sounds.num_output_features()).cuda(0)

class AudioLoader:
    def __init__(self, inner_dataset) -> None:
        self.inner_dataset = inner_dataset
        self.next_sound_idx = None
        
        if MODE == "conv":
            self._next_fn = AudioLoader._next_conv_mode
            self.chunk_queue = None
            self.next_labels = None
        elif MODE == "lstm":
            self._next_fn = AudioLoader._next_lstm_mode
        else:
            raise NotImplementedError
    
    def __next__(self):
        return self._next_fn(self)

    def __iter__(self):
        self.chunk_queue = deque()
        self.next_sound_idx = 0
        
        return self

    def _next_conv_mode(self):
        while len(self.chunk_queue) < conv_chunks_per_batch:
            if self.next_sound_idx == len(self.inner_dataset):
                # dropping the short batch
                raise StopIteration

            sound, label = self.inner_dataset[self.next_sound_idx]
            self.next_sound_idx += 1

            win_length = FFT_SIZE // 2
            hop_length = win_length // 2
            window = torch.hann_window(win_length)

            # pad such that the output chunks are a multiple of the chunk size
            sound = torch.stft(sound, n_fft=FFT_SIZE, win_length=win_length, 
                hop_length=hop_length, window=window, return_complex=True).abs()

            output_len = sound.shape[2]
            output_pad = model.conv_chunk_width - output_len % model.conv_chunk_width
            sound = nn.functional.pad(sound, (0, output_pad), value=0.0)

            data_chunks = torch.split(sound, model.conv_chunk_width, dim=2)

            self.chunk_queue.extend([(chunk, label) for chunk in data_chunks])
            # self.next_labels = torch.tensor(sound_label_list)
        
        batch_list = []
        batch_label_list = []
        for _ in range(conv_chunks_per_batch):
            chunk, label = self.chunk_queue.popleft()
            batch_list.append(chunk[None, :, : , :])
            batch_label_list.append(label.item())

        return torch.cat(batch_list).cuda(0), torch.tensor(batch_label_list).cuda(0)
    
    def _next_lstm_mode(self):
        sound_batch_list = []
        sound_label_list = []
        for _ in range(lstm_sounds_per_batch):
            if self.next_sound_idx == len(self.inner_dataset):
                break

            sound, label = self.inner_dataset[self.next_sound_idx]
            self.next_sound_idx += 1
            sound_label_list.append(label)
            # only first channel for now
            sound_batch_list.append(sound)

        if len(sound_batch_list) == 0:
            raise StopIteration

        win_length = FFT_SIZE // 2
        hop_length = win_length // 2
        window = torch.hann_window(win_length)
        sound_labels = torch.tensor(sound_label_list)

        # pad such that the output chunks are a multiple of the chunk size

        

        for i in range(len(sound_batch_list)):
            sound_batch_list[i] = torch.stft(sound_batch_list[i], n_fft=FFT_SIZE, win_length=win_length, 
                hop_length=hop_length, window=window, return_complex=True).abs()[0].T # first channel
            
        sound_batch = torch.nn.utils.rnn.pad_sequence(sound_batch_list).cuda(0)
        sound_batch = torch.nn.utils.rnn.pack_padded_sequence(sound_batch, [len(sound) for sound in sound_batch_list], enforce_sorted=False)

        return sound_batch, sound_labels.cuda(0)
            
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
