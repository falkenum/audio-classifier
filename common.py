from numpy import repeat
from torch.utils.data import Dataset
from collections import deque
import torch
from torch import nn, relu
from torch.nn import functional
from torch.utils.data import IterableDataset
import torchaudio
import os
from db import AudioDatabase
from noisereduce.noisereduce import reduce_noise
import sys
from scipy.signal import get_window
from pathlib2 import Path
import random

# SAMPLE_RATE = 44100
PREFIX_DIR = os.path.dirname(__file__)
PICKLE_DIR = f"{PREFIX_DIR}/pickle/"
CAT_DOG_SOUNDS_DIR = f"{PREFIX_DIR}/cat-dog-sounds/"
MUSIC_SOUNDS_DIR = f"{PREFIX_DIR}/music-sounds/"
BIRD_SOUNDS_DIR = f"{PREFIX_DIR}/bird-sounds/"
SAMPLE_RATE = 44100
FFT_SIZE = 512
NUM_INPUT_FEATURES = FFT_SIZE//2+1
# NUM_MELS = 2048
FREESOUND_AUTH_PATH = f"{PREFIX_DIR}/freesound_auth.json"
MODEL_PATH = f"{PICKLE_DIR}model.pickle"

if not os.path.exists(PICKLE_DIR):
    os.makedirs(PICKLE_DIR)

class ConvLSTMCell(nn.Module):

    def __init__(self, in_channels, out_channels, frame_height): 

        super(ConvLSTMCell, self).__init__()  

        self.activation = torch.tanh 
        
        # Idea adapted from https://github.com/ndrplz/ConvLSTM_pytorch
        self.conv = nn.Conv1d(
            in_channels=in_channels + out_channels, 
            out_channels=4 * out_channels, 
            kernel_size=5, 
            padding="same")           

        # Initialize weights for Hadamard Products
        self.W_ci = nn.Parameter(torch.Tensor(out_channels, frame_height).cuda(0))
        self.W_co = nn.Parameter(torch.Tensor(out_channels, frame_height).cuda(0))
        self.W_cf = nn.Parameter(torch.Tensor(out_channels, frame_height).cuda(0))

    def forward(self, X, H_prev, C_prev):

        # Idea adapted from https://github.com/ndrplz/ConvLSTM_pytorch
        conv_output = self.conv(torch.cat([X, H_prev], dim=1))

        # Idea adapted from https://github.com/ndrplz/ConvLSTM_pytorch
        i_conv, f_conv, C_conv, o_conv = torch.chunk(conv_output, chunks=4, dim=1)

        input_gate = torch.sigmoid(i_conv + self.W_ci * C_prev )
        forget_gate = torch.sigmoid(f_conv + self.W_cf * C_prev )

        # Current Cell output
        C = forget_gate*C_prev + input_gate * self.activation(C_conv)

        output_gate = torch.sigmoid(o_conv + self.W_co * C )

        # Current Hidden State
        H = output_gate * self.activation(C)

        return H, C

class ConvLSTM(nn.Module):

    def __init__(self, in_channels, out_channels, frame_height):

        super(ConvLSTM, self).__init__()

        self.out_channels = out_channels

        # We will unroll this over time steps
        self.convlstm = ConvLSTMCell(in_channels, out_channels, frame_height)
        self.layers = nn.Sequential(
            nn.Conv1d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding='same'),
            nn.MaxPool1d(5),
            nn.Conv1d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding='same'),
            nn.MaxPool1d(8),
            nn.Flatten(1, 2),
        )

    def forward(self, X):

        # X is a frame sequence (batch_size, num_channels, seq_len, height, width)

        # Get the dimensions
        batch_size, _, seq_len, height = X.size()

        # Initialize Hidden State
        H = torch.zeros(batch_size, self.out_channels, height).cuda(0)

        # Initialize Cell Input
        C = torch.zeros(batch_size,self.out_channels, height).cuda(0)

        # Unroll over time steps
        for time_step in range(seq_len):

            H, C = self.convlstm(X[:,:,time_step], H, C)

        return self.layers(H)

class ConvModel(torch.nn.Module):
    def __init__(self, output_labels) -> None:
        super().__init__()
        self.conv_chunk_width = 64

        self.layers = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=5, padding='same'),
            nn.ReLU(),
            nn.BatchNorm2d(8),
            nn.MaxPool2d((2, 4), ceil_mode=True), # batchx1x128x128
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.MaxPool2d((5, 4)), #batchx16x8x8
            nn.Conv2d(in_channels=16, out_channels=8, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.BatchNorm2d(8),
            nn.MaxPool2d((1, 2)), #batchx32x2x2
            nn.Flatten(1, 3), #batchx128
            nn.Linear(in_features=80, out_features=output_labels),
        )

    def forward(self, chunk):
        return self.layers(chunk)

class LSTMModel(torch.nn.Module):
    def __init__(self, output_labels) -> None:
        super().__init__()
        self.lstm_hidden_size = 128
        self.lstm_input_channels = FFT_SIZE//2 + 1
        self.chunk_width = 512

        # self.lstm = nn.LSTMCell(input_size=self.lstm_input_channels, hidden_size=self.lstm_hidden_size)
        self.lstm = nn.LSTM(input_size=self.lstm_input_channels, hidden_size=self.lstm_hidden_size)

        self.final_layers = nn.Sequential(
            nn.ReLU(),
            nn.Linear(in_features=self.lstm_hidden_size, out_features=output_labels),
        )

    def forward(self, sound_batch):
        # batch_size = len(sound_batch)

        # hn = torch.zeros((batch_size, self.lstm_hidden_size)).cuda(0)
        # cn = torch.zeros((batch_size, self.lstm_hidden_size)).cuda(0)
        output, (hn, cn) = self.lstm(sound_batch)

        return self.final_layers(hn.view(-1, self.lstm_hidden_size))

class AudioDataset(IterableDataset):
    def __init__(self, sounds_dirname, num_sounds, shuffle = False, sorted = False, max_classes=2) -> None:
        super().__init__()
        self.db = AudioDatabase()
        self.labelled_filepaths = []
        # query_result = list(self.db.get_bird_sounds(limit=self.num_sounds, shuffle=True))
        sounds_dir = Path(os.path.join(os.path.dirname(__file__), sounds_dirname))
        label_names = []
        complete_labelled_filepaths = []

        label_dirs = list(sounds_dir.iterdir())
        if shuffle:
            random.shuffle(label_dirs)
        for label_dir in label_dirs[:max_classes]:
            label_names.append(label_dir.name)
            for audio_file in label_dir.iterdir():
                complete_labelled_filepaths.append((label_dir.name, str(audio_file)))


        self.label_to_feature = {label: idx for idx, label in enumerate(label_names)}

        if shuffle:
            random.shuffle(complete_labelled_filepaths)

        for label, filepath in complete_labelled_filepaths[:num_sounds]:
            label = torch.tensor(self.label_to_feature[label])
            self.labelled_filepaths.append((label, filepath))
        
        def sort_key(sound):
            label, filepath = sound
            sound_data, fs = torchaudio.load(filepath)

            return sound_data.shape[1]
        
        if sorted:
            self.labelled_filepaths.sort(key=sort_key, reverse=True)
    
    def num_output_features(self):
        return len(self.label_to_feature)

    def __getitem__(self, idx):
        label, filepath = self.labelled_filepaths[idx]
        sound, fs = torchaudio.load(filepath)

        resampler = torchaudio.transforms.Resample(fs, SAMPLE_RATE)
        sound = resampler(sound)

        return sound, label

    def __len__(self):
        return len(self.labelled_filepaths)
