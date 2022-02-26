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
FFT_SIZE = 1024
NUM_INPUT_FEATURES = FFT_SIZE//2+1
# NUM_MELS = 2048
FREESOUND_AUTH_PATH = f"{PREFIX_DIR}/freesound_auth.json"
DATA_PATH = f"{PICKLE_DIR}data.pickle"
# TAGS_PATH = f"{PICKLE_DIR}tags.pickle"
# TAG_BY_FEATURE_PATH = f"{PICKLE_DIR}tag_by_feature.pickle"
MODEL_PATH = f"{PICKLE_DIR}model.pickle"
# AC_ANALYSIS_PATH = f"{PICKLE_DIR}ac_analysis.pickle"

if not os.path.exists(PICKLE_DIR):
    os.makedirs(PICKLE_DIR)

class ConvModel(torch.nn.Module):
    def __init__(self, output_labels) -> None:
        super().__init__()
        self.conv_chunk_width = 64

        self.layers = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=5, padding='same'),
            nn.ReLU(),
            nn.BatchNorm2d(8),
            nn.MaxPool2d((8, 8), ceil_mode=True), # batchx1x128x128
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.MaxPool2d((8, 4)), #batchx16x8x8
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.MaxPool2d((4, 1)), #batchx32x2x2
            nn.Flatten(1, 3), #batchx128
            nn.Linear(in_features=64, out_features=output_labels),
        )

    def forward(self, chunk):
        return self.layers(chunk)

class ConvLSTMModel(torch.nn.Module):
    def __init__(self, output_labels) -> None:
        super().__init__()
        self.lstm_hidden_size = 64
        self.lstm_input_channels = 128

        self.conv_chunk_width = 1024 # about 6 seconds per chunk

        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=5, padding='same'),
            nn.ReLU(),
            nn.BatchNorm2d(8),
            nn.MaxPool2d((8, 8), ceil_mode=True), # batchx1x128x128
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.MaxPool2d((8, 8)), #batchx16x8x8
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d((8, 8)), #batchx32x2x2
            nn.Flatten(1, 3), #batchx128
            # nn.Linear(in_features=128, out_features=output_labels),
        )

        self.lstm = nn.LSTMCell(input_size=self.lstm_input_channels, hidden_size=self.lstm_hidden_size)

        self.final_layers = nn.Sequential(
            nn.ReLU(),
            nn.Linear(in_features=self.lstm_hidden_size, out_features=output_labels),
        )

        self.hn = None
        self.cn = None


    def forward(self, chunk):
        batch_size = len(chunk)

        # add channel dimension back in
        chunk = chunk[:, None, :, :]

        self.hn = torch.zeros((batch_size, self.lstm_hidden_size)).cuda(0)
        self.cn = torch.zeros((batch_size, self.lstm_hidden_size)).cuda(0)

        # last dimension will be short probably
        self.hn, self.cn = self.lstm(self.conv_layers(chunk), (self.hn, self.cn))

        return self.final_layers(self.hn)

class AudioDataset(IterableDataset):
    def __init__(self, sounds_dirname, num_sounds, shuffle = False, sorted = False) -> None:
        super().__init__()
        self.db = AudioDatabase()
        self.labelled_filepaths = []
        # query_result = list(self.db.get_bird_sounds(limit=self.num_sounds, shuffle=True))
        sounds_dir = Path(os.path.join(os.path.dirname(__file__), sounds_dirname))
        label_names = []
        complete_labelled_filepaths = []

        for label_dir in sounds_dir.iterdir():
            if label_dir.is_dir():
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
