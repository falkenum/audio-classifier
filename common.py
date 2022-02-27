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
SAMPLE_RATE = 32000
# FFT_SIZE = 1024
# NUM_INPUT_FEATURES = FFT_SIZE//2+1
# NUM_MELS = 2048
FREESOUND_AUTH_PATH = f"{PREFIX_DIR}/freesound_auth.json"
MODEL_PATH = f"{PICKLE_DIR}model.pickle"

if not os.path.exists(PICKLE_DIR):
    os.makedirs(PICKLE_DIR)

class ConvModel(torch.nn.Module):
    def __init__(self, output_labels) -> None:
        super().__init__()

        self.layers = nn.Sequential(
            torchaudio.transforms.MelSpectrogram(n_mels=40),
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=5, padding='same'),
            nn.ReLU(),
            nn.BatchNorm2d(8),
            nn.AdaptiveMaxPool2d((200, 10000)),
            nn.Conv2d(in_channels=8, out_channels=8, kernel_size=5, padding='same'),
            nn.ReLU(),
            nn.BatchNorm2d(8),
            nn.AdaptiveMaxPool2d((100, 5000)),
            nn.Conv2d(in_channels=8, out_channels=8, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.BatchNorm2d(8),
            nn.AdaptiveMaxPool2d((20, 500)),
            nn.Conv2d(in_channels=8, out_channels=4, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.BatchNorm2d(4),
            nn.AdaptiveMaxPool2d((5, 100)),
            nn.Flatten(1, 3), #batchx128
            nn.Linear(in_features=2000, out_features=output_labels),
        )

    def forward(self, chunk):
        return self.layers(chunk)

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

        return sound.cuda(), label.cuda()

    def __len__(self):
        return len(self.labelled_filepaths)
