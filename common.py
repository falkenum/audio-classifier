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
FFT_SIZE = 2048
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

class AudioModel(torch.nn.Module):
    def __init__(self, output_labels) -> None:
        super().__init__()
        self.lstm_hidden_size = 64
        self.conv_chunk_width = 8 # 8 hops of len 512 per hop, corresponds to 4096 raw samples per chunk
        # self.conv_chunk_height = NUM_INPUT_FEATURES - 1 # make power of 2

        self.lstm_input_channels = 32

        # going from 1 by conv_chunk_height by conv_chunk_width to lstm_input_channels by 1
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, padding='same'),
            nn.MaxPool2d((2, 2)), # 32x512x4
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, padding='same'),
            nn.MaxPool2d((2, 2)), # 16x256x2
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=4, kernel_size=3, padding='same'),
            nn.MaxPool2d((2, 2)), # 4x128x1
            nn.ReLU(),
            nn.Conv2d(in_channels=4, out_channels=1, kernel_size=3, padding='same'),
            nn.MaxPool2d((4, 1)), # 1x32x1
            nn.ReLU(),
            nn.Flatten(1, 3), # 32x1
        )

        self.lstm = nn.LSTMCell(input_size=self.lstm_input_channels, hidden_size=self.lstm_hidden_size)
        self.final_layers = nn.Sequential(
            nn.ReLU(),
            nn.Linear(in_features=self.lstm_hidden_size, out_features=output_labels),
            nn.LogSoftmax(dim=1)
        )


    def forward(self, specs):
        batch_size = len(specs)

        chunks = torch.split(specs, self.conv_chunk_width, dim=3)
        hn = torch.zeros((batch_size, self.lstm_hidden_size)).cuda(0)
        cn = torch.zeros((batch_size, self.lstm_hidden_size)).cuda(0)

        post_conv_chunks = []

        # last dimension will be short probably
        for chunk in chunks[:-1]:
            # indexing to form power of 2 height
            chunk = chunk[:, :, 1:, :].cuda(0)
            post_conv_chunks.append(self.conv_layers(chunk)) # batch_sizex32x1
        
        # post_conv = torch.stack(post_conv_chunks, dim=2)

        for chunk in post_conv_chunks:
            hn, cn = self.lstm(chunk, (hn, cn))
        
        return self.final_layers(hn)

class AudioDataset(IterableDataset):
    def __init__(self, sounds_dirname, num_sounds, batch_size, shuffle = False) -> None:
        super().__init__()
        self.batch_size = batch_size
        self.db = AudioDatabase()
        self.sound_queue = None
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


        self.spectrogram = torchaudio.transforms.Spectrogram(FFT_SIZE, FFT_SIZE//2+1, FFT_SIZE//4, power = 2)
        # self.spectrogram = torchaudio.transforms.MelSpectrogram(sample_rate=32000, n_mels=NUM_MELS, n_fft=FFT_SIZE, win_length=FFT_SIZE//2+1, hop_length=FFT_SIZE//8, power = 2)
        # self.mfcc = torchaudio.transforms.MFCC(sample_rate=32000)
        self.to_db = torchaudio.transforms.AmplitudeToDB()
        self.label_to_feature = {label: idx for idx, label in enumerate(label_names)}

        if shuffle:
            random.shuffle(complete_labelled_filepaths)

        for label, filepath in complete_labelled_filepaths[:num_sounds]:
            label = torch.tensor(self.label_to_feature[label])
            self.labelled_filepaths.append((label, filepath))
        
        # arrange chosen sounds by length of the sound to help with padding later
        def sort_key(sound):
            label, filepath = sound
            sound_data, fs = torchaudio.load(filepath)

            # length of first channel (should be the same for all channels)
            return len(sound_data[0])
        
        # sort such that the longest files are first
        self.labelled_filepaths.sort(key=sort_key)
    
    def num_output_features(self):
        return len(self.label_to_feature)

    def __iter__(self):
        self.sound_queue = deque()
        self.sound_queue.extend(self.labelled_filepaths)
        return self

    def __next__(self):
        if len(self.sound_queue) == 0:
            raise StopIteration

        sound_batch = []
        label_batch = []

        sound_idx = 0

        while sound_idx < self.batch_size and len(self.sound_queue) > 0:
            label, filepath = self.sound_queue.pop()
            sound, fs = torchaudio.load(filepath)

            resampler = torchaudio.transforms.Resample(fs, SAMPLE_RATE)
            sound = resampler(sound)
            # only first channel for now
            sound_batch.append(sound[0])
            label_batch.append(label)

            sound_idx += 1

        padded_sounds = nn.utils.rnn.pad_sequence(sound_batch, padding_value=0.0, batch_first=True)
        specs = self.to_db(self.spectrogram(padded_sounds))[:, None, :, :]# add channel dimension back in
        # packed_sounds = nn.utils.rnn.pack_padded_sequence(sounds, [len(sound) for sound in sound_list], enforce_sorted=True).cuda(0)
        
        labels = torch.tensor(label_batch).cuda(0)

        return specs, labels
    