from numpy import repeat
from torch.utils.data import Dataset
from collections import deque
import torch
from torch import nn
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
# NUM_INPUT_FEATURES = FFT_SIZE//2+1
# NUM_MELS = 2048
FREESOUND_AUTH_PATH = f"{PREFIX_DIR}/freesound_auth.json"
DATA_PATH = f"{PICKLE_DIR}data.pickle"
# TAGS_PATH = f"{PICKLE_DIR}tags.pickle"
# TAG_BY_FEATURE_PATH = f"{PICKLE_DIR}tag_by_feature.pickle"
MODEL_PATH = f"{PICKLE_DIR}model.pickle"
# AC_ANALYSIS_PATH = f"{PICKLE_DIR}ac_analysis.pickle"

if not os.path.exists(PICKLE_DIR):
    os.makedirs(PICKLE_DIR)

# if not os.path.exists(MUSIC_SOUNDS_DIR):
#     os.makedirs(MUSIC_SOUNDS_DIR)

# def load_wav(id):
#     return torchaudio.load(f"{MUSIC_SOUNDS_DIR}/{id}.wav")

# sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../sms-tools/software/models/'))
# import sineModel as SM

# def sm(x, fs, window='blackman', M=257, N=512, t=-65):
#     # read input sound
#     # compute analysis window
#     w = get_window(window, M)
#     return (torch.from_numpy(SM.sineModel(x.squeeze().numpy(), fs, w, N, t))[None, :]).type(torch.float32)

class AudioModel(torch.nn.Module):
    def __init__(self, n_input, n_output) -> None:
        super().__init__()
        # self.lstm = nn.LSTM(input_size=1, hidden_size=64, num_layers=1)
        self.hidden_size = 32
        self.n_input = n_input

        self.lstm = nn.LSTMCell(input_size=n_input, hidden_size=self.hidden_size)
        self.layers = nn.Sequential(
            nn.ReLU(),
            nn.Linear(in_features=self.hidden_size, out_features=n_output),
            nn.LogSoftmax(dim=1)
        )


    def forward(self, spec_batch_list):
        # sound_list.sort(key=lambda sound: len(sound), reverse=True)

        hn = torch.zeros((len(spec_batch_list), self.hidden_size)).cuda(0)
        cn = torch.zeros((len(spec_batch_list), self.hidden_size)).cuda(0)
        specs = nn.utils.rnn.pad_sequence(spec_batch_list, padding_value=-100.0, batch_first=True).cuda(0)
        # packed_sounds = nn.utils.rnn.pack_padded_sequence(sounds, [len(sound) for sound in sound_list], enforce_sorted=True).cuda(0)

        for hop in specs.split(1, dim=1):
            hn, cn = self.lstm(hop.view(-1, self.n_input), (hn, cn))
        
        return self.layers(hn)


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

    def num_input_features(self):
        return FFT_SIZE//2+1
    
    def num_output_features(self):
        return len(self.label_to_feature)

    def __iter__(self):
        self.sound_queue = deque()
        self.sound_queue.extend(self.labelled_filepaths)
        return self

    def __next__(self):
        if len(self.sound_queue) == 0:
            raise StopIteration

        spec_batch_list = []
        label_batch = []

        # TODO rename this
        num_batches = 0

        while num_batches < self.batch_size and len(self.sound_queue) > 0:
            label, filepath = self.sound_queue.pop()
            sound, fs = torchaudio.load(filepath)

            resampler = torchaudio.transforms.Resample(fs, SAMPLE_RATE)
            # only first channel for now
            sound = sound[0:1]
            spec = self.to_db(self.spectrogram(resampler(sound)))
            
            spec_batch_list.append(spec[0].T)
            label_batch.append(label)

            num_batches += 1
        
        # print(max_len)
        
        labels = torch.tensor(label_batch).cuda(0)

        return spec_batch_list, labels
    