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

# SAMPLE_RATE = 44100
PREFIX_DIR = os.path.dirname(__file__)
PICKLE_DIR = f"{PREFIX_DIR}/pickle/"
MUSIC_SOUNDS_DIR = f"{PREFIX_DIR}/music-sounds/"
BIRD_SOUNDS_DIR = f"{PREFIX_DIR}/bird-sounds/"
FFT_SIZE = 4096
NUM_MELS = 128
FREESOUND_AUTH_PATH = f"{PREFIX_DIR}/freesound_auth.json"
DATA_PATH = f"{PICKLE_DIR}data.pickle"
# TAGS_PATH = f"{PICKLE_DIR}tags.pickle"
# TAG_BY_FEATURE_PATH = f"{PICKLE_DIR}tag_by_feature.pickle"
MODEL_PATH = f"{PICKLE_DIR}model.pickle"
# AC_ANALYSIS_PATH = f"{PICKLE_DIR}ac_analysis.pickle"

if not os.path.exists(PICKLE_DIR):
    os.makedirs(PICKLE_DIR)

if not os.path.exists(MUSIC_SOUNDS_DIR):
    os.makedirs(MUSIC_SOUNDS_DIR)

# def load_wav(id):
#     return torchaudio.load(f"{MUSIC_SOUNDS_DIR}/{id}.wav")

# sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../sms-tools/software/models/'))
# import sineModel as SM

# def sm(x, fs, window='blackman', M=257, N=512, t=-65):
#     # read input sound
#     # compute analysis window
#     w = get_window(window, M)
#     return (torch.from_numpy(SM.sineModel(x.squeeze().numpy(), fs, w, N, t))[None, :]).type(torch.float32)

class BirdModel(torch.nn.Module):
    def __init__(self, n_input, n_output) -> None:
        super().__init__()
        # self.lstm = nn.LSTM(input_size=1, hidden_size=64, num_layers=1)
        self.hidden_size = 64

        self.lstm = nn.LSTMCell(input_size=NUM_MELS, hidden_size=self.hidden_size)
        self.layers = nn.Sequential(
            nn.Linear(in_features=self.hidden_size, out_features=n_output),
        )


    def forward(self, spec_list):
        # print(x.shape)
        # for layer in self.layers:
        #     x = layer(x)
        #     print(x.shape, "after layer", layer)
        # return x
        # assert(x.shape[0] == 1)
        # assert(x.shape[1] == 1)

        # x = x.view(1, -1, 1)
        # sound_list.sort(key=lambda sound: len(sound), reverse=True)
        specs = nn.utils.rnn.pad_sequence(spec_list, padding_value=-100.0, batch_first=True).cuda(0)
        # packed_sounds = nn.utils.rnn.pack_padded_sequence(sounds, [len(sound) for sound in sound_list], enforce_sorted=True).cuda(0)
        # specs = torch.stack(tuple(spec_list), dim=0).cuda(0)

        # h = nvmlDeviceGetHandleByIndex(0)
        # info = nvmlDeviceGetMemoryInfo(h)
        # print(f'total    : {info.total}')
        # print(f'free     : {info.free}')
        # print(f'used     : {info.used}')
        # batch_size = 32000
        hn = torch.zeros((len(spec_list), self.hidden_size)).cuda(0)
        cn = torch.zeros((len(spec_list), self.hidden_size)).cuda(0)

        for hop in specs.split(1, dim=1):
            hn, cn = self.lstm(hop.view(-1, NUM_MELS), (hn, cn))
        
        return self.layers(hn)


class BirdsDataset(IterableDataset):
    def __init__(self, num_sounds, batch_size, shuffle = False) -> None:
        super().__init__()
        self.num_sounds = num_sounds
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.db = AudioDatabase()
        self.sound_queue = None
        self.sound_list = []
        query_result = list(self.db.get_bird_sounds(limit=self.num_sounds, shuffle=True))
        # self.spectrogram = torchaudio.transforms.Spectrogram(FFT_SIZE, FFT_SIZE//2+1, FFT_SIZE//8, power = 2)
        self.spectrogram = torchaudio.transforms.MelSpectrogram(sample_rate=32000, n_mels=NUM_MELS, n_fft=FFT_SIZE, win_length=FFT_SIZE//2+1, hop_length=FFT_SIZE//8, power = 2)
        # self.mfcc = torchaudio.transforms.MFCC(sample_rate=32000)
        self.to_db = torchaudio.transforms.AmplitudeToDB()

        bird_names = set()
        for bird_name, filename in query_result:
            bird_names.add(bird_name)
        
        bird_names_list = list(bird_names)
        self.num_output_labels = self.db.get_num_birds()

        self.bird_to_feature = {bird_name: idx for idx, bird_name in enumerate(bird_names_list)}

        for bird_name, filename in query_result:
            label = torch.tensor(self.bird_to_feature[bird_name])
            sound_file = f"{bird_name}/{filename}"
            self.sound_list.append((sound_file, label))
    
    def __iter__(self):
        self.sound_queue = deque()
        self.sound_queue.extend(self.sound_list)
        return self

    def __next__(self):
        if len(self.sound_queue) == 0:
            raise StopIteration

        spec_list = []
        label_list = []
        num_batches = 0

        while num_batches < self.batch_size and len(self.sound_queue) > 0:
            sound_file, label = self.sound_queue.pop()
            sound, fs = torchaudio.load(f"{BIRD_SOUNDS_DIR}/{sound_file}")

            assert(fs == 32000)
            assert(sound.shape[0] == 1)
            spec = self.to_db(self.spectrogram(sound))
            
            spec_list.append(spec[0].T.cuda(0))
            label_list.append(label)

            num_batches += 1
        
        # print(max_len)
        
        labels = torch.tensor(label_list).cuda(0)

        return spec_list, labels
    