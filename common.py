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
import sys
from scipy.signal import get_window
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../sms-tools/software/models/'))
import sineModel as SM

# SAMPLE_RATE = 44100
PREFIX_DIR = os.path.dirname(__file__)
PICKLE_DIR = f"{PREFIX_DIR}/pickle/"
MUSIC_SOUNDS_DIR = f"{PREFIX_DIR}/music-sounds/"
BIRD_SOUNDS_DIR = f"{PREFIX_DIR}/bird-sounds/"
FFT_SIZE = 1024
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

def load_wav(id):
    return torchaudio.load(f"{MUSIC_SOUNDS_DIR}/{id}.wav")

class AudioClassifierModule(torch.nn.Module):
    def __init__(self, n_input, n_output, chunk_size) -> None:
        super().__init__()
        n_channel = 32
        assert(chunk_size % 800 == 0)
        self.layers = torch.nn.Sequential(
            nn.Conv1d(in_channels=n_input, out_channels=n_channel, stride=10, kernel_size=1001, padding=500),
            nn.ReLU(),
            nn.BatchNorm1d(n_channel),
            # nn.MaxPool1d(5),
            nn.Conv1d(in_channels=n_channel, out_channels=n_channel//2, stride=5, kernel_size=101, padding=50),
            nn.ReLU(),
            nn.BatchNorm1d(n_channel//2),
            # nn.MaxPool1d(4),
            nn.Conv1d(in_channels=n_channel//2, out_channels=n_channel//4, stride=4, kernel_size=21, padding=10),
            nn.ReLU(),
            # nn.MaxPool1d(10),
            nn.Conv1d(in_channels=n_channel//4, out_channels=1, stride=4, kernel_size=5, padding=2),
            nn.ReLU(),
            # nn.MaxPool1d(4),
            nn.Linear(in_features=chunk_size//800, out_features=n_output),
            nn.Flatten(1, 2),
        )
        # n_channel = n_input
        # assert(chunk_size % 80 == 0)
        # self.layers = torch.nn.Sequential(
        #     nn.Conv1d(in_channels=n_input, out_channels=n_channel//2, stride=1, kernel_size=25, padding='same'),
        #     nn.ReLU(),
        #     nn.BatchNorm1d(n_channel//2),
        #     nn.MaxPool1d(5),
        #     nn.Conv1d(in_channels=n_channel//2, out_channels=n_channel//4, stride=1, kernel_size=15, padding='same'),
        #     nn.ReLU(),
        #     nn.BatchNorm1d(n_channel//4),
        #     nn.MaxPool1d(4),
        #     nn.Conv1d(in_channels=n_channel//4, out_channels=n_channel//8, stride=1, kernel_size=5, padding='same'),
        #     nn.ReLU(),
        #     nn.Conv1d(in_channels=n_channel//8, out_channels=1, stride=1, kernel_size=5, padding='same'),
        #     nn.ReLU(),
        #     nn.MaxPool1d(4),
        #     nn.Linear(in_features=chunk_size//80, out_features=n_output),
        #     nn.Flatten(1, 2),
        # )

    def forward(self, x):
        # print(x.shape)
        # for layer in self.layers:
        #     x = layer(x)
        #     print(x.shape, "after layer", layer)
        # return x
        
        return self.layers(x)

class CatDogDataset(IterableDataset):
    def __init__(self, num_sounds, chunk_size, shuffle = False) -> None:
        super().__init__()
        self.num_sounds = num_sounds
        self.shuffle = shuffle
        self.chunk_size = chunk_size
        self.db = AudioDatabase()
        self.data_queue = None
        self.next_label = None
        self.sound_queue = None
        self.sound_list = []
        query_result = list(self.db.get_catdog_sounds(limit=self.num_sounds, shuffle=True))
        # self.spectrogram = torchaudio.transforms.Spectrogram(FFT_SIZE, FFT_SIZE//2+1, FFT_SIZE//4)
        # self.to_db = torchaudio.transforms.AmplitudeToDB()

        types_list = ["cat", "dog"]
        self.num_output_labels = 2

        self.type_to_feature = {type: idx for idx, type in enumerate(types_list)}

        for filename, animal_type in query_result:
            label = torch.tensor(self.type_to_feature[animal_type])
            self.sound_list.append((filename, label))
    
    def __iter__(self):
        self.data_queue = deque()
        self.sound_queue = deque()
        self.sound_queue.extend(self.sound_list)
        return self

    def __next__(self):
        if len(self.data_queue) == 0:
            while True:
                if len(self.sound_queue) == 0:
                    raise StopIteration
                sound_file, label = self.sound_queue.pop()
                self.next_label = label
                sound, fs = torchaudio.load(sound_file)
                assert(fs == 44100)
                assert(sound.shape[0] == 1)

                offset = 0
                while offset + self.chunk_size < sound.shape[1]:
                    new_chunk = sound[:, offset:offset+self.chunk_size]
                    if new_chunk.shape[1] != self.chunk_size:
                        break
                    self.data_queue.append(new_chunk)
                    offset += self.chunk_size
                if len(self.data_queue) > 0:
                    break
            

        data = self.data_queue.pop()
        return data.cuda(0), self.next_label.cuda(0)
    
    def __len__(self):
        return len(self.id_queue)

class BirdsDataset(IterableDataset):
    def __init__(self, num_sounds, chunk_size, shuffle = False) -> None:
        super().__init__()
        self.num_sounds = num_sounds
        self.shuffle = shuffle
        self.chunk_size = chunk_size
        self.db = AudioDatabase()
        self.data_queue = None
        self.next_label = None
        self.sound_queue = None
        self.sound_list = []
        query_result = list(self.db.get_bird_sounds(limit=self.num_sounds, shuffle=True))
        # self.spectrogram = torchaudio.transforms.MelSpectrogram(sample_rate=32000, n_fft=FFT_SIZE, win_length=FFT_SIZE//2+1, hop_length=FFT_SIZE//4)
        self.spectrogram = torchaudio.transforms.Spectrogram(FFT_SIZE, FFT_SIZE//2+1, FFT_SIZE//4, power = 1)
        self.ispectrogram = torchaudio.transforms.InverseSpectrogram(FFT_SIZE, FFT_SIZE//2+1, FFT_SIZE//4)
        # self.to_db = torchaudio.transforms.AmplitudeToDB()

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
        self.data_queue = deque()
        self.sound_queue = deque()
        self.sound_queue.extend(self.sound_list)
        return self

    def __next__(self):
        if len(self.data_queue) == 0:
            while True:
                if len(self.sound_queue) == 0:
                    raise StopIteration
                sound_file, label = self.sound_queue.pop()
                self.next_label = label
                sound, fs = torchaudio.load(f"bird-sounds-foreground/{sound_file}")
                assert(fs == 32000)
                assert(sound.shape[0] == 1)

                # sound = sound.squeeze()
                # compute analysis window
                # w = get_window("blackman", 513)
                
                # sound = SM.sineModel(sound.numpy(), fs, w, 1024, -60)
                # sound = torch.FloatTensor(sound)[None, :]

                # torchaudio.save(f"{BIRD_SOUNDS_DIR}/filtered_{sound_file}", sound, fs)

                offset = 0
                while offset + self.chunk_size < sound.shape[1]:
                    new_chunk = sound[:, offset:offset+self.chunk_size]
                    if new_chunk.shape[1] != self.chunk_size:
                        break
                    self.data_queue.append(new_chunk)
                    offset += self.chunk_size
                if len(self.data_queue) > 0:
                    break
            

        data = self.data_queue.pop()
        return data.cuda(0), self.next_label.cuda(0)
    
class MusicNotesDataset(IterableDataset):
    def __init__(self, num_sounds, chunk_size, shuffle = False) -> None:
        super().__init__()
        self.num_sounds = num_sounds
        self.shuffle = shuffle
        self.chunk_size = chunk_size
        self.db = AudioDatabase()
        self.data_queue = None
        self.next_label = None
        self.sound_queue = None
        self.sound_list = []
        query_result = list(self.db.get_notes_sounds(limit=self.num_sounds, shuffle=True))

        note_types = set()
        for filename, note_type in query_result:
            note_types.add(note_type)
        
        note_types_list = list(note_types)
        self.num_output_labels = self.db.get_num_note_types()

        self.note_type_to_feature = {note_type: idx for idx, note_type in enumerate(note_types_list)}

        for filename, note_type in query_result:
            label = torch.tensor(self.note_type_to_feature[note_type])
            self.sound_list.append((filename, label))
    
    def __iter__(self):
        self.data_queue = deque()
        self.sound_queue = deque()
        self.sound_queue.extend(self.sound_list)
        return self

    def __next__(self):
        if len(self.data_queue) == 0:
            if len(self.sound_queue) == 0:
                raise StopIteration
            while True:
                sound_file, label = self.sound_queue.pop()
                self.next_label = label
                sound, fs = torchaudio.load(sound_file)
                assert(fs == 44100)
                assert(sound.shape[0] == 1)

                offset = 0
                while offset + self.chunk_size < sound.shape[1]:
                    new_chunk = sound[:, offset:offset+self.chunk_size]
                    if new_chunk.shape[1] != self.chunk_size:
                        break
                    self.data_queue.append(new_chunk)
                    offset += self.chunk_size
                if len(self.data_queue) > 0:
                    break
            

        data = self.data_queue.pop()
        return data.cuda(0), self.next_label.cuda(0)
    
    def __len__(self):
        return len(self.id_queue)