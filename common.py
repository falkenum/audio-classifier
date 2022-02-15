from torch.utils.data import Dataset
from collections import deque
import torch
from torch import nn
from torch.utils.data import IterableDataset
import torchaudio
import os
from db import AudioDatabase

SAMPLE_RATE = 44100
PREFIX_DIR = os.path.dirname(__file__)
PICKLE_DIR = f"{PREFIX_DIR}/pickle/"
SOUNDS_DIR = f"{PREFIX_DIR}/sounds/"
NUM_FEATURE_LABELS = 10
FFT_SIZE = 400
NUM_FEATURE_DATA = FFT_SIZE // 2 + 1

FREESOUND_AUTH_PATH = f"{PREFIX_DIR}/freesound_auth.json"
DATA_PATH = f"{PICKLE_DIR}data.pickle"
# TAGS_PATH = f"{PICKLE_DIR}tags.pickle"
# TAG_BY_FEATURE_PATH = f"{PICKLE_DIR}tag_by_feature.pickle"
MODEL_PATH = f"{PICKLE_DIR}model.pickle"
# AC_ANALYSIS_PATH = f"{PICKLE_DIR}ac_analysis.pickle"

if not os.path.exists(PICKLE_DIR):
    os.makedirs(PICKLE_DIR)

if not os.path.exists(SOUNDS_DIR):
    os.makedirs(SOUNDS_DIR)

def load_wav(id):
    return torchaudio.load(f"{SOUNDS_DIR}/{id}.wav")

class AudioClassifierModule(torch.nn.Module):
    def __init__(self, n_input, n_output) -> None:
        super().__init__()
        n_channel = 16
        # self.layers = torch.nn.Linear(in_features, out_features)
        self.layers = torch.nn.Sequential(
            nn.Conv1d(in_channels=n_input, out_channels=n_channel, kernel_size=25, padding='same'),
            nn.ReLU(),
            nn.BatchNorm1d(n_channel),
            nn.MaxPool1d(25),
            nn.Conv1d(in_channels=n_channel, out_channels=n_channel, kernel_size=5, padding='same'),
            nn.ReLU(),
            nn.BatchNorm1d(n_channel),
            nn.MaxPool1d(40),
            nn.Conv1d(in_channels=n_channel, out_channels=1, kernel_size=5, padding='same'),
            nn.ReLU(),
            # nn.BatchNorm1d(n_channel),
            nn.MaxPool1d(10),
            nn.Linear(in_features=n_output, out_features=n_output),
        )

    def forward(self, x):
        # print(x.shape)
        # for layer in self.layers:
        #     x = layer(x)
        #     print(x.shape, "after layer", layer)
        return torch.sigmoid(self.layers(x))

class SamplesDataset(IterableDataset):
    def __init__(self, num_sounds, shuffle = False, chunk_size = 100000) -> None:
        super().__init__()
        self.num_sounds = num_sounds
        self.shuffle = shuffle
        self.chunk_size = chunk_size
        self.db = AudioDatabase()
        self.data_queue = None
        self.next_label = None
        self.sound_queue = None

    def _reinit(self):
        self.data_queue = deque()
        self.sound_queue = deque()

        query_result = list(self.db.get_sounds(limit=self.num_sounds))
        tag_counts = {}
        for sound_id, sound_tags in query_result:
            for sound_tag in sound_tags:
                if sound_tag not in tag_counts.keys():
                    tag_counts[sound_tag] = 0
                tag_counts[sound_tag] += 1

        tag_counts = [(k, v) for k, v in tag_counts.items()]
        def sort_key(elt):
            k, v = elt
            return v
        tag_counts.sort(key=sort_key, reverse=True)
        tags = map(lambda elt: elt[0], tag_counts[:NUM_FEATURE_LABELS])
        tag_to_feature = {tag: idx for idx, tag in enumerate(tags)}

        for sound_id, sound_tags in query_result:
            label = torch.zeros(NUM_FEATURE_LABELS)
            for sound_tag in sound_tags:
                if tag_to_feature.get(sound_tag) is not None:
                    label[tag_to_feature[sound_tag]] = 1
            self.sound_queue.append((sound_id, label))

    def __iter__(self):
        self._reinit()
        return self

    def __next__(self):
        if len(self.data_queue) == 0:
            if len(self.sound_queue) == 0:
                raise StopIteration
            while True:
                sound_id, label = self.sound_queue.pop()
                self.next_label = label
                raw_sound, fs = load_wav(sound_id)
                resampler = torchaudio.transforms.Resample(fs, SAMPLE_RATE)

                # only first channel for now
                resampled_sound = resampler(raw_sound)[0:1]
                offset = 0
                while offset + self.chunk_size < resampled_sound.shape[1]:
                    new_chunk = resampled_sound[:, offset:offset+self.chunk_size]
                    if new_chunk.shape[1] != self.chunk_size:
                        break
                    self.data_queue.append(new_chunk)
                    offset += self.chunk_size
                if len(self.data_queue) > 0:
                    break
            

        data = self.data_queue.pop()
        return data.cuda(0), self.next_label[None, :].cuda(0)
    
    def __len__(self):
        return len(self.id_queue)
    

