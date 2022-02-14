from torch.utils.data import Dataset
from collections import deque
import torch
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
    def __init__(self, in_features, out_features) -> None:
        super().__init__()
        # self.layers = torch.nn.Linear(in_features, out_features)
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(in_features, in_features, device="cuda:0"),
            torch.nn.AvgPool1d(10, 10),
            torch.nn.Linear(in_features//10, out_features, device="cuda:0"),
        )

    def forward(self, x):
        return torch.sigmoid(self.layers(x))

# class AudioClassDataset(Dataset):
#     def __init__(self):
#         super().__init__()
#         self.data = []
#         self.labels = []
#         self.id_set = set()
    
#     def __getitem__(self, index):
#         if torch.cuda.is_available():
#             return self.data[index].to("cuda:0"), self.labels[index].to("cuda:0")
#         else:
#             return self.data[index], self.labels[index]

#     def __len__(self):
#         return len(self.data)
    
#     def add_sample(self, data, label, id):
#         self.data.append(data)
#         self.labels.append(label)
#         self.id_set.add(id)

class SamplesIterator:
    def __init__(self, num_sounds, shuffle) -> None:
        self.samples_queue = deque()
        self.source_id_queue = deque()
        self.db = AudioDatabase()
        sound_ids = self.db.get_sound_ids_from_samples(num_sounds, shuffle)
        self.source_id_queue.extend(sound_ids)

    def __iter__(self):
        return self

    def __next__(self):
        if len(self.samples_queue) == 0:
            while True:
                if len(self.source_id_queue) == 0:
                    raise StopIteration
                source_id = self.source_id_queue.pop()
                self.samples_queue.extend(self.db.get_samples_for_id(source_id))

                if len(self.samples_queue) != 0:
                    break

        
        x, y = self.samples_queue.pop()
        if torch.cuda.is_available():
            return torch.tensor(x, device="cuda:0", dtype=torch.float32), torch.tensor(y, device="cuda:0", dtype=torch.float32)
        else:
            return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


class SamplesDataset(IterableDataset):
    def __init__(self, num_sounds, shuffle = False) -> None:
        super().__init__()
        self.num_sounds = num_sounds
        self.shuffle = shuffle
        self.it = None
        self.db = AudioDatabase()

    def __iter__(self):
        self.it = SamplesIterator(self.num_sounds, self.shuffle)
        return self.it
    
    def __len__(self):
        return self.db.get_num_samples()
    

