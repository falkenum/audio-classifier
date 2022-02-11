from torch.utils.data import Dataset
import torch
import torchaudio
import os

SAMPLE_RATE = 48000
PREFIX_DIR = os.path.dirname(__file__)
PICKLE_DIR = f"{PREFIX_DIR}/pickle/"
SOUNDS_DIR = f"{PREFIX_DIR}/sounds/"

FREESOUND_AUTH_PATH = f"{PREFIX_DIR}/freesound_auth.json"
DATA_PATH = f"{PICKLE_DIR}data.pickle"
TAGS_PATH = f"{PICKLE_DIR}tags.pickle"
TAG_BY_FEATURE_PATH = f"{PICKLE_DIR}tag_by_feature.pickle"
MODEL_PATH = f"{PICKLE_DIR}model.pickle"
AC_ANALYSIS_PATH = f"{PICKLE_DIR}ac_analysis.pickle"

FFT_SIZE = 1024
FEATURE_SIZE = 8

if not os.path.exists(PICKLE_DIR):
    os.makedirs(PICKLE_DIR)

if not os.path.exists(SOUNDS_DIR):
    os.makedirs(SOUNDS_DIR)

class AudioClassifierModule(torch.nn.Module):
    def __init__(self, out_features) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(FEATURE_SIZE, out_features)

    def forward(self, x):
        return self.linear(x)

class AudioClassDataset(Dataset):
    def __init__(self):
        super().__init__()
        self.data = torch.Tensor(FEATURE_SIZE, 0)
        self.labels = None
        self.id_set = set()
    
    def __getitem__(self, index):
        return self.data[:, index], self.labels[:, index]

    def __len__(self):
        return self.data.shape[1]
    
    def add_samples(self, data, label, id):
        self.data = torch.cat((self.data, data), 1)

        for i in range(data.shape[1]):
            if self.labels is not None:
                self.labels = torch.cat((self.labels, label), 1)
            else:
                self.labels = label


        self.id_set.add(id)
