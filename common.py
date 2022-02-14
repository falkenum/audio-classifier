from torch.utils.data import Dataset
import torch
import torchaudio
import os

SAMPLE_RATE = 44100
PREFIX_DIR = os.path.dirname(__file__)
PICKLE_DIR = f"{PREFIX_DIR}/pickle/"
SOUNDS_DIR = f"{PREFIX_DIR}/sounds/"

FREESOUND_AUTH_PATH = f"{PREFIX_DIR}/freesound_auth.json"
DATA_PATH = f"{PICKLE_DIR}data.pickle"
# TAGS_PATH = f"{PICKLE_DIR}tags.pickle"
# TAG_BY_FEATURE_PATH = f"{PICKLE_DIR}tag_by_feature.pickle"
MODEL_PATH = f"{PICKLE_DIR}model.pickle"
# AC_ANALYSIS_PATH = f"{PICKLE_DIR}ac_analysis.pickle"

FFT_SIZE = 1024

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
            torch.nn.Linear(in_features, in_features),
            torch.nn.AvgPool1d(10, 10),
            torch.nn.Linear(in_features//10, out_features),
        )

    def forward(self, x):
        return torch.sigmoid(self.layers(x))

class AudioClassDataset(Dataset):
    def __init__(self):
        super().__init__()
        self.data = []
        self.labels = []
        self.id_set = set()
    
    def __getitem__(self, index):
        if torch.cuda.is_available():
            return self.data[index].to("cuda:0"), self.labels[index].to("cuda:0")
        else:
            return self.data[index], self.labels[index]

    def __len__(self):
        return len(self.data)
    
    def add_sample(self, data, label, id):
        self.data.append(data)
        self.labels.append(label)
        self.id_set.add(id)
