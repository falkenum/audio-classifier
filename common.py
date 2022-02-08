from torch.utils.data import Dataset
import torch
import torchaudio
import pathlib

CLASSES_LIST = ["bass", "percussion", "guitar", "trumpet", "violin"]
CLASSES_MAP = {i: CLASSES_LIST[i] for i in range(len(CLASSES_LIST))}
CLASSES_MAP_REV = {v: k for k, v in CLASSES_MAP.items()}
SAMPLERATE = 48000

DATAPATH = "./data.pickle"
SOUNDSDIR = pathlib.Path("./sounds")
FFT_SIZE = 1024
FEATURE_SIZE = FFT_SIZE // 2 + 1

class AudioClassDataset(Dataset):
    def __init__(self):
        super().__init__()
        self.data = torch.Tensor(FEATURE_SIZE, 0)
        self.labels = []
        self.id_set = set()
    
    def __getitem__(self, index):
        return self.data[:, index], self.labels[index]

    def __len__(self):
        return self.data.shape[1]
    
    def add_samples(self, data, label, id):
        self.data = torch.cat((self.data, data), 1)

        self.labels.extend([label for i in range(data.shape[1])])
        self.id_set.add(id)
