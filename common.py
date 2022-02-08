from torch.utils.data import Dataset
import torch
import torchaudio
import pathlib

SAMPLERATE = 48000
DATAPATH = "./data.pickle"
TAGSPATH = "./tags.pickle"
SOUNDSDIR = pathlib.Path("./sounds")
FFT_SIZE = 1024
FEATURE_SIZE = FFT_SIZE // 2 + 1

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
