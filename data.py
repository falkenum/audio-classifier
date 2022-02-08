import pickle
import torch
import torchaudio
import pathlib
import os
from common import AudioClassDataset

soundsdir = pathlib.Path("./sounds")
transform = torchaudio.transforms.Spectrogram()
feature_size = transform.n_fft // 2 + 1
datasets = {}

datapath = "sounds/data.pickle"
if os.path.exists(datapath):
    with open(datapath, "rb") as f:
        datasets = pickle.load(f)

for class_path in soundsdir.iterdir():
    if class_path.is_dir():
        if class_path.name in datasets:
            class_data = datasets[class_path.name].class_data
            id_set = datasets[class_path.name].id_set
        else:
            class_data = torch.Tensor(feature_size, 0)
            id_set = set()

        for file_path in class_path.iterdir():

            if file_path.stem not in id_set:
                waveform, samplerate = torchaudio.load(file_path)
                if len(waveform) > 1:
                    new_waveform = torch.zeros(1, waveform.shape[1])
                    for i in range(waveform.shape[1]):
                        new_waveform[0, i] = torch.mean(waveform[:, i])
                    waveform = new_waveform

                spectrogram = transform(waveform)[0]
                print(class_path.name, file_path.name)
                class_data = torch.cat((class_data, spectrogram), 1)
                id_set.add(file_path.stem)

        datasets[class_path.name] = AudioClassDataset(class_data, id_set)

with open(datapath, "wb") as f:
    pickle.dump(datasets, f)