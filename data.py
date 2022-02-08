import pickle
import torch
import torchaudio
import os
from common import AudioClassDataset, SOUNDSDIR, TRANSFORM, CLASSES_MAP_REV
dataset = AudioClassDataset()

datapath = "./data.pickle"
if os.path.exists(datapath):
    with open(datapath, "rb") as f:
        dataset = pickle.load(f)

for class_path in SOUNDSDIR.iterdir():
    for file_path in class_path.iterdir():
        if file_path.stem not in dataset.id_set:
            waveform, samplerate = torchaudio.load(file_path)
            if len(waveform) > 1:
                new_waveform = torch.zeros(1, waveform.shape[1])
                for i in range(waveform.shape[1]):
                    new_waveform[0, i] = torch.mean(waveform[:, i])
                waveform = new_waveform

            spectrogram = TRANSFORM(waveform)[0]
            label = CLASSES_MAP_REV[class_path.name]
            id = int(file_path.stem)
            dataset.add_samples(spectrogram, label, id)
            print("added ", class_path.name, file_path.name)

with open(datapath, "wb") as f:
    pickle.dump(dataset, f)