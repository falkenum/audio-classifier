import pickle
import torch
import torchaudio
import os
from common import TAGSPATH, AudioClassDataset, SOUNDSDIR, FFT_SIZE, DATAPATH
dataset = AudioClassDataset()
with open(TAGSPATH, "rb") as f:
    tags = pickle.load(f)

tag_to_feature = {}

for id in tags:
    for tag in tags[id]:
        tag_to_feature[tag] = None

for i, tag in enumerate(tag_to_feature.keys()):
    tag_to_feature[tag] = i
num_out_features = len(tag_to_feature)

if os.path.exists(DATAPATH):
    with open(DATAPATH, "rb") as f:
        dataset = pickle.load(f)

spectrogram = torchaudio.transforms.Spectrogram(n_fft=FFT_SIZE)
to_db = torchaudio.transforms.AmplitudeToDB()

for filepath in SOUNDSDIR.iterdir():
    if filepath.stem not in dataset.id_set:
        waveform, samplerate = torchaudio.load(filepath)
        spec = spectrogram(waveform)
        spec_db = to_db(spec)[0]
        label = torch.zeros(num_out_features, 1)
        id = int(filepath.stem)
        for tag in tags[id]:
            label[tag_to_feature[tag], 0] = 1
        dataset.add_samples(spec_db, label, id)
        print("added", filepath.name)

with open(DATAPATH, "wb") as f:
    pickle.dump(dataset, f)