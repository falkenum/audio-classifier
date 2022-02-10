import pickle
import torch
import torchaudio
import os
import pathlib
from common import AC_ANALYSIS_PATH, TAG_BY_FEATURE_PATH, TAGS_PATH, AudioClassDataset, SOUNDS_DIR, FFT_SIZE, DATA_PATH
dataset = AudioClassDataset()
with open(TAGS_PATH, "rb") as f:
    tags = pickle.load(f)

tag_to_feature = {}
tag_by_feature = []

for id in tags:
    for tag in tags[id]:
        tag_to_feature[tag] = None

for i, tag in enumerate(tag_to_feature.keys()):
    tag_to_feature[tag] = i
    tag_by_feature.append(tag)

with open(TAG_BY_FEATURE_PATH, "wb") as f:
    pickle.dump(tag_by_feature, f)

with open(AC_ANALYSIS_PATH, "rb") as f:
    ac_analysis = pickle.load(f)

num_out_features = len(tag_to_feature)

if os.path.exists(DATA_PATH):
    with open(DATA_PATH, "rb") as f:
        dataset = pickle.load(f)

spectrogram = torchaudio.transforms.Spectrogram(n_fft=FFT_SIZE)
to_db = torchaudio.transforms.AmplitudeToDB()

for filepath in pathlib.Path(SOUNDS_DIR).iterdir():
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

with open(DATA_PATH, "wb") as f:
    pickle.dump(dataset, f)
