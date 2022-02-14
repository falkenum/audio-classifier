from common import *
from torch.utils.data import TensorDataset
from db import AudioDatabase
import pickle
import os
import gc
import psutil
import faulthandler


# if os.path.exists(DATA_PATH):
#     with open(DATA_PATH, "rb") as f:
#         dataset = pickle.load(f)
# else:
#     dataset = TensorDataset()


db = AudioDatabase()
spectrogram = torchaudio.transforms.Spectrogram(n_fft=FFT_SIZE)

query_result = list(db.get_sounds(limit=1200))
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


for idx, (sound_id, sound_tags) in enumerate(query_result):
    # if sound_id not in dataset.id_set:
    # print(idx, sound_id, psutil.virtual_memory()[1] / 2**30, "GB")
    print(idx, sound_id)
    raw_sound, fs = load_wav(sound_id)

    resampler = torchaudio.transforms.Resample(fs, SAMPLE_RATE)
    # only using first channel for now
    resampled_sound = resampler(raw_sound[0])
    spec = spectrogram(resampled_sound)

    label = torch.zeros(NUM_FEATURE_LABELS, dtype=int)
    for sound_tag in sound_tags:
        if tag_to_feature.get(sound_tag) is not None:
            label[tag_to_feature[sound_tag]] = 1

    for col in range(spec.shape[1]):
        db.insert_sample(int(sound_id), col, spec[:, col].tolist(), label.tolist())
