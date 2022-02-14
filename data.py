from common import *
from torch.utils.data import TensorDataset
from db import AudioDatabase
import pickle
import os


# if os.path.exists(DATA_PATH):
#     with open(DATA_PATH, "rb") as f:
#         dataset = pickle.load(f)
# else:
#     dataset = TensorDataset()


db = AudioDatabase()
num_files_per_tag = 10
spectrogram = torchaudio.transforms.Spectrogram()

query_result = list(db.get_sounds(limit=50).to_records())
tag_counts = {}
for row, sound_id, sound_tags in query_result:
    for sound_tag in sound_tags:
        if sound_tag not in tag_counts.keys():
            tag_counts[sound_tag] = 0
        tag_counts[sound_tag] += 1

tag_counts = [(k, v) for k, v in tag_counts.items()]
def sort_key(elt):
    k, v = elt
    return v
tag_counts.sort(key=sort_key, reverse=True)
num_feature_tags = 10
tags = map(lambda elt: elt[0], tag_counts[:num_feature_tags])

tag_to_feature = {tag: idx for idx, tag in enumerate(tags)}


for row, sound_id, sound_tags in query_result:
    # if sound_id not in dataset.id_set:
    raw_sound, fs = load_wav(sound_id)

    resampler = torchaudio.transforms.Resample(fs, SAMPLE_RATE)
    # only using first channel for now
    resampled_sound = resampler(raw_sound[0])
    spec = spectrogram(resampled_sound)

    label = torch.zeros(num_feature_tags, dtype=int)
    for sound_tag in sound_tags:
        if tag_to_feature.get(sound_tag) is not None:
            label[tag_to_feature[sound_tag]] = 1

    for col in range(spec.shape[1]):
        # dataset.add_sample(spec[:, col], label, sound_id)
        db.insert_sample(int(sound_id), col, spec[:, col].tolist(), label.tolist())

# with open(DATA_PATH, "wb") as f:
#     pickle.dump(dataset, f)