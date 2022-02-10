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

# ac_features.append(ac_analysis["ac_tempo_confidence"])
# ac_features.append(ac_analysis["ac_note_confidence"])
# ac_features.append(ac_analysis["ac_note_name"])
# ac_features.append(ac_analysis["ac_tonality"])
# ac_features.append(ac_analysis["ac_single_event"])
# ac_features.append(ac_analysis["ac_note_frequency"])
# ac_features.append(ac_analysis["ac_note_midi"])
# ac_features.append(ac_analysis["ac_loop"])
# ac_features.append(ac_analysis["ac_reverb"])
# ac_features.append(float(ac_analysis["ac_tempo"]))


num_out_features = len(tag_to_feature)

# if os.path.exists(DATA_PATH):
#     with open(DATA_PATH, "rb") as f:
#         dataset = pickle.load(f)


spectrogram = torchaudio.transforms.Spectrogram(n_fft=FFT_SIZE)
to_db = torchaudio.transforms.AmplitudeToDB()

# for filepath in pathlib.Path(SOUNDS_DIR).iterdir():
for id in ac_analysis:
    # if id not in dataset.id_set:
    # waveform, samplerate = torchaudio.load(filepath)
    # spec = spectrogram(waveform)
    # spec_db = to_db(spec)[0]

    ac_features = []
    if ac_analysis[id] is None:
        print(f"ac data missing, skipping {id}")
        continue
    try:
        ac_features.append(ac_analysis[id]["ac_depth"])
        ac_features.append(ac_analysis[id]["ac_temporal_centroid"])
        ac_features.append(ac_analysis[id]["ac_warmth"])
        ac_features.append(ac_analysis[id]["ac_hardness"])
        ac_features.append(ac_analysis[id]["ac_loudness"])
        ac_features.append(ac_analysis[id]["ac_roughness"])
        ac_features.append(ac_analysis[id]["ac_log_attack_time"])
        ac_features.append(ac_analysis[id]["ac_boominess"])
    except KeyError:
        print(f"ac features missing, skipping {id}")
        continue
    ac_features = torch.Tensor([ac_features]).T

    label = torch.zeros(num_out_features, 1)
    for tag in tags[id]:
        label[tag_to_feature[tag], 0] = 1
    dataset.add_samples(ac_features, label, id)
    print("added", id)

with open(DATA_PATH, "wb") as f:
    pickle.dump(dataset, f)
