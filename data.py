from common import *
from db import AudioDatabase
import pickle

dataset = AudioClassDataset()

db = AudioDatabase()
sounds = db.get_sounds()

tag_to_feature = {}
idx = 0
for _, tags, _, _, _ in sounds:
    for tag in tags:
        if tag not in tag_to_feature:
            tag_to_feature[tag] = idx
            idx += 1

num_out_features = len(tag_to_feature)

for id, tags, se_mean, se_max, se_min in sounds:
    in_features = torch.Tensor([[se_mean, se_max, se_min]]).T

    label = torch.zeros(num_out_features, 1)
    for tag in tags:
        label[tag_to_feature[tag], 0] = 1
    dataset.add_samples(in_features, label, id)

with open(DATA_PATH, "wb") as f:
    pickle.dump(dataset, f)