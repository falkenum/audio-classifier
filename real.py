import enum
import torch
import pickle
from torch.utils.data import DataLoader
from common import MODEL_PATH, DATAPATH, TAG_BY_FEATURE_PATH, AudioClassifierModule

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

with open(DATAPATH, "rb") as f:
    dataset = pickle.load(f)

with open(TAG_BY_FEATURE_PATH, "rb") as f:
    tag_by_feature = pickle.load(f)

eval_dataloader = DataLoader(dataset, batch_size=1)
with torch.no_grad():
    for X, y in eval_dataloader:
        pred = model(X)

        tags = []
        for i, val in enumerate(pred[0]):
            if val.round() == 1:
                tags.append(tag_by_feature[i])
        tags.sort()
        print("predicted tags:", tags)
