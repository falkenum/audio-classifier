from math import ceil, floor
from numpy import count_nonzero
import torch
from torch.utils.data import random_split, DataLoader, SequentialSampler
import pickle
from common import *
from db import AudioDatabase

num_sounds = 500
train_sounds, test_sounds = (floor(num_sounds * 0.8), ceil(num_sounds * 0.2))
train_data = SamplesDataset(train_sounds, shuffle=True)
test_data = SamplesDataset(test_sounds, shuffle=True)

model = AudioClassifierModule(1, NUM_FEATURE_LABELS).cuda(0)
db = AudioDatabase()

learning_rate = 1e-3
batch_size = 20
epochs = 20

train_dataloader = DataLoader(train_data, batch_size=batch_size, drop_last=True)
test_dataloader = DataLoader(test_data, batch_size=batch_size, drop_last=True)

def train_loop(dataloader, model, loss_fn, optimizer):
    # size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}]")


def test_loop(dataloader, model, loss_fn):
    size = 0
    num_batches = 0
    test_loss, correct = 0, 0
    positive_pred_correct = 0
    negative_pred_correct = 0
    positive_preds = 0
    negative_preds = 0

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            #print("y shape", y.shape)
            #print("pred max", pred.max())
            #print("pred min", pred.min())
            for row_idx, pred_row in enumerate(pred):
                pred_row = pred_row.round().int()
                correct += 1 if torch.all(y[row_idx].int().eq(pred_row)) else 0
                # TODO handle more than one channel here
                pred_row = pred_row.squeeze()
                #print("pred_row", pred_row)
                for col_idx, pred_elt in enumerate(pred_row):
                    #print("pred_elt", pred_elt)
                    #print("pred_elt shape", pred_elt.shape)
                    y_elt = y[row_idx].squeeze()[col_idx].int()
                    if pred_elt == 1:
                        positive_preds += 1
                        if y_elt == 1:
                            positive_pred_correct += 1
                    else:
                        negative_preds += 1
                        if y_elt == 0:
                            negative_pred_correct += 1


            size += len(X)
            num_batches += 1

    tag_preds = positive_preds + negative_preds
    tag_pred_correct = (positive_pred_correct + negative_pred_correct) / tag_preds
    negative_pred_correct /= negative_preds
    positive_pred_correct /= positive_preds

    test_loss /= num_batches
    correct /= size
    print(f"\tPrediction total accuracy: {(100*correct):>0.1f}%")
    print(f"\tPrediction tag accuracy: {(100*tag_pred_correct):>0.1f}%")
    print(f"\tPos pred accuracy: {(100*positive_pred_correct):>0.1f}%")
    print(f"\tNeg pred accuracy: {(100*negative_pred_correct):>0.1f}%")
    print()

loss_fn = torch.nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(train_dataloader, model, loss_fn, optimizer)
    test_loop(test_dataloader, model, loss_fn)
print("Done!")

with open(MODEL_PATH, "wb") as f:
    pickle.dump(model, f)
