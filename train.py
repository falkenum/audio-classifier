from math import ceil, floor
import torch
from torch.utils.data import random_split, DataLoader
import pickle
import common

with open(common.TAG_BY_FEATURE_PATH, "rb") as f:
    tag_by_feature = pickle.load(f)

with open(common.DATA_PATH, "rb") as f:
    dataset = pickle.load(f)

num_out_features = dataset.labels.shape[0]
model = common.AudioClassifierModule(num_out_features)

learning_rate = 1e-3
batch_size = 10
epochs = 20

data_lengths = (floor(len(dataset) * 0.9), ceil(len(dataset) * 0.1))
train_data, test_data = random_split(dataset, data_lengths)
train_dataloader = DataLoader(train_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
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
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(dataloader, model, loss_fn):
    # top_count = 3
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()

            for row_idx, row in enumerate(pred):
                pmax_idx = row.argmax()
                # pmax = row[pmax_idx]
                # for p_idx, p in enumerate(row):
                #     if p > pmax:
                #         pmax_idx = p_idx
                # idx_with_prob.sort(key=lambda elt: elt[1], reverse=True)
                # idx_with_prob
                # pred_with_prob.append(idx_with_prob)
                correct += y[row_idx][pmax_idx]

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(train_dataloader, model, loss_fn, optimizer)
    test_loop(test_dataloader, model, loss_fn)
print("Done!")

with open(common.MODEL_PATH, "wb") as f:
    pickle.dump(model, f)