import torch
import pickle
import common

class AudioClassifierModule(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(common.FEATURE_SIZE, 1)
    
    def forward(self, x):
        return self.linear(x)


# criterion = torch.nn.NLLLoss()
model = AudioClassifierModule()
print(model)
# optimizer = torch.optim.SGD(model.parameters())

datapath = "./data.pickle"
with open(datapath, "rb") as f:
    dataset = pickle.load(f)
