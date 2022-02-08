import torch

class AudioClassifierModel(torch.nn.Module):
    def __init__(self, in_features) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(in_features, 1)
        
