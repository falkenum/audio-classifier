from torch.utils.data import Dataset

class AudioClassDataset(Dataset):
    def __init__(self, class_data, id_set: set):
        super(AudioClassDataset, self).__init__()
        self.id_set = id_set
        self.class_data = class_data
    
    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)
