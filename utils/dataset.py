from torch.utils.data import Dataset

class XaiDataset(Dataset):
    def __init__(self, data_list):
        self.data = []
        self.build_dataset(data_list)

    def build_dataset(self, data_list):
        for elem in data_list:
            x, y = elem
            self.data.append((x, y))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]