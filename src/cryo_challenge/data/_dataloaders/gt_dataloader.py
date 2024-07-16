import numpy as np
import torch
from torch.utils.data import Dataset


class GT_Dataset(Dataset):
    def __init__(self, npy_file):
        self.npy_file = npy_file
        self.data = np.load(npy_file, mmap_mode="r")

        self.shape = self.data.shape
        self._dim = len(self.data.shape)

    def dim(self):
        return self._dim

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        sample = self.data[idx]
        return torch.from_numpy(sample.copy())
