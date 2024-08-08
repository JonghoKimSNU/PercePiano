from torch.utils.data import Dataset
import torch
import numpy as np
import pickle
import os

class FinetuneDataset(Dataset):
    """
    Expected data shape: (data_num, data_len)
    """

    def __init__(self, X, y):
        self.data = X
        self.label = y

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return torch.tensor(self.data[index]), torch.tensor(self.label[index])


class FinetuneDatasetAddon(Dataset):
    """
    Expected data shape: (data_num, data_len)
    """

    def __init__(self, X, y, addons_path):
        self.data = X
        self.label = y
        self.addons = np.load(addons_path + "_addons.npy", allow_pickle=True)
        self.note_location = pickle.load(open(addons_path + "_note_location.pkl", "rb"))
        self.data_len = np.load(addons_path + "_data_len.npy", allow_pickle=True)
        print("addon shape", self.addons.shape)
        print("note_location shape", len(self.note_location))
        print("data_len shape", self.data_len.shape)
        if os.path.exists(addons_path + "_found_addon_idxs.pkl"):
            self.found_addon_idxs = pickle.load(open(addons_path + "_found_addon_idxs.pkl", "rb"))
        else:    
            self.found_addon_idxs = None


    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        note_location = self.note_location[index]
        note_location = {
        'beat': torch.Tensor(note_location['beat']).type(torch.int32),
        'measure': torch.Tensor(note_location['measure']).type(torch.int32),
        'section': torch.Tensor(note_location['section']).type(torch.int32),
        'voice': torch.Tensor(note_location['voice']).type(torch.int32),
        }
        return_tuple = (
            torch.tensor(self.data[index]),
            torch.tensor(self.label[index]),
            torch.tensor(self.addons[index].astype(np.float32).tolist()),
            note_location,
            self.data_len[index],
            )
        if self.found_addon_idxs is not None:
            return_tuple += (self.found_addon_idxs[index],)
        return return_tuple


class FinetuneDatasetAlign(Dataset):
    """
    Expected data shape: (data_num, data_len)
    """

    def __init__(self, X, y, addons_path):
        self.data = X
        self.label = y
        self.addons = np.load(addons_path + "_align.npy", allow_pickle=True)
        print("addon shape", self.addons.shape)
        self.data_len = np.load(addons_path + "_data_len.npy", allow_pickle=True)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return (
            torch.tensor(self.data[index]),
            torch.tensor(self.label[index]),
            torch.tensor(self.addons[index].astype(np.float32).tolist()),
            self.data_len[index],
        )
