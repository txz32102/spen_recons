import glob
import os
from scipy.io import loadmat
import numpy as np
import torch
from torch.utils.data import Dataset

def _first_data_array(mat_dict: dict) -> np.ndarray:
    """Return the first non-meta array from a scipy.io.loadmat dict."""
    for k, v in mat_dict.items():
        if not k.startswith("__"):
            return v
    raise KeyError("No data key found in .mat file (only __meta keys present).")

def _load_hr(path: str) -> torch.Tensor:
    """Load HR real image (HxW) -> 1xHxW tensor in [0,1]."""
    mat = loadmat(path)
    arr = _first_data_array(mat)
    arr = np.expand_dims(arr, axis=0)
    arr = arr / arr.max()
    return torch.tensor(arr.astype(np.float32))

class SpenDataset(Dataset):
    """
    Loads paired .mat files from:
        <root>/hr/*.mat  (real, HxW)

    Args:
        root: dataset root folder
    """
    def __init__(
        self,
        root: str,
    ):
        self.root = root
        self.file_hr = sorted(glob.glob(os.path.join(root, "hr", "*.mat")))

    def __len__(self):
        return len(self.file_hr)

    def __getitem__(self, index: int):
        path_hr = self.file_hr[index % len(self.file_hr)]
        hr = _load_hr(path_hr)
        return hr
