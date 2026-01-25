from typing import Callable, Optional
import glob
import os
import random
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


def _load_lr_mag(path: str) -> torch.Tensor:
    """
    Load LR complex image (H x W) from .mat file.
    Normalize by max(abs) to [0,1], then split into
    2xHxW tensor: [real, imag].
    """
    # Load array
    mat = loadmat(path)
    arr = _first_data_array(mat)
    arr = np.expand_dims(arr, axis=0)

    # Ensure complex dtype
    arr = arr.astype(np.complex64)

    # Normalize by max magnitude
    mag = np.abs(arr)
    max_val = mag.max()
    if max_val > 0:
        arr = arr / max_val

    # Split real & imag into 2 channels
    real = np.real(arr)
    imag = np.imag(arr)
    out = np.sqrt(real ** 2 + imag ** 2)

    return torch.from_numpy(out.astype(np.float32))


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
        <root>/lr/*.mat  (complex, HxW)  -> use magnitude

    Args:
        root: dataset root folder
        transforms_: optional torchvision transforms to apply (PIL or tensor).
                     If the transforms expect PIL images, they will be given a PIL image
                     converted from the 3ch tensor. Otherwise they’ll receive a tensor.
        unaligned: if True, sample B from a random index instead of aligned index
    """
    def __init__(
        self,
        root: str,
        transforms_: Optional[Callable] = None,
        unaligned: bool = False,
    ):
        self.root = root
        self.unaligned = unaligned
        self.transform = transforms_

        # Collect .mat files
        self.file_lr = sorted(glob.glob(os.path.join(root, "lr", "*.mat")))
        self.file_hr = sorted(glob.glob(os.path.join(root, "hr", "*.mat")))

        if len(self.file_lr) == 0 or len(self.file_hr) == 0:
            raise FileNotFoundError(
                f"No .mat files found under '{root}/hr' or '{root}/lr'. "
                f"Got {len(self.file_lr)} HR files and {len(self.file_hr)} LR files."
            )

    def __len__(self):
        return max(len(self.file_lr), len(self.file_hr))

    def __getitem__(self, index: int):
        path_hr = self.file_hr[index % len(self.file_hr)]
        if self.unaligned:
            j = random.randint(0, len(self.file_hr) - 1)
        else:
            j = index % len(self.file_hr)
        path_lr = self.file_lr[j]

        hr = _load_hr(path_hr) * 2 - 1
        lr = _load_lr_mag(path_lr) * 2 - 1
    
        return {"hr": hr, "lr": lr}


class SpenDataset_test(Dataset):
    def __init__(
        self,
        root: str,
    ):
        self.root = root

        # Collect .mat files
        self.file_lr = sorted(glob.glob(os.path.join(root, "lr", "*.mat")))
        self.file_hr = sorted(glob.glob(os.path.join(root, "hr", "*.mat")))

        if len(self.file_lr) == 0 or len(self.file_hr) == 0:
            raise FileNotFoundError(
                f"No .mat files found under '{root}/hr' or '{root}/lr'. "
                f"Got {len(self.file_lr)} HR files and {len(self.file_hr)} LR files."
            )

    def __len__(self):
        return max(len(self.file_lr), len(self.file_hr))

    def __getitem__(self, index: int):
        path_lr = self.file_lr[index]
        lr_id = os.path.splitext(os.path.basename(path_lr))[0]

        lr = _load_lr_mag(path_lr) * 2 - 1
    
        return {"lr": lr, "lr_id": lr_id, "lr_path": path_lr}