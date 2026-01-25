""" 
cd /home/data1/musong/workspace/python/spen_recons
source .venv/bin/activate
export PYTHONPATH=$PYTHONPATH:.
CUDA_VISIBLE_DEVICES=1 python3 script/0125_pm_lr_test.py
"""
import argparse
import random
import sys
import os
from pathlib import Path
import csv
import glob
from datetime import datetime

from scipy.io import loadmat
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from PIL import Image

import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from torch.utils.data import DataLoader

from util.utils_0125 import set_seed
from model.simple_gan_0125 import Generator
from dataset.spen_dataset_0125 import SpenDataset_test as SpenDataset

def mat_to_img01(path: str) -> np.ndarray:
    """Load .mat, pick first non-meta key, magnitude if complex, min-max to [0,1], return HxW float32."""
    md = loadmat(path)
    arr = None
    for k, v in md.items():
        if not k.startswith("__"):
            arr = v
            break
    if arr is None:
        raise KeyError(f"No data key found in {path} (only __meta keys).")

    x = np.asarray(arr).squeeze()
    if x.ndim > 2:
        x = x[..., 0]
    if np.iscomplexobj(x):
        x = np.abs(x)
    x = np.nan_to_num(x.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    xmin, xmax = float(np.min(x)), float(np.max(x))
    if xmax > xmin:
        x = (x - xmin) / (xmax - xmin)
    else:
        x = np.zeros_like(x, dtype=np.float32)
    return x

def png_to_img01(path: str) -> np.ndarray:
    """Load PNG (H x W x C or H x W), convert to float32 [0,1], return grayscale HxW by taking channel 0 if needed."""
    img = Image.open(path)
    # Ensure float in [0,1]
    arr = np.array(img).astype(np.float32)
    # If PNG saved from torchvision.save_image of float tensors, it’s already 0..255 uint8
    if arr.max() > 1.0:
        arr /= 255.0
    # If 3-channel, take the first channel (your saved fake_hr/fake_lr are 3ch copies)
    if arr.ndim == 3:
        arr = arr[..., 0]
    return arr

def compute_set_metrics(
    data_root: str,
    log_dir: str,
    out_csv: str,
) -> None:
    png_dir = log_dir
    mat_dir = data_root

    png_files = sorted(glob.glob(os.path.join(png_dir, "*.png")))

    # Map mat files by stem
    mat_map = {Path(p).stem: p for p in glob.glob(os.path.join(mat_dir, "*.mat"))}
    if not mat_map:
        raise FileNotFoundError(f"No .mat files found under {mat_dir}")

    rows = []
    psnrs, ssims = [], []

    for p_png in png_files:
        # Your saved names are like "<id>_<iter>.png"
        stem_iter = Path(p_png).stem
        # split by last underscore to recover original id (robust if id contains underscores)
        parts = stem_iter.rsplit("_", 1)
        if len(parts) == 2 and parts[1].isdigit():
            stem = parts[0]
            iter_idx = parts[1]
        else:
            stem = stem_iter
            iter_idx = ""

        if stem not in mat_map:
            # If unaligned or naming differs, skip politely
            # (You can adapt here to a different mapping scheme if needed.)
            continue

        p_mat = mat_map[stem]
        gt = mat_to_img01(p_mat)
        pr = png_to_img01(p_png)

        # skimage expects data_range set correctly for float images
        psnr = peak_signal_noise_ratio(gt, pr, data_range=1.0)
        ssim = structural_similarity(gt, pr, data_range=1.0)
        psnrs.append(psnr)
        ssims.append(ssim)

        rows.append({
            "id": stem,
            "iter": iter_idx,
            "png": p_png,
            "mat": p_mat,
            "H": gt.shape[0],
            "W": gt.shape[1],
            "PSNR": psnr,
            "SSIM": ssim
        })

    if not rows:
        raise RuntimeError("Found no matched (PNG, MAT) pairs. Check filenames and directories.")

    # Write CSV
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"[+] Wrote {len(rows)} rows -> {out_csv}")
    print(f"[=] Mean PSNR: {np.mean(psnrs):.4f} dB | Mean SSIM: {np.mean(ssims):.4f}")


set_seed(0)
time_prefix = datetime.now().strftime("%m%d%H%M")

parser = argparse.ArgumentParser()
parser.add_argument('--batchSize', type=int, default=1, help='size of the batches')
parser.add_argument('--dataroot', type=str, default='data/0125_test_data', help='root directory of the dataset')
parser.add_argument('--input_nc', type=int, default=1, help='number of channels of input data')
parser.add_argument('--output_nc', type=int, default=1, help='number of channels of output data')
parser.add_argument('--size', type=int, default=96, help='size of the data (squared assumed)')
parser.add_argument("--which", choices=["hr","lr"], default="hr", help="Evaluate recon set.")
parser.add_argument('--cuda', action='store_true', help='use GPU computation')
parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')
parser.add_argument('--generator_lr2hr', type=str, default='log/01251509_pm_lr/weights/netG_lr2hr.pth', help='B2A generator checkpoint file')
parser.add_argument('--log_dir', type=str, default=f'log/{time_prefix}_pm_lr_test', help='directory to save logs and model checkpoints')
opt = parser.parse_args()
print(opt)

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")


netG = Generator(opt.input_nc, opt.output_nc)

device = torch.device("cuda" if opt.cuda else "cpu")

if opt.cuda:
    netG.cuda()

# Load state dicts
netG.load_state_dict(torch.load(opt.generator_lr2hr))

# Set model's test mode
netG.eval()

dataloader = DataLoader(SpenDataset(opt.dataroot),
                        batch_size=opt.batchSize,
                        num_workers=opt.n_cpu)

for i, batch in enumerate(dataloader):
    real_lr_11 = batch['lr'].to(device)   

    recovered_hr = netG(real_lr_11)

    os.makedirs(f'{opt.log_dir}/recons_hr', exist_ok=True)

    # Save each sample in the batch
    for b in range(len(batch['lr_id'])):
        save_image((recovered_hr[b]+1)/2, f"{opt.log_dir}/recons_hr/{batch['lr_id'][b]}_{i+1}.png")
    sys.stdout.write('\rGenerated images %04d of %04d' % (i+1, len(dataloader)))
    
    # calculate the metric
compute_set_metrics(
    data_root=f"{opt.dataroot}/hr",
    log_dir=f"{opt.log_dir}/recons_hr",
    out_csv=f"{opt.log_dir}/metrix.csv",
)

sys.stdout.write('\n')
###################################