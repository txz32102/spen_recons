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
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from PIL import Image

import torch
from torchvision.utils import save_image
from torch.utils.data import DataLoader

from util.utils_0125 import set_seed
from model.simple_gan_0125 import Generator
from dataset.spen_dataset_0125 import SpenDataset_test as SpenDataset

# ==========================================
# 1. Image Loading Utilities
# ==========================================
def mat_to_img01(path: str) -> np.ndarray:
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
    img = Image.open(path)
    arr = np.array(img).astype(np.float32)
    if arr.max() > 1.0:
        arr /= 255.0
    if arr.ndim == 3:
        arr = arr[..., 0]
    return arr

# ==========================================
# 2. Plotting Utility (Updated for 5 Random Images)
# ==========================================
def plot_comparison_random_5(lrs, gts, dls, psnrs, ssims, save_path):
    """
    Creates a 3x5 grid using exactly 5 images.
    Row 1: LR | Row 2: GT | Row 3: DL (Recons)
    """
    cols = 5
    total_rows = 3 
    
    # Create figure
    fig, axes = plt.subplots(total_rows, cols, figsize=(3 * cols, 3 * total_rows),
                             gridspec_kw={'wspace': 0.02, 'hspace': 0.02})
                             
    for i in range(cols):
        ax_lr = axes[0, i]
        ax_gt = axes[1, i]
        ax_dl = axes[2, i]
        
        ax_lr.imshow(lrs[i], cmap='gray', vmin=0, vmax=1)
        ax_gt.imshow(gts[i], cmap='gray', vmin=0, vmax=1)
        ax_dl.imshow(dls[i], cmap='gray', vmin=0, vmax=1)
        
        # 1. Simple Numbering Top-Left
        ax_lr.text(0.05, 0.95, f"#{i+1}", color='white', fontsize=14, fontweight='bold',
                   ha='left', va='top', transform=ax_lr.transAxes)
                   
        # 2. Annotate Row Names on the FIRST column only (Bottom-Left to avoid overlap)
        if i == 0:
            ax_lr.text(0.05, 0.05, "LR", color='white', fontsize=14, fontweight='bold',
                       ha='left', va='bottom', transform=ax_lr.transAxes)
            ax_gt.text(0.05, 0.05, "GT", color='cyan', fontsize=14, fontweight='bold',
                       ha='left', va='bottom', transform=ax_gt.transAxes)
            ax_dl.text(0.05, 0.05, "Recons", color='yellow', fontsize=14, fontweight='bold',
                       ha='left', va='bottom', transform=ax_dl.transAxes)

        # 3. Add PSNR and SSIM to DL reconstruction (Top-Left)
        metric_text = f"PSNR: {psnrs[i]:.2f}\nSSIM: {ssims[i]:.3f}"
        ax_dl.text(0.05, 0.95, metric_text, color='yellow', fontsize=11, fontweight='bold',
                    ha='left', va='top', transform=ax_dl.transAxes)
                    
        ax_lr.axis('off')
        ax_gt.axis('off')
        ax_dl.axis('off')
        
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0.05, dpi=300)
    plt.close(fig)
    print(f"\n[+] Saved random 5x3 comparison grid to: {save_path}")

# ==========================================
# 3. Metric Computation & Plot Trigger
# ==========================================
def compute_set_metrics_and_plot(
    hr_dir: str,
    lr_dir: str,
    log_dir: str,
    out_csv: str,
    plot_path: str,
) -> None:
    png_files = sorted(glob.glob(os.path.join(log_dir, "*.png")))

    # Map mat files by stem
    mat_map_hr = {Path(p).stem: p for p in glob.glob(os.path.join(hr_dir, "*.mat"))}
    mat_map_lr = {Path(p).stem: p for p in glob.glob(os.path.join(lr_dir, "*.mat"))}
    
    if not mat_map_hr:
        raise FileNotFoundError(f"No .mat files found under {hr_dir}")

    rows = []
    p_lrs, p_gts, p_dls = [], [], []
    p_psnrs, p_ssims = [], []

    for p_png in png_files:
        stem_iter = Path(p_png).stem
        parts = stem_iter.rsplit("_", 1)
        if len(parts) == 2 and parts[1].isdigit():
            stem = parts[0]
            iter_idx = parts[1]
        else:
            stem = stem_iter
            iter_idx = ""

        if stem not in mat_map_hr:
            continue

        p_mat_hr = mat_map_hr[stem]
        gt = mat_to_img01(p_mat_hr)
        pr = png_to_img01(p_png)
        lr = mat_to_img01(mat_map_lr[stem]) if stem in mat_map_lr else np.zeros_like(gt)

        psnr = peak_signal_noise_ratio(gt, pr, data_range=1.0)
        ssim = structural_similarity(gt, pr, data_range=1.0)
        
        p_lrs.append(lr)
        p_gts.append(gt)
        p_dls.append(pr)
        p_psnrs.append(psnr)
        p_ssims.append(ssim)

        rows.append({
            "id": stem,
            "iter": iter_idx,
            "png": p_png,
            "mat": p_mat_hr,
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

    print(f"\n[+] Wrote {len(rows)} rows -> {out_csv}")
    print(f"[=] Mean PSNR: {np.mean(p_psnrs):.4f} dB | Mean SSIM: {np.mean(p_ssims):.4f}")
    
    # --- RANDOM SAMPLING LOGIC ---
    if len(p_lrs) >= 5:
        random.seed(42)  # Set seed to 42 for reproducible randomness
        indices = random.sample(range(len(p_lrs)), 5)
        
        # Filter lists down to the 5 random indices
        r_lrs = [p_lrs[i] for i in indices]
        r_gts = [p_gts[i] for i in indices]
        r_dls = [p_dls[i] for i in indices]
        r_psnrs = [p_psnrs[i] for i in indices]
        r_ssims = [p_ssims[i] for i in indices]
        
        # Generate the 3x5 plot
        plot_comparison_random_5(r_lrs, r_gts, r_dls, r_psnrs, r_ssims, plot_path)
    else:
        print("\n[-] Not enough images to generate a 5-column plot (need at least 5).")


# ==========================================
# 4. Main Inference Script
# ==========================================
if __name__ == "__main__":
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

    netG.load_state_dict(torch.load(opt.generator_lr2hr))
    netG.eval()

    dataloader = DataLoader(SpenDataset(opt.dataroot),
                            batch_size=opt.batchSize,
                            num_workers=opt.n_cpu)

    os.makedirs(f'{opt.log_dir}/recons_hr', exist_ok=True)

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            real_lr_11 = batch['lr'].to(device)   
            recovered_hr = netG(real_lr_11)

            for b in range(len(batch['lr_id'])):
                save_image((recovered_hr[b]+1)/2, f"{opt.log_dir}/recons_hr/{batch['lr_id'][b]}_{i+1}.png")
            sys.stdout.write('\rGenerated images %04d of %04d' % (i+1, len(dataloader)))
        
    sys.stdout.write('\n')
    
    # Calculate the metric and trigger the plot
    compute_set_metrics_and_plot(
        hr_dir=f"{opt.dataroot}/hr",
        lr_dir=f"{opt.dataroot}/lr",
        log_dir=f"{opt.log_dir}/recons_hr",
        out_csv=f"{opt.log_dir}/metrix.csv",
        plot_path=f"{opt.log_dir}/Simulation_Comparison_Grid_3x5.png"
    )