""" 
cd /home/data1/musong/workspace/python/spen_recons
source .venv/bin/activate
export PYTHONPATH=$PYTHONPATH:.
CUDA_VISIBLE_DEVICES=1 python3 script/0319_pm_lr_real_data.py
"""
import argparse
import sys
import os
import glob
import re
import math
from pathlib import Path
from datetime import datetime

import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import save_image

from util.utils_0125 import set_seed
from model.simple_gan_0125 import Generator

# ==========================================
# 1. Plotting Utility
# ==========================================
def plot_comparison_5xn(inputs, trad_recons, dl_recons, slice_nums, save_path, slice_gap=4):
    """
    Creates a 5-column grid. 
    For each group of 5 slices, there are 3 rows:
      - Row 1: Low Res Inputs
      - Row 2: Traditional Reconstructions
      - Row 3: DL Reconstructions
    """
    # Apply slice gap
    inputs = inputs[::slice_gap]
    trad_recons = trad_recons[::slice_gap]
    dl_recons = dl_recons[::slice_gap]
    slice_nums = slice_nums[::slice_gap]
    
    num_slices = len(inputs)
    if num_slices == 0:
        return
        
    cols = 5
    row_groups = math.ceil(num_slices / cols)
    total_rows = row_groups * 3  # 3 rows per group (Input, Trad, DL)
    
    # Create figure
    fig, axes = plt.subplots(total_rows, cols, figsize=(3 * cols, 3 * total_rows),
                             gridspec_kw={'wspace': 0.02, 'hspace': 0.02})
                             
    # Ensure axes is a 2D array for consistent indexing
    if total_rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif total_rows == 1:
        axes = axes[None, :]
    elif cols == 1:
        axes = axes[:, None]
        
    for i in range(num_slices):
        c = i % cols
        r_group = i // cols
        
        r_in = r_group * 3          # Row 1: Input
        r_trad = r_group * 3 + 1    # Row 2: Traditional Recon
        r_dl = r_group * 3 + 2      # Row 3: DL Recon
        
        ax_in = axes[r_in, c]
        ax_trad = axes[r_trad, c]
        ax_dl = axes[r_dl, c]
        
        ax_in.imshow(inputs[i], cmap='gray')
        ax_trad.imshow(trad_recons[i], cmap='gray')
        ax_dl.imshow(dl_recons[i], cmap='gray')
        
        # Add labels to the top left of each image
        ax_in.text(0.05, 0.95, f"{slice_nums[i]} (LR)", color='white', fontsize=14, fontweight='bold',
                   ha='left', va='top', transform=ax_in.transAxes)
                   
        ax_trad.text(0.05, 0.95, f"{slice_nums[i]} (Trad)", color='cyan', fontsize=14, fontweight='bold',
                    ha='left', va='top', transform=ax_trad.transAxes)

        ax_dl.text(0.05, 0.95, f"{slice_nums[i]} (DL)", color='yellow', fontsize=14, fontweight='bold',
                    ha='left', va='top', transform=ax_dl.transAxes)
                    
        ax_in.axis('off')
        ax_trad.axis('off')
        ax_dl.axis('off')
        
    # Turn off unused subplots in the last group
    for i in range(num_slices, row_groups * cols):
        c = i % cols
        r_group = i // cols
        axes[r_group * 3, c].axis('off')
        axes[r_group * 3 + 1, c].axis('off')
        axes[r_group * 3 + 2, c].axis('off')
        
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0.05, dpi=300)
    plt.close(fig)
    print(f"\nSaved comparison grid to: {save_path}")

# ==========================================
# 2. Real Data Inference Dataset
# ==========================================
class SPENInferenceDataset(Dataset):
    """
    Loads real SPEN .mat files.
    Extracts both the Low Res Input and the Traditional Reconstruction.
    """
    def __init__(self, data_dir, input_key='Imag_low', trad_key='Image_SPEN', value_range=(-1, 1)):
        self.data_dir = data_dir
        self.input_key = input_key
        self.trad_key = trad_key
        self.value_range = value_range
        
        raw_files = glob.glob(os.path.join(data_dir, 'ratbrain_SPEN_96_*.mat'))
        raw_files.sort(key=lambda x: int(re.search(r'_(\d+)\.mat', os.path.basename(x)).group(1)))
        
        self.valid_files = []
        for f_path in raw_files:
            try:
                mat_data = scipy.io.loadmat(f_path)
                # Ensure BOTH keys exist in the file before adding it to the dataset
                if self._extract_array(mat_data, self.input_key) is not None and \
                   self._extract_array(mat_data, self.trad_key) is not None:
                    self.valid_files.append(f_path)
            except Exception:
                continue
                
        print(f"Dataset initialized: Found {len(self.valid_files)} valid slices containing both '{self.input_key}' and '{self.trad_key}'.")

    def _extract_array(self, data_dict, key):
        if key not in data_dict: return None
        raw_data = data_dict[key]
        if raw_data is None or (isinstance(raw_data, np.ndarray) and raw_data.size == 0): return None
        if not isinstance(raw_data, np.ndarray) or not (np.issubdtype(raw_data.dtype, np.number) or np.issubdtype(raw_data.dtype, np.complexfloating)): return None
        data = np.abs(np.squeeze(raw_data))
        if data.ndim == 2: return data
        elif data.ndim > 2: return data[:, :, 0]
        return None

    def _normalize(self, img):
        img_min, img_max = img.min(), img.max()
        norm_img = np.zeros_like(img, dtype=np.float32) if img_max == img_min else (img - img_min) / (img_max - img_min)
        if self.value_range == (-1, 1):
            norm_img = (norm_img * 2.0) - 1.0
        return norm_img.astype(np.float32)

    def __len__(self):
        return len(self.valid_files)

    def __getitem__(self, idx):
        f_path = self.valid_files[idx]
        file_id = Path(f_path).stem 
        
        mat_data = scipy.io.loadmat(f_path)
        
        # Extract and normalize both images
        img_in = self._normalize(self._extract_array(mat_data, self.input_key))
        img_trad = self._normalize(self._extract_array(mat_data, self.trad_key))
        
        # ==========================================
        # VERTICAL FLIP APPLIED HERE
        # ==========================================
        # .copy() is required to reset memory strides for PyTorch
        img_in = np.rot90(img_in, k=2).copy()
        
        # If the traditional image is also upside down in the plot, uncomment the line below:
        # img_trad = np.flipud(img_trad).copy()
        
        img_in = np.expand_dims(img_in, axis=0) 
        img_trad = np.expand_dims(img_trad, axis=0) 
        
        return torch.from_numpy(img_in), torch.from_numpy(img_trad), file_id


# ==========================================
# 3. Main Inference Script
# ==========================================
if __name__ == "__main__":
    set_seed(0)
    time_prefix = datetime.now().strftime("%m%d%H%M")

    parser = argparse.ArgumentParser()
    parser.add_argument('--batchSize', type=int, default=1, help='size of the batches')
    parser.add_argument('--dataroot', type=str, default='/home/data1/musong/workspace/2026/03/17/spen_matlab/export_data/pv360', help='root directory of the real .mat dataset')
    parser.add_argument('--input_nc', type=int, default=1, help='number of channels of input data')
    parser.add_argument('--output_nc', type=int, default=1, help='number of channels of output data')
    parser.add_argument('--size', type=int, default=96, help='size of the data (squared assumed)')
    parser.add_argument('--cuda', action='store_true', help='use GPU computation')
    parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')
    parser.add_argument('--generator_lr2hr', type=str, default='log/01251509_pm_lr/weights/netG_lr2hr.pth', help='B2A generator checkpoint file')
    parser.add_argument('--log_dir', type=str, default=f'log/{time_prefix}_pm_real_inference', help='directory to save generated images')
    opt = parser.parse_args()
    print(opt)

    if torch.cuda.is_available() and not opt.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

    device = torch.device("cuda" if opt.cuda else "cpu")
    netG = Generator(opt.input_nc, opt.output_nc)
    if opt.cuda:
        netG.cuda()

    print(f"Loading weights from: {opt.generator_lr2hr}")
    netG.load_state_dict(torch.load(opt.generator_lr2hr, map_location=device))
    netG.eval()

    dataset = SPENInferenceDataset(opt.dataroot, input_key='Imag_low', trad_key='Image_SPEN', value_range=(-1, 1))
    dataloader = DataLoader(dataset, batch_size=opt.batchSize, shuffle=False, num_workers=opt.n_cpu)

    out_dir = os.path.join(opt.log_dir, 'recons_hr_real')
    os.makedirs(out_dir, exist_ok=True)

    print(f"Starting inference. DL Outputs will be saved to {out_dir}")

    # Containers for the comparison plot
    collected_inputs = []
    collected_trad_recons = []
    collected_dl_recons = []
    collected_slice_nums = []

    with torch.no_grad(): 
        for i, (real_lr, real_trad, file_ids) in enumerate(dataloader):
            real_lr = real_lr.to(device)   
            recovered_hr = netG(real_lr)

            for b in range(len(file_ids)):
                # 1. Rescale to [0, 1] for saving and plotting
                lr_01 = (real_lr[b] + 1) / 2
                trad_01 = (real_trad[b] + 1) / 2
                dl_hr_01 = (recovered_hr[b] + 1) / 2
                
                # 2. Save individual generated PNGs
                save_path = os.path.join(out_dir, f"{file_ids[b]}.png")
                save_image(dl_hr_01, save_path)
                
                # 3. Store arrays for the comparison plot
                collected_inputs.append(lr_01.cpu().squeeze().numpy())
                collected_trad_recons.append(trad_01.cpu().squeeze().numpy())
                collected_dl_recons.append(dl_hr_01.cpu().squeeze().numpy())
                
                # Extract the pure slice number from the filename
                slice_num = str(int(re.search(r'_(\d+)$', file_ids[b]).group(1)))
                collected_slice_nums.append(slice_num)
                
            sys.stdout.write(f'\rProcessed batches {i+1:04d} of {len(dataloader):04d}')
            
    sys.stdout.write('\nDone!\n')
    
    # Generate the unified 5*N grid comparison plot
    plot_path = os.path.join(opt.log_dir, "Inference_Comparison_Grid_5xN.png")
    plot_comparison_5xn(collected_inputs, collected_trad_recons, collected_dl_recons, collected_slice_nums, plot_path, slice_gap=4)