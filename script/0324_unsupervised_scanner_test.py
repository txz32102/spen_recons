""" 
cd /home/data1/musong/workspace/python/spen_recons
source .venv/bin/activate
export PYTHONPATH=$PYTHONPATH:.
CUDA_VISIBLE_DEVICES=1 python3 script/0324_unsupervised_scanner_test.py
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
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import save_image

# Assuming set_seed exists in util.utils_0125 as per previous codes
try:
    from util.utils_0125 import set_seed
except ImportError:
    import random
    def set_seed(seed=42):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

from spenpy.spen import spen

# ==========================================
# 1. Unsupervised Network Architecture
# ==========================================
class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        )
    def forward(self, x):
        return x + self.net(x)

class PhaseEstimator(nn.Module):
    def __init__(self, num_blocks=3, base_channels=64):
        super().__init__()
        layers = [
            nn.Conv2d(2, base_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        ]
        for _ in range(num_blocks):
            layers.append(ResBlock(base_channels))
        layers.append(nn.Conv2d(base_channels, 1, kernel_size=(4, 3), stride=(2, 1), padding=(1, 1)))
        self.net = nn.Sequential(*layers)
        
    def forward(self, lr):
        x = torch.cat([lr.real, lr.imag], dim=1) 
        phase_map = self.net(x)                  
        return phase_map

class ImageRefiner(nn.Module):
    def __init__(self, num_blocks=5, base_channels=64):
        super().__init__()
        layers = [
            nn.Conv2d(2, base_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        ]
        for _ in range(num_blocks):
            layers.append(ResBlock(base_channels))
        layers.append(nn.Conv2d(base_channels, 1, kernel_size=3, padding=1))
        self.net = nn.Sequential(*layers)
        
    def forward(self, rough_recon):
        x = torch.cat([rough_recon.real, rough_recon.imag], dim=1)
        final_hr = self.net(x)
        return final_hr

class SpenReconNet(nn.Module):
    def __init__(self, InvA):
        super().__init__()
        self.phase_estimator = PhaseEstimator()
        self.refiner = ImageRefiner()
        self.register_buffer('InvA', InvA)

    def forward(self, lr):
        phase_map = self.phase_estimator(lr)
        
        even_data = lr[:, :, 1::2, :].clone()
        phase_shift = torch.exp(-1j * phase_map.to(torch.complex64))
        even_data = even_data * phase_shift
        
        phase_corrected_data = lr.clone()
        phase_corrected_data[:, :, 1::2, :] = even_data
        
        rough_recon = torch.matmul(self.InvA, phase_corrected_data)
        final_hr = self.refiner(rough_recon)
        
        return final_hr, phase_map, rough_recon

# ==========================================
# 2. Plotting Utility
# ==========================================
def plot_comparison_5xn(inputs, trad_recons, dl_recons, slice_nums, save_path, slice_gap=4):
    inputs = inputs[::slice_gap]
    trad_recons = trad_recons[::slice_gap]
    dl_recons = dl_recons[::slice_gap]
    slice_nums = slice_nums[::slice_gap]
    
    num_slices = len(inputs)
    if num_slices == 0:
        print("No slices collected for plotting grid.")
        return
        
    cols = 5
    row_groups = math.ceil(num_slices / cols)
    total_rows = row_groups * 3  
    
    fig, axes = plt.subplots(total_rows, cols, figsize=(3 * cols, 3 * total_rows),
                             gridspec_kw={'wspace': 0.02, 'hspace': 0.02})
                             
    if total_rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif total_rows == 1:
        axes = axes[None, :]
    elif cols == 1:
        axes = axes[:, None]
        
    for i in range(num_slices):
        c = i % cols
        r_group = i // cols
        
        r_in = r_group * 3          
        r_trad = r_group * 3 + 1    
        r_dl = r_group * 3 + 2      
        
        ax_in = axes[r_in, c]
        ax_trad = axes[r_trad, c]
        ax_dl = axes[r_dl, c]
        
        ax_in.imshow(inputs[i], cmap='gray')
        ax_trad.imshow(trad_recons[i], cmap='gray')
        ax_dl.imshow(dl_recons[i], cmap='gray')
        
        ax_in.text(0.05, 0.95, f"{slice_nums[i]} (LR)", color='white', fontsize=14, fontweight='bold',
                   ha='left', va='top', transform=ax_in.transAxes)
        ax_trad.text(0.05, 0.95, f"{slice_nums[i]} (Trad)", color='cyan', fontsize=14, fontweight='bold',
                     ha='left', va='top', transform=ax_trad.transAxes)
        ax_dl.text(0.05, 0.95, f"{slice_nums[i]} (DL)", color='yellow', fontsize=14, fontweight='bold',
                    ha='left', va='top', transform=ax_dl.transAxes)
                    
        ax_in.axis('off')
        ax_trad.axis('off')
        ax_dl.axis('off')
        
    for i in range(num_slices, row_groups * cols):
        c = i % cols
        r_group = i // cols
        axes[r_group * 3, c].axis('off')
        axes[r_group * 3 + 1, c].axis('off')
        axes[r_group * 3 + 2, c].axis('off')
        
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0.05, dpi=300)
    plt.close(fig)
    print(f"\n[+] Saved comparison grid to: {save_path}")

# ==========================================
# 3. Real Data Inference Dataset (Complex)
# ==========================================
class SPENRealComplexDataset(Dataset):
    """
    Loads real SPEN .mat files.
    Extracts the Low Res Input as COMPLEX, and the Traditional Recon as MAGNITUDE.
    """
    def __init__(self, data_dir, input_key='Imag_low', trad_key='Image_SPEN'):
        self.data_dir = data_dir
        self.input_key = input_key
        self.trad_key = trad_key
        
        raw_files = glob.glob(os.path.join(data_dir, 'ratbrain_SPEN_96_*.mat'))
        raw_files.sort(key=lambda x: int(re.search(r'_(\d+)\.mat', os.path.basename(x)).group(1)))
        
        self.valid_files = []
        for f_path in raw_files:
            try:
                mat_data = scipy.io.loadmat(f_path)
                if self._extract_array(mat_data, self.input_key, keep_complex=True) is not None and \
                   self._extract_array(mat_data, self.trad_key, keep_complex=False) is not None:
                    self.valid_files.append(f_path)
            except Exception:
                continue
                
        print(f"Dataset initialized: Found {len(self.valid_files)} valid slices.")

    def _extract_array(self, data_dict, key, keep_complex):
        if key not in data_dict: return None
        raw_data = data_dict[key]
        if raw_data is None or (isinstance(raw_data, np.ndarray) and raw_data.size == 0): return None
        
        data = np.squeeze(raw_data)
        if not keep_complex:
            data = np.abs(data)
            
        if data.ndim == 2: return data
        elif data.ndim > 2: return data[:, :, 0]
        return None

    def __len__(self):
        return len(self.valid_files)

    def __getitem__(self, idx):
        f_path = self.valid_files[idx]
        file_id = Path(f_path).stem 
        
        mat_data = scipy.io.loadmat(f_path)
        
        img_in_complex = self._extract_array(mat_data, self.input_key, keep_complex=True)
        img_trad_mag = self._extract_array(mat_data, self.trad_key, keep_complex=False)
        
        # Normalize complex input by max magnitude to keep phase intact
        in_max_mag = np.abs(img_in_complex).max()
        if in_max_mag > 0:
            img_in_complex = img_in_complex / in_max_mag
            
        # Normalize traditional recon to [0, 1]
        t_min, t_max = img_trad_mag.min(), img_trad_mag.max()
        if t_max > t_min:
            img_trad_mag = (img_trad_mag - t_min) / (t_max - t_min)
            
        # Input for DL network requires 180 deg rotation based on training convention
        img_in_complex = np.rot90(img_in_complex, k=2).copy().astype(np.complex64)
        
        # --- MODIFICATION HERE ---
        # Removed np.rot90(..., k=2) for traditional data to rotate it 180 degrees visually
        # compared to previous script version.
        img_trad_mag = img_trad_mag.copy().astype(np.float32)
        # --------------------------
        
        img_in_complex = np.expand_dims(img_in_complex, axis=0) 
        img_trad_mag = np.expand_dims(img_trad_mag, axis=0) 
        
        return torch.from_numpy(img_in_complex), torch.from_numpy(img_trad_mag), file_id

# ==========================================
# 4. Main Inference Script
# ==========================================
if __name__ == "__main__":
    set_seed(42)
    time_prefix = datetime.now().strftime("%m%d%H%M")

    parser = argparse.ArgumentParser()
    parser.add_argument('--batchSize', type=int, default=1)
    parser.add_argument('--dataroot', type=str, default='/home/data1/musong/workspace/2026/03/17/spen_matlab/export_data/pv360')
    parser.add_argument('--cuda', action='store_true', help='use GPU computation')
    parser.add_argument('--n_cpu', type=int, default=8)
    parser.add_argument('--ckpt_path', type=str, 
                        default='/home/data1/musong/workspace/python/spen_recons/log/03242047_spen_unsupervised_base/checkpoints/best_ckpt.pth', 
                        help='Path to the best checkpoint from unsupervised training')
    parser.add_argument('--log_dir', type=str, default=f'log/{time_prefix}_unsupervised_real_inference')
    opt = parser.parse_args()
    print(opt)

    device = torch.device("cuda" if opt.cuda else "cpu")

    # Fetch InvA necessary for Unsupervised Model Initialization
    print("Pre-computing InvA from spen simulator...")
    try:
        InvA, _ = spen(acq_point=(96, 96)).get_InvA()
        InvA = InvA.to(device)
    except Exception as e:
        print(f"Error computing InvA. Ensure spenpy is installed and configured. Error: {e}")
        sys.exit(1)

    # Initialize Model
    netG = SpenReconNet(InvA)
    if opt.cuda:
        netG.cuda()

    print(f"Loading weights from: {opt.ckpt_path}")
    try:
        netG.load_state_dict(torch.load(opt.ckpt_path, map_location=device))
    except FileNotFoundError:
        print(f"Checkpoint not found at: {opt.ckpt_path}")
        sys.exit(1)
    netG.eval()

    dataset = SPENRealComplexDataset(opt.dataroot, input_key='Imag_low', trad_key='Image_SPEN')
    dataloader = DataLoader(dataset, batch_size=opt.batchSize, shuffle=False, num_workers=opt.n_cpu)

    out_dir = os.path.join(opt.log_dir, 'recons_hr_real')
    os.makedirs(out_dir, exist_ok=True)

    print(f"Starting inference. Outputs will be saved to {out_dir}")

    collected_inputs = []
    collected_trad_recons = []
    collected_dl_recons = []
    collected_slice_nums = []

    with torch.no_grad(): 
        for i, (real_lr_complex, real_trad_mag, file_ids) in enumerate(dataloader):
            real_lr_complex = real_lr_complex.to(device)   
            
            # Unsupervised model returns: final_hr, pred_phase, rough_recon
            final_hr, pred_phase, rough_recon = netG(real_lr_complex)

            for b in range(len(file_ids)):
                # Convert complex LR input to magnitude for plotting
                lr_mag = torch.abs(real_lr_complex[b]).cpu()
                
                # Normalize output to [0, 1] for saving
                dl_hr = final_hr[b].cpu()
                dl_min, dl_max = dl_hr.min(), dl_hr.max()
                dl_hr_01 = (dl_hr - dl_min) / (dl_max - dl_min + 1e-8)
                
                trad_01 = real_trad_mag[b].cpu()

                # Save individual generated PNGs
                save_path = os.path.join(out_dir, f"{file_ids[b]}.png")
                save_image(dl_hr_01, save_path)
                
                collected_inputs.append(lr_mag.squeeze().numpy())
                collected_trad_recons.append(trad_01.squeeze().numpy())
                collected_dl_recons.append(dl_hr_01.squeeze().numpy())
                
                match = re.search(r'_(\d+)$', file_ids[b])
                slice_num = str(int(match.group(1))) if match else file_ids[b]
                collected_slice_nums.append(slice_num)
                
            sys.stdout.write(f'\rProcessed batches {i+1:04d} of {len(dataloader):04d}')
            
    sys.stdout.write('\nDone!\n')
    
    plot_path = os.path.join(opt.log_dir, "Unsupervised_Inference_Grid_5xN.png")
    plot_comparison_5xn(collected_inputs, collected_trad_recons, collected_dl_recons, collected_slice_nums, plot_path, slice_gap=4)