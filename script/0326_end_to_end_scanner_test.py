""" 
cd /home/data1/musong/workspace/python/spen_recons
source .venv/bin/activate
export PYTHONPATH=$PYTHONPATH:.
CUDA_VISIBLE_DEVICES=3 python3 script/0326_end_to_end_scanner_test.py
"""

"""
This project evaluates a supervised deep learning pipeline for high-resolution SPEN (Spatiotemporal Encoding) image reconstruction using real-world scanner data. A lightweight SimpleUNet architecture, previously trained on simulated degraded images with dynamic noise injection, is deployed to reconstruct 96x96 rat brain images from raw .mat scanner files. The script automates the transition from complex-valued scanner input to normalized magnitude images, applying orientation corrections to ensure consistency with the training distribution. Performance is visually validated through a multi-row grid comparison against traditional MATLAB reconstructions, demonstrating the model's ability to suppress artifacts and recover structural details on unseen, physical acquisition data.
"""
import argparse
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

# ==========================================
# 1. Model Architecture (Must match 0325_end_to_end.py)
# ==========================================
class SimpleUNet(nn.Module):
    def __init__(self):
        super(SimpleUNet, self).__init__()
        def conv_block(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True)
            )
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.enc1 = conv_block(1, 64)
        self.enc2 = conv_block(64, 128)
        self.enc3 = conv_block(128, 256)
        self.bottleneck = conv_block(256, 512)
        self.up3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec3 = conv_block(512 + 256, 256)
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec2 = conv_block(256 + 128, 128)
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec1 = conv_block(128 + 64, 64)
        self.final = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b = self.bottleneck(self.pool(e3))
        d3 = self.dec3(torch.cat([self.up3(b), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return torch.sigmoid(self.final(d1))

# ==========================================
# 2. Plotting Utility (Original 5xN Style)
# ==========================================
def plot_comparison_5xn(inputs, trad_recons, dl_recons, slice_nums, save_path, slice_gap=4):
    inputs = inputs[::slice_gap]
    trad_recons = trad_recons[::slice_gap]
    dl_recons = dl_recons[::slice_gap]
    slice_nums = slice_nums[::slice_gap]
    
    num_slices = len(inputs)
    if num_slices == 0: return
        
    cols = 5
    row_groups = math.ceil(num_slices / cols)
    total_rows = row_groups * 3  
    
    fig, axes = plt.subplots(total_rows, cols, figsize=(3 * cols, 3 * total_rows),
                             gridspec_kw={'wspace': 0.01, 'hspace': 0.01})
    
    # Ensure axes is always a 2D array for consistent indexing
    if total_rows == 1: axes = axes[None, :]
    if cols == 1: axes = axes[:, None]
    
    # CRITICAL: Turn off ALL axes globally first to handle empty subplots
    for ax in axes.flatten():
        ax.axis('off')
        
    for i in range(num_slices):
        c = i % cols
        r_group = i // cols
        r_in, r_trad, r_dl = r_group * 3, r_group * 3 + 1, r_group * 3 + 2
        
        # Plotting
        axes[r_in, c].imshow(inputs[i], cmap='gray')
        axes[r_trad, c].imshow(trad_recons[i], cmap='gray')
        axes[r_dl, c].imshow(dl_recons[i], cmap='gray')
        
        # Text Labels
        axes[r_in, c].text(0.05, 0.95, f"{slice_nums[i]} (LR)", color='white', 
                           fontsize=10, fontweight='bold', transform=axes[r_in, c].transAxes,
                           va='top', ha='left')
        axes[r_trad, c].text(0.05, 0.95, "Trad", color='cyan', 
                             fontsize=10, fontweight='bold', transform=axes[r_trad, c].transAxes,
                             va='top', ha='left')
        axes[r_dl, c].text(0.05, 0.95, "DL", color='yellow', 
                           fontsize=10, fontweight='bold', transform=axes[r_dl, c].transAxes,
                           va='top', ha='left')

    # Use bbox_inches='tight' to remove the extra white border around the figure
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0, dpi=300)
    plt.close(fig)

# ==========================================
# 3. Real Scanner Dataset
# ==========================================
class SPENScannerDataset(Dataset):
    def __init__(self, data_dir, input_key='Imag_low', trad_key='Image_SPEN'):
        self.data_dir = data_dir
        self.input_key, self.trad_key = input_key, trad_key
        self.valid_files = glob.glob(os.path.join(data_dir, 'ratbrain_SPEN_96_*.mat'))
        self.valid_files.sort(key=lambda x: int(re.search(r'_(\d+)\.mat', os.path.basename(x)).group(1)))

    def __len__(self): return len(self.valid_files)

    def __getitem__(self, idx):
        f_path = self.valid_files[idx]
        mat_data = scipy.io.loadmat(f_path)
        
        # Process Input: Magnitude -> Rotate -> Normalize
        img_in = np.abs(np.squeeze(mat_data[self.input_key]))
        img_in = np.rot90(img_in, k=2) # Align with training PNGs
        img_in = (img_in - img_in.min()) / (img_in.max() - img_in.min() + 1e-8)
        
        # Process Trad: Magnitude -> Normalize
        img_trad = np.abs(np.squeeze(mat_data[self.trad_key]))
        img_trad = (img_trad - img_trad.min()) / (img_trad.max() - img_trad.min() + 1e-8)
        
        return torch.FloatTensor(img_in).unsqueeze(0), torch.FloatTensor(img_trad).unsqueeze(0), Path(f_path).stem

# ==========================================
# 4. Main Inference
# ==========================================
if __name__ == "__main__":
    time_prefix = datetime.now().strftime("%m%d%H%M")
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataroot', type=str, default='/home/data1/musong/workspace/2026/03/17/spen_matlab/export_data/pv360')
    parser.add_argument('--ckpt', type=str, default='/home/data1/musong/workspace/python/spen_recons/log/03251740_end_to_end_spen/ckpt/best.pth')
    parser.add_argument('--log_dir', type=str, default=f'log/{time_prefix}_supervised_scanner_test')
    opt = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(os.path.join(opt.log_dir, 'recons'), exist_ok=True)

    # Load Model
    model = SimpleUNet().to(device)
    checkpoint = torch.load(opt.ckpt, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    dataloader = DataLoader(SPENScannerDataset(opt.dataroot), batch_size=1, shuffle=False)
    all_in, all_trad, all_dl, all_ids = [], [], [], []

    print(f"[*] Testing {len(dataloader)} slices...")
    with torch.no_grad():
        for i, (img_in, img_trad, file_id) in enumerate(dataloader):
            output = model(img_in.to(device))
            
            # Save individual PNG
            save_image(output, os.path.join(opt.log_dir, 'recons', f"{file_id[0]}.png"))
            
            # Collect for grid
            all_in.append(img_in.squeeze().numpy())
            all_trad.append(img_trad.squeeze().numpy())
            all_dl.append(output.cpu().squeeze().numpy())
            match = re.search(r'_(\d+)$', file_id[0])
            all_ids.append(match.group(1) if match else file_id[0])

    plot_comparison_5xn(all_in, all_trad, all_dl, all_ids, os.path.join(opt.log_dir, "Scanner_Comparison_Grid.png"))
    print(f"[+] Complete. Results in {opt.log_dir}")