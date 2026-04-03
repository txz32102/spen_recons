""" 
cd /home/data1/musong/workspace/python/spen_recons
source .venv/bin/activate
export PYTHONPATH=$PYTHONPATH:.
CUDA_VISIBLE_DEVICES=1 python3 script/0323_scanner_data_phase_map.py
"""

"""
We visualize the phase map of the scanner data as well as different ways of reconstruction
"""
import os
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import torch
import logging

# Mute the missing font spam from matplotlib
logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)

# Import spenpy (Ensure this is in your PYTHONPATH as per your shell script)
from spenpy.spen import spen

# --- 1. Configuration & Styling ---
timestamp = datetime.now().strftime("%m%d%H%M")
log_name = f"{timestamp}_spen_scanner_visualization"
SAVE_DIR = os.path.join('log', log_name)
os.makedirs(SAVE_DIR, exist_ok=True)

# Set Times New Roman globally, but add standard Linux fallbacks so it doesn't crash/spam
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif', 'serif']

MAT_FILE = '/home/data1/musong/workspace/2026/03/17/spen_matlab/export_data/pv360/ratbrain_SPEN_96_13.mat'

# --- 2. Load and Process Complex Data ---
mat = scipy.io.loadmat(MAT_FILE)

# Extract complex arrays
img_low_raw = np.squeeze(mat['Imag_low'])
img_origin_raw = np.squeeze(mat['Imag_origin'])
img_spen_raw = np.squeeze(mat['Image_SPEN']) # Traditional recon from the scanner

# Rotate origin and low images by 180 degrees (k=2) to match scanner orientation
img_low = np.rot90(img_low_raw, 2).copy()
img_origin = np.rot90(img_origin_raw, 2).copy()
img_spen = img_spen_raw.copy() 

# --- 3. Extract Phase Map ---
# Calculate the phase difference map using the complex conjugate
phase_map = np.angle(img_low * np.conj(img_origin))

# --- 4. Setup Matrices and PyTorch Tensors ---
InvA, _ = spen(acq_point=(96, 96)).get_InvA()
if not isinstance(InvA, torch.Tensor):
    InvA = torch.tensor(InvA, dtype=torch.complex64)

# Convert arrays to PyTorch tensors
input_tensor = torch.tensor(img_origin, dtype=torch.complex64)     
img_low_tensor = torch.tensor(img_low, dtype=torch.complex64)      

# --- 5. Reconstructions ---
# Uncorrected: applying InvA directly to the original image
result_uncorrected = torch.matmul(InvA, input_tensor)

# Corrected: applying InvA to the low image
result_corrected = torch.matmul(InvA, img_low_tensor)

# --- 6. Plotting the 2x3 Figure ---
# Increased hspace from 0.05 to 0.3 to make room for titles "in front" (above) of the images
fig, axes = plt.subplots(2, 3, figsize=(14, 9), gridspec_kw={'wspace': 0.1, 'hspace': 0.15})

# Updated helper function: Use set_title instead of ax.text
def plot_mri(ax, image_data, label):
    ax.imshow(np.abs(image_data), cmap='gray')
    ax.axis('off')
    # Position the title "in front" (above) the axis
    ax.set_title(label, fontsize=14, pad=10, fontweight='bold', fontfamily='serif')

# --- Row 1: Inputs and Scanner Recon ---
plot_mri(axes[0, 0], img_origin, 'Image Original')
plot_mri(axes[0, 1], img_low, 'Image Low')
plot_mri(axes[0, 2], img_spen, 'Scanner Reconstructed')

# --- Row 2: Decoded Recons and Phase Map ---
plot_mri(axes[1, 0], result_uncorrected.cpu().numpy(), 'Uncorrected (Original + InvA)')
plot_mri(axes[1, 1], result_corrected.cpu().numpy(), 'Corrected (Image Low + InvA)')

# Plot Phase Map with Colorbar (axes[1, 2])
im_phase = axes[1, 2].imshow(phase_map, cmap='twilight', vmin=-np.pi, vmax=np.pi)
axes[1, 2].axis('off')
axes[1, 2].set_title('Extracted Phase Map (radians)', fontsize=14, pad=10, fontweight='bold', fontfamily='serif')

cbar = fig.colorbar(im_phase, ax=axes[1, 2], fraction=0.046, pad=0.04)
cbar.ax.tick_params(labelsize=12)

# Save and show
save_path = os.path.join(SAVE_DIR, '2x3_reconstruction_and_phase_titled.png')
plt.savefig(save_path, bbox_inches='tight', dpi=300)
print(f"Plot successfully saved to: {save_path}")

plt.show()