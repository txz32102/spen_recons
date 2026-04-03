""" 
cd /home/data1/musong/workspace/python/spen_recons
source .venv/bin/activate
export PYTHONPATH=$PYTHONPATH:.
CUDA_VISIBLE_DEVICES=3 python3 script/0325_end_to_end.py
"""

"""
This project implements an end-to-end deep learning pipeline for image reconstruction. A lightweight UNet architecture is trained to recover high-resolution ground truth images from degraded inputs. To create these inputs, a SPEN (Spatiotemporal Encoding) simulation is applied dynamically during the training process. This simulation mimics realistic acquisition artifacts by injecting a random noise level uniformly sampled between 0.0 and 0.02 on the fly. The dataset is explicitly partitioned into training and validation splits, allowing the model's generalization and reconstruction quality to be rigorously tested and tracked on unseen data using PSNR and SSIM metrics.
"""

import os
import glob
import random
import argparse
import logging
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as T
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

from skimage.metrics import peak_signal_noise_ratio as calc_psnr
from skimage.metrics import structural_similarity as calc_ssim

from spenpy.spen import spen

# ==========================================
# 0. Reproducibility
# ==========================================
def set_seed(seed=42):
    """Sets the random seed for deterministic execution."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# ==========================================
# 1. Argument Parser
# ==========================================
def parse_args():
    parser = argparse.ArgumentParser(description="SPEN Unsupervised Training Pipeline")
    
    parser.add_argument("--exp_name", type=str, default="end_to_end_spen", 
                        help="Name of the experiment for logging purposes")
    parser.add_argument("--log_dir", type=str, default="log", 
                        help="Base directory for saving logs and images")
    parser.add_argument("--data_dir", type=str, default="/home/data1/musong/workspace/python/spen_recons/data/0325_rat/hr", 
                        help="Path to the dataset directory")
    parser.add_argument("--batch_size", type=int, default=8, 
                        help="Batch size for training and evaluation")
    parser.add_argument("--epochs", type=int, default=100, 
                        help="Number of epochs to train")
    parser.add_argument("--lr", type=float, default=1e-4, 
                        help="Learning rate for the optimizer")
    parser.add_argument("--train_ratio", type=float, default=0.8, 
                        help="Ratio of the dataset to use for training (e.g., 0.8 for 80%)")
    parser.add_argument("--seed", type=int, default=42, 
                        help="Random seed for reproducibility")
    
    return parser.parse_args()

# ==========================================
# 2. Dataset Definition
# ==========================================
class SPENDataset(Dataset):
    def __init__(self, data_dir):
        # Assuming images are saved as pngs, adjust extension if they are .npy or .tif
        self.img_paths = glob.glob(os.path.join(data_dir, "*.png")) 
        self.transform = T.Compose([
            T.ToTensor() # Converts to [0, 1] and shape (1, H, W)
        ])

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img = Image.open(self.img_paths[idx]).convert('L') # Ensure Grayscale
        gt_tensor = self.transform(img)
        return gt_tensor

# ==========================================
# 3. Simple Reconstruction Model (UNet)
# ==========================================
class SimpleUNet(nn.Module):
    """A standard, deeper UNet for 96x96 image reconstruction."""
    def __init__(self):
        super(SimpleUNet, self).__init__()
        
        # Added BatchNorm2d for training stability in deeper networks
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

        # Encoder (Downsampling)
        self.enc1 = conv_block(1, 64)       # Input: 96x96
        self.enc2 = conv_block(64, 128)     # Input: 48x48
        self.enc3 = conv_block(128, 256)    # Input: 24x24

        # Bottleneck
        self.bottleneck = conv_block(256, 512) # Input: 12x12

        # Decoder (Upsampling)
        self.up3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec3 = conv_block(512 + 256, 256)

        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec2 = conv_block(256 + 128, 128)

        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec1 = conv_block(128 + 64, 64)

        # Final projection
        self.final = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)                   # Shape: (B, 64, 96, 96)
        e2 = self.enc2(self.pool(e1))       # Shape: (B, 128, 48, 48)
        e3 = self.enc3(self.pool(e2))       # Shape: (B, 256, 24, 24)

        # Bottleneck
        b = self.bottleneck(self.pool(e3))  # Shape: (B, 512, 12, 12)

        # Decoder with skip connections
        d3 = self.dec3(torch.cat([self.up3(b), e3], dim=1))  # Shape: (B, 256, 24, 24)
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1)) # Shape: (B, 128, 48, 48)
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1)) # Shape: (B, 64, 96, 96)

        return torch.sigmoid(self.final(d1))

# ==========================================
# 4. Provided Visualization Function
# ==========================================
def plot_training_progress(inputs, finals, gts, input_psnrs, input_ssims, final_psnrs, final_ssims, save_path):
    """
    Creates a 3-row grid for visualizing training progress.
    Rows: Input (Degraded) | Recons (Final) | GT
    """
    cols = len(inputs)
    total_rows = 3 
    
    fig, axes = plt.subplots(total_rows, cols, figsize=(3 * cols, 3 * total_rows),
                             gridspec_kw={'wspace': 0.02, 'hspace': 0.02})
                             
    if cols == 1:
        axes = axes[:, None]
        
    for i in range(cols):
        ax_in = axes[0, i]
        ax_final = axes[1, i]
        ax_gt = axes[2, i]
        
        ax_in.imshow(inputs[i], cmap='gray', vmin=0, vmax=1)
        ax_final.imshow(finals[i], cmap='gray', vmin=0, vmax=1)
        ax_gt.imshow(gts[i], cmap='gray', vmin=0, vmax=1)
        
        # Turn off axis ticks but keep the box to display ylabels
        for ax in [ax_in, ax_final, ax_gt]:
            ax.set_xticks([])
            ax.set_yticks([])
        
        # Index Numbering at the Bottom Left
        ax_in.text(0.05, 0.05, f"#{i+1}", color='white', fontsize=14, fontweight='bold',
                   ha='left', va='bottom', transform=ax_in.transAxes)
                   
        # Annotate Row Names on the FIRST column (using Y-axis label to prevent overlap)
        if i == 0:
            ax_in.set_ylabel("Input", color='orange', fontsize=14, fontweight='bold', rotation=0, labelpad=30, va='center')
            ax_final.set_ylabel("Recons", color='yellow', fontsize=14, fontweight='bold', rotation=0, labelpad=30, va='center')
            ax_gt.set_ylabel("GT", color='cyan', fontsize=14, fontweight='bold', rotation=0, labelpad=30, va='center')

        # Add PSNR and SSIM to Input (Top Left)
        in_metric_text = f"PSNR: {input_psnrs[i]:.2f}\nSSIM: {input_ssims[i]:.3f}"
        ax_in.text(0.05, 0.95, in_metric_text, color='orange', fontsize=11, fontweight='bold',
                    ha='left', va='top', transform=ax_in.transAxes)

        # Add PSNR and SSIM to Final DL reconstruction (Top Left)
        final_metric_text = f"PSNR: {final_psnrs[i]:.2f}\nSSIM: {final_ssims[i]:.3f}"
        ax_final.text(0.05, 0.95, final_metric_text, color='yellow', fontsize=11, fontweight='bold',
                    ha='left', va='top', transform=ax_final.transAxes)
        
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0.05, dpi=300)
    plt.close(fig)

# ==========================================
# 5. Main Training Logic
# ==========================================
def train(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # --- 0. Logging & Directory Setup ---
    timestamp = datetime.now().strftime("%m%d%H%M")
    run_log_dir = os.path.join(args.log_dir, f"{timestamp}_{args.exp_name}")
    
    # Create subdirectories for ckpts and images
    ckpt_dir = os.path.join(run_log_dir, "ckpt")
    img_dir = os.path.join(run_log_dir, "images")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(img_dir, exist_ok=True)
    
    # Configure logging to save to file AND print to console
    log_file = os.path.join(run_log_dir, "train_log.txt")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()]
    )
    logging.info(f"Experiment started: {args.exp_name}")
    logging.info(f"Arguments: {args}")

    # --- 1. Data Loading ---
    dataset = SPENDataset(args.data_dir)
    train_size = int(args.train_ratio * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    logging.info(f"Dataset Loaded: {len(train_dataset)} Train, {len(val_dataset)} Val")

    # --- 2. Model & Optimizer ---
    model = SimpleUNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.L1Loss() # L1 Loss is standard for image restoration tasks
    
    best_psnr = 0.0

    # --- 3. Training Loop ---
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        
        for gts in train_loader:
            gts = gts.to(device) # Shape: (B, 1, 96, 96)
            
            # Simulate Degradation (On-the-fly)
            noise_lvl = random.uniform(0.0, 0.02)
            
            # Squeeze to shape (B, 96, 96) for SPEN simulation, then restore channel dim
            with torch.no_grad():
                rough_complex = spen(acq_point=[96, 96], noise_level=noise_lvl, device=device).sim(gts.squeeze(1))
                # rough_complex is (B, 96, 96). Abs and add channel back -> (B, 1, 96, 96)
                rough = rough_complex.abs().unsqueeze(1)
                
                # Normalize rough input to [0,1] for stable neural network training
                rough = (rough - rough.min()) / (rough.max() - rough.min() + 1e-8)

            optimizer.zero_grad()
            recons = model(rough)
            
            loss = criterion(recons, gts)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
        avg_train_loss = epoch_loss / len(train_loader)
        
        # --- 4. Validation & Evaluation ---
        model.eval()
        val_loss = 0.0
        val_psnrs, val_ssims = [], []
        
        first_batch = True # Used to grab the first batch for plotting
        
        with torch.no_grad():
            for gts in val_loader:
                gts = gts.to(device)
                
                # Fixed noise for consistent validation tracking
                rough_complex = spen(acq_point=[96, 96], noise_level=0.01, device=device).sim(gts.squeeze(1))
                rough = rough_complex.abs().unsqueeze(1)
                rough = (rough - rough.min()) / (rough.max() - rough.min() + 1e-8)
                
                recons = model(rough)
                val_loss += criterion(recons, gts).item()
                
                # Move to CPU numpy for metric calculation
                gts_np = gts.cpu().numpy().squeeze(1)     # (B, 96, 96)
                rough_np = rough.cpu().numpy().squeeze(1) # (B, 96, 96)
                recons_np = recons.cpu().numpy().squeeze(1) # (B, 96, 96)
                
                batch_rough_psnrs = []
                batch_rough_ssims = []
                batch_final_psnrs = []
                batch_final_ssims = []
                
                for i in range(gts_np.shape[0]):
                    # Compute Metrics
                    r_psnr = calc_psnr(gts_np[i], rough_np[i], data_range=1.0)
                    r_ssim = calc_ssim(gts_np[i], rough_np[i], data_range=1.0)
                    f_psnr = calc_psnr(gts_np[i], recons_np[i], data_range=1.0)
                    f_ssim = calc_ssim(gts_np[i], recons_np[i], data_range=1.0)
                    
                    batch_rough_psnrs.append(r_psnr)
                    batch_rough_ssims.append(r_ssim)
                    batch_final_psnrs.append(f_psnr)
                    batch_final_ssims.append(f_ssim)
                    
                    val_psnrs.append(f_psnr)
                    val_ssims.append(f_ssim)
                
                # Visualization for the first validation batch
                if first_batch:
                    plot_save_path = os.path.join(img_dir, f"epoch_{epoch:03d}.png") # Saved to images/
                    # Using 'rough' for both LR and Rough fields since SPEN gives us one degraded output.
                    # If you have separate LR inputs vs rough analytic reconstructions, swap them here.
                    plot_training_progress(
                        inputs=rough_np, 
                        finals=recons_np, 
                        gts=gts_np,
                        input_psnrs=batch_rough_psnrs, 
                        input_ssims=batch_rough_ssims,
                        final_psnrs=batch_final_psnrs, 
                        final_ssims=batch_final_ssims,
                        save_path=plot_save_path
                    )
                    first_batch = False

        avg_val_loss = val_loss / len(val_loader)
        
        # Calculate mean and standard deviation
        avg_val_psnr = np.mean(val_psnrs)
        std_val_psnr = np.std(val_psnrs)
        avg_val_ssim = np.mean(val_ssims)
        std_val_ssim = np.std(val_ssims)
        
        # --- 5. Logging and Checkpointing ---
        log_msg = (f"Epoch [{epoch:03d}/{args.epochs}] | "
                   f"Train L1: {avg_train_loss:.4f} | Val L1: {avg_val_loss:.4f} | "
                   f"Val PSNR: {avg_val_psnr:.2f} ± {std_val_psnr:.2f} | "
                   f"Val SSIM: {avg_val_ssim:.4f} ± {std_val_ssim:.4f}")
        logging.info(log_msg)
        
        # Save Last Checkpoint (Saved to ckpt/)
        last_ckpt_path = os.path.join(ckpt_dir, "last.pth")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, last_ckpt_path)
        
        # Save Best Checkpoint (Saved to ckpt/)
        if avg_val_psnr > best_psnr:
            best_psnr = avg_val_psnr
            best_ckpt_path = os.path.join(ckpt_dir, "best.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_psnr': best_psnr
            }, best_ckpt_path)
            logging.info(f"--> Saved new best checkpoint at epoch {epoch} with PSNR: {best_psnr:.2f}")

if __name__ == "__main__":
    args = parse_args()
    train(args)