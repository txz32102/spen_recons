""" 
cd /home/data1/musong/workspace/python/spen_recons
source .venv/bin/activate
export PYTHONPATH=$PYTHONPATH:.
CUDA_VISIBLE_DEVICES=1 python3 script/0320_spen_unsupervised_training.py
"""
import os
import argparse
import logging
from datetime import datetime
import random

import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

from torchmetrics.image import PeakSignalNoiseRatio
from torchmetrics.image import StructuralSimilarityIndexMeasure as SSIM

from dataset.spen_dataset_0309 import SpenDataset
from spenpy.spen import spen


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    print(f"Random seed set to: {seed}")
    
set_seed(42)

# ==========================================
# 1. Visualization Utility
# ==========================================
def plot_training_progress(lrs, roughs, finals, gts, rough_psnrs, rough_ssims, final_psnrs, final_ssims, save_path):
    """
    Creates a 4-row grid for visualizing training progress.
    Rows: LR | Rough | Recons (Final) | GT
    """
    cols = len(lrs)
    total_rows = 4 
    
    fig, axes = plt.subplots(total_rows, cols, figsize=(3 * cols, 3 * total_rows),
                             gridspec_kw={'wspace': 0.02, 'hspace': 0.02})
                             
    if cols == 1:
        axes = axes[:, None]
        
    for i in range(cols):
        ax_lr = axes[0, i]
        ax_rough = axes[1, i]
        ax_final = axes[2, i]
        ax_gt = axes[3, i]
        
        ax_lr.imshow(lrs[i], cmap='gray', vmin=0, vmax=1)
        ax_rough.imshow(roughs[i], cmap='gray', vmin=0, vmax=1)
        ax_final.imshow(finals[i], cmap='gray', vmin=0, vmax=1)
        ax_gt.imshow(gts[i], cmap='gray', vmin=0, vmax=1)
        
        # Simple Numbering
        ax_lr.text(0.05, 0.95, f"#{i+1}", color='white', fontsize=14, fontweight='bold',
                   ha='left', va='top', transform=ax_lr.transAxes)
                   
        # Annotate Row Names on the FIRST column
        if i == 0:
            ax_lr.text(0.05, 0.05, "LR", color='white', fontsize=14, fontweight='bold',
                       ha='left', va='bottom', transform=ax_lr.transAxes)
            ax_rough.text(0.05, 0.05, "Rough", color='orange', fontsize=14, fontweight='bold',
                       ha='left', va='bottom', transform=ax_rough.transAxes)
            ax_final.text(0.05, 0.05, "Recons", color='yellow', fontsize=14, fontweight='bold',
                       ha='left', va='bottom', transform=ax_final.transAxes)
            ax_gt.text(0.05, 0.05, "GT", color='cyan', fontsize=14, fontweight='bold',
                       ha='left', va='bottom', transform=ax_gt.transAxes)

        # Add PSNR and SSIM to Rough reconstruction
        rough_metric_text = f"PSNR: {rough_psnrs[i]:.2f}\nSSIM: {rough_ssims[i]:.3f}"
        ax_rough.text(0.05, 0.95, rough_metric_text, color='orange', fontsize=11, fontweight='bold',
                    ha='left', va='top', transform=ax_rough.transAxes)

        # Add PSNR and SSIM to Final DL reconstruction
        final_metric_text = f"PSNR: {final_psnrs[i]:.2f}\nSSIM: {final_ssims[i]:.3f}"
        ax_final.text(0.05, 0.95, final_metric_text, color='yellow', fontsize=11, fontweight='bold',
                    ha='left', va='top', transform=ax_final.transAxes)
                    
        ax_lr.axis('off')
        ax_rough.axis('off')
        ax_final.axis('off')
        ax_gt.axis('off')
        
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0.05, dpi=300)
    plt.close(fig)


# ==========================================
# 2. Model Architecture
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
        layers.append(
            nn.Conv2d(base_channels, 1, kernel_size=(4, 3), stride=(2, 1), padding=(1, 1))
        )
        self.net = nn.Sequential(*layers)
    def forward(self, lr):
        x = torch.cat([lr.real, lr.imag], dim=1) 
        return self.net(x)

class ImageRefiner(nn.Module):
    def __init__(self, num_blocks=5, base_channels=64):
        super().__init__()
        layers = [
            nn.Conv2d(2, base_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        ]
        for _ in range(num_blocks):
            layers.append(ResBlock(base_channels))
        layers.append(
            nn.Conv2d(base_channels, 1, kernel_size=3, padding=1)
        )
        self.net = nn.Sequential(*layers)
    def forward(self, rough_recon):
        x = torch.cat([rough_recon.real, rough_recon.imag], dim=1)
        return self.net(x)

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
# 3. Setup & Pipeline
# ==========================================
def setup_logger(log_dir):
    logger = logging.getLogger("SPEN_Trainer")
    logger.setLevel(logging.INFO)
    if logger.hasHandlers():
        logger.handlers.clear()
        
    formatter = logging.Formatter('%(asctime)s | %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    
    file_handler = logging.FileHandler(os.path.join(log_dir, 'training.log'))
    file_handler.setFormatter(formatter)
    
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger

def parse_args():
    parser = argparse.ArgumentParser(description="SPEN Unsupervised Training Pipeline")
    parser.add_argument("--exp_name", type=str, default="spen_unsupervised_base", help="Name of the experiment for logging purposes")
    parser.add_argument("--log_dir", type=str, default="log", help="Base directory for saving logs and images")
    parser.add_argument("--data_dir", type=str, default="data/0125_mixed_2000", help="Path to the dataset directory")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for training and evaluation")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs to train")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate for the optimizer")
    parser.add_argument("--train_ratio", type=float, default=0.8, help="Ratio of the dataset to use for training (e.g., 0.8 for 80%)")
    return parser.parse_args()

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # --- 0. Logging & Checkpoint Setup ---
    timestamp = datetime.now().strftime("%m%d%H%M")
    run_log_dir = os.path.join(args.log_dir, f"{timestamp}_{args.exp_name}")
    ckpt_dir = os.path.join(run_log_dir, "weights")
    os.makedirs(run_log_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)
    
    logger = setup_logger(run_log_dir)
    logger.info(f"Starting experiment: {args.exp_name}")
    logger.info(f"Arguments: {vars(args)}")

    # --- 1. Data Setup ---
    dt = SpenDataset(args.data_dir)
    train_size = int(args.train_ratio * len(dt))
    test_size = len(dt) - train_size
    train_dataset, test_dataset = random_split(dt, [train_size, test_size])
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    logger.info(f"Dataset Split: {train_size} Train | {test_size} Test")

    # Pre-compute InvA
    InvA, _ = spen(acq_point=(96, 96)).get_InvA()
    InvA = InvA.to(device)
    
    # --- 2. Model, Loss, Metrics, Optimizer ---
    model = SpenReconNet(InvA).to(device)
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    data_range = 2.0 
    psnr_metric = PeakSignalNoiseRatio(data_range=data_range).to(device)
    ssim_metric = SSIM(data_range=data_range).to(device)
    
    spen_simulator = spen(acq_point=(96, 96), device=device)

    best_test_psnr = -1.0  # Track best PSNR for checkpointing

    # --- 3. Training Loop ---
    for epoch in range(args.epochs):
        # -- TRAIN PHASE --
        model.train()
        train_loss = 0.0
        train_psnr_list, train_ssim_list = [], [] 
        
        for hr in train_loader:
            hr = hr.to(device) 
            hr_sim = hr.squeeze(1) 
            
            lr = spen_simulator.sim(hr_sim).to(device) 
            lr = lr.unsqueeze(1) 
            
            optimizer.zero_grad()
            final_hr, pred_phase, rough_recon = model(lr)
            
            loss = criterion(final_hr, hr)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_psnr_list.append(psnr_metric(final_hr, hr).item())
            train_ssim_list.append(ssim_metric(final_hr, hr).item())
            
        avg_train_loss = train_loss / len(train_loader)
        avg_train_psnr = np.mean(train_psnr_list)
        std_train_psnr = np.std(train_psnr_list)
        avg_train_ssim = np.mean(train_ssim_list)
        std_train_ssim = np.std(train_ssim_list)

        # -- EVALUATION PHASE --
        model.eval()
        test_loss = 0.0
        test_psnr_list, test_ssim_list = [], []
        
        with torch.no_grad():
            for batch_idx, hr in enumerate(test_loader):
                hr = hr.to(device)
                hr_sim = hr.squeeze(1)
                
                lr = spen_simulator.sim(hr_sim).to(device)
                lr = lr.unsqueeze(1)
                
                final_hr, pred_phase, rough_recon = model(lr)
                loss = criterion(final_hr, hr)
                
                test_loss += loss.item()
                test_psnr_list.append(psnr_metric(final_hr, hr).item())
                test_ssim_list.append(ssim_metric(final_hr, hr).item())
                
                # --- VISUALIZATION BLOCK (First Batch Only) ---
                if batch_idx == 0 and (epoch == 0 or (epoch + 1) % 10 == 0):
                    n_viz = min(5, hr.size(0))  # Max 5 images
                    
                    # Extract to CPU numpy
                    viz_lr = lr[:n_viz].abs().cpu().numpy()
                    viz_rough = rough_recon[:n_viz].abs().cpu().numpy()
                    viz_final = final_hr[:n_viz].cpu().numpy()
                    viz_gt = hr[:n_viz].cpu().numpy()

                    # Helper to safely map tensors to [0,1] for display
                    def norm_01(x):
                        return (x - x.min()) / (x.max() - x.min() + 1e-8)

                    p_lrs = [norm_01(viz_lr[i].squeeze()) for i in range(n_viz)]
                    p_roughs = [norm_01(viz_rough[i].squeeze()) for i in range(n_viz)]
                    p_finals = [norm_01(viz_final[i].squeeze()) for i in range(n_viz)]
                    p_gts = [norm_01(viz_gt[i].squeeze()) for i in range(n_viz)]
                    
                    # Calculate skimage metrics specifically for the visual plot (Both Final and Rough)
                    p_rough_psnrs = [peak_signal_noise_ratio(p_gts[i], p_roughs[i], data_range=1.0) for i in range(n_viz)]
                    p_rough_ssims = [structural_similarity(p_gts[i], p_roughs[i], data_range=1.0) for i in range(n_viz)]
                    
                    p_final_psnrs = [peak_signal_noise_ratio(p_gts[i], p_finals[i], data_range=1.0) for i in range(n_viz)]
                    p_final_ssims = [structural_similarity(p_gts[i], p_finals[i], data_range=1.0) for i in range(n_viz)]

                    img_path = os.path.join(run_log_dir, f"viz_epoch_{epoch+1:03d}.png")
                    plot_training_progress(p_lrs, p_roughs, p_finals, p_gts, 
                                           p_rough_psnrs, p_rough_ssims, 
                                           p_final_psnrs, p_final_ssims, img_path)
                    logger.info(f"[*] Generated visualization: {img_path}")

                    # Log Data Ranges
                    logger.info(f"--- Data ranges for Epoch {epoch+1:03d} (Test Batch 0) ---")
                    tensors_to_track = {
                        "LR": lr[:n_viz].abs(),
                        "Rough Recon": rough_recon[:n_viz].abs(),
                        "Final HR": final_hr[:n_viz],
                        "Ground Truth": hr[:n_viz]
                    }
                    for name, tensor in tensors_to_track.items():
                        stat_line = f"{name:<15} | Min: {tensor.min().item():>7.4f} | Max: {tensor.max().item():>7.4f} | Mean: {tensor.mean().item():>7.4f} | Std: {tensor.std().item():>7.4f}"
                        logger.info(stat_line) 
                    logger.info("-" * 55)

        avg_test_loss = test_loss / len(test_loader)
        avg_test_psnr = np.mean(test_psnr_list)
        std_test_psnr = np.std(test_psnr_list)
        avg_test_ssim = np.mean(test_ssim_list)
        std_test_ssim = np.std(test_ssim_list)

        # -- LOGGING & CHECKPOINTING --
        logger.info(
            f"Epoch [{epoch+1:03d}/{args.epochs}] "
            f"TRAIN: Loss {avg_train_loss:.4f}, PSNR {avg_train_psnr:.2f}±{std_train_psnr:.2f}, SSIM {avg_train_ssim:.4f}±{std_train_ssim:.4f} | "
            f"TEST: Loss {avg_test_loss:.4f}, PSNR {avg_test_psnr:.2f}±{std_test_psnr:.2f}, SSIM {avg_test_ssim:.4f}±{std_test_ssim:.4f}"
        )

        # Save Best Model
        if avg_test_psnr > best_test_psnr:
            best_test_psnr = avg_test_psnr
            torch.save(model.state_dict(), os.path.join(ckpt_dir, "best_ckpt.pth"))
            logger.info(f"    [+] New Best Model saved! (Test PSNR: {best_test_psnr:.2f})")

    # Save Last Model at the end of training
    torch.save(model.state_dict(), os.path.join(ckpt_dir, "last_ckpt.pth"))
    logger.info(f"    [+] Training complete. Last model saved.")

if __name__ == "__main__":
    args = parse_args()
    train(args)