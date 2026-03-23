""" 
cd /home/data1/musong/workspace/python/spen_recons
source .venv/bin/activate
export PYTHONPATH=$PYTHONPATH:.
CUDA_VISIBLE_DEVICES=1 python3 script/0309_spen_unsupervised_training.py
"""
import os
import argparse
import logging
from datetime import datetime
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import torchvision.utils as vutils

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

class ResBlock(nn.Module):
    """A standard Residual Block with skip connection."""
    def __init__(self, channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        )
        
    def forward(self, x):
        # Skip connection: Add the original input back to the output of the conv layers
        return x + self.net(x)

class PhaseEstimator(nn.Module):
    def __init__(self, num_blocks=3, base_channels=64):
        super().__init__()
        # 1. Initial feature extraction
        layers = [
            nn.Conv2d(2, base_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        ]
        
        # 2. Deep feature processing using Residual Blocks
        for _ in range(num_blocks):
            layers.append(ResBlock(base_channels))
            
        # 3. Final downsampling specific to SPEN phase mapping (stride 2 on height)
        layers.append(
            nn.Conv2d(base_channels, 1, kernel_size=(4, 3), stride=(2, 1), padding=(1, 1))
        )
        
        self.net = nn.Sequential(*layers)
        
    def forward(self, lr):
        x = torch.cat([lr.real, lr.imag], dim=1) 
        phase_map = self.net(x)                  
        return phase_map

class ImageRefiner(nn.Module):
    def __init__(self, num_blocks=5, base_channels=64):
        super().__init__()
        # 1. Initial feature extraction
        layers = [
            nn.Conv2d(2, base_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        ]
        
        # 2. Deep feature refinement using Residual Blocks
        for _ in range(num_blocks):
            layers.append(ResBlock(base_channels))
            
        # 3. Collapse back to a single HR image channel
        layers.append(
            nn.Conv2d(base_channels, 1, kernel_size=3, padding=1)
        )
        
        self.net = nn.Sequential(*layers)
        
    def forward(self, rough_recon):
        x = torch.cat([rough_recon.real, rough_recon.imag], dim=1)
        final_hr = self.net(x)
        return final_hr

class SpenReconNet(nn.Module):
    def __init__(self, InvA):
        super().__init__()
        # The main network structure remains identical
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
    
    parser.add_argument("--exp_name", type=str, default="spen_unsupervised_base", 
                        help="Name of the experiment for logging purposes")
    parser.add_argument("--log_dir", type=str, default="log", 
                        help="Base directory for saving logs and images")
    parser.add_argument("--data_dir", type=str, default="data/0125_mixed_2000", 
                        help="Path to the dataset directory")
    parser.add_argument("--batch_size", type=int, default=8, 
                        help="Batch size for training and evaluation")
    parser.add_argument("--epochs", type=int, default=100, 
                        help="Number of epochs to train")
    parser.add_argument("--lr", type=float, default=1e-4, 
                        help="Learning rate for the optimizer")
    parser.add_argument("--train_ratio", type=float, default=0.8, 
                        help="Ratio of the dataset to use for training (e.g., 0.8 for 80%)")
    
    return parser.parse_args()

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # --- 0. Logging Setup ---
    timestamp = datetime.now().strftime("%m%d%H%M")
    # Combine the base log directory with the dynamically generated folder name
    run_log_dir = os.path.join(args.log_dir, f"{timestamp}_{args.exp_name}")
    os.makedirs(run_log_dir, exist_ok=True)
    
    logger = setup_logger(run_log_dir)
    
    # Log the parsed arguments
    logger.info(f"Starting experiment: {args.exp_name}")
    logger.info(f"Arguments: {vars(args)}")
    logger.info(f"Logs and images saved to: {run_log_dir}")

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

    # --- 3. Training Loop ---
    for epoch in range(args.epochs):
        # -- TRAIN PHASE --
        model.train()
        train_loss = 0.0
        train_psnr_list, train_ssim_list = [], [] # Track all batches for std dev
        
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
                
                # Save a visualization image for the first batch of testing
                if batch_idx == 0:
                    n_viz = min(4, hr.size(0))
                    
                    # Extract initial tensors
                    viz_lr = lr[:n_viz].abs()
                    viz_rough = rough_recon[:n_viz].abs()
                    viz_final = final_hr[:n_viz]
                    viz_gt = hr[:n_viz]

                    # 1. Helper function to min-max normalize a tensor to [0, 1] per image
                    def norm_to_neg1_1(x):
                        x_min = x.amin(dim=(1, 2, 3), keepdim=True)
                        x_max = x.amax(dim=(1, 2, 3), keepdim=True)
                        # Scale to [0, 1], then map to [0, 1]
                        return (x - x_min) / (x_max - x_min + 1e-8)

                    # 2. Map the positive magnitude images to the [-1, 1] domain
                    viz_lr = norm_to_neg1_1(viz_lr)
                    viz_rough = norm_to_neg1_1(viz_rough)

                    # 3. Stack them: LR -> Rough -> Final -> GT
                    grid = torch.cat([viz_lr, viz_rough, viz_final, viz_gt], dim=0)

                    # 4. Save image with an explicit value_range to enforce global consistency
                    img_path = os.path.join(run_log_dir, f"viz_epoch_{epoch+1:03d}.png")
                    vutils.save_image(
                        grid, 
                        img_path, 
                        nrow=n_viz, 
                        normalize=True, 
                        value_range=(0, 1.0),
                        scale_each=False
                    )

                    # -- NEW: Log data ranges directly to the main logger --
                    logger.info(f"--- Data ranges for Epoch {epoch+1:03d} (Test Batch 0) ---")
                    
                    tensors_to_track = {
                        "LR": lr[:n_viz].abs(),
                        "Rough Recon": rough_recon[:n_viz].abs(),
                        "Final HR": final_hr[:n_viz],
                        "Ground Truth": hr[:n_viz]
                    }
                    
                    for name, tensor in tensors_to_track.items():
                        t_min = tensor.min().item()
                        t_max = tensor.max().item()
                        t_mean = tensor.mean().item()
                        t_std = tensor.std().item()
                        
                        stat_line = f"{name:<15} | Min: {t_min:>7.4f} | Max: {t_max:>7.4f} | Mean: {t_mean:>7.4f} | Std: {t_std:>7.4f}"
                        logger.info(stat_line) 
                    logger.info("-" * 55)
                    # -----------------------------------------

        avg_test_loss = test_loss / len(test_loader)
        avg_test_psnr = np.mean(test_psnr_list)
        std_test_psnr = np.std(test_psnr_list)
        avg_test_ssim = np.mean(test_ssim_list)
        std_test_ssim = np.std(test_ssim_list)

        # -- LOGGING --
        logger.info(
            f"Epoch [{epoch+1:03d}/{args.epochs}] "
            f"TRAIN: Loss {avg_train_loss:.4f}, PSNR {avg_train_psnr:.2f}±{std_train_psnr:.2f}, SSIM {avg_train_ssim:.4f}±{std_train_ssim:.4f} | "
            f"TEST: Loss {avg_test_loss:.4f}, PSNR {avg_test_psnr:.2f}±{std_test_psnr:.2f}, SSIM {avg_test_ssim:.4f}±{std_test_ssim:.4f}"
        )

    logger.info("Training complete.")

if __name__ == "__main__":
    args = parse_args()
    train(args)