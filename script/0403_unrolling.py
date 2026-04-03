""" 
cd /home/data1/musong/workspace/python/spen_recons
source .venv/bin/activate
export PYTHONPATH=$PYTHONPATH:.
CUDA_VISIBLE_DEVICES=2 python3 script/0403_unrolling.py
"""

"""
This project implements an end-to-end deep learning pipeline for image reconstruction. 
It uses an Unrolled ADMM architecture, combining a lightweight UNet (for regularization/denoising) 
with data consistency layers to recover high-resolution ground truth images from degraded inputs. 
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
    parser = argparse.ArgumentParser(description="SPEN Matrix Reconstruction - Unrolled ADMM")
    
    parser.add_argument("--exp_name", type=str, default="spen_matrix_recons_admm", 
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
                        help="Ratio of the dataset to use for training")
    parser.add_argument("--noise_min", type=float, default=0.0, 
                        help="Minimum noise level to inject into HR images")
    parser.add_argument("--noise_max", type=float, default=0.02, 
                        help="Maximum noise level to inject into HR images")
    parser.add_argument("--admm_iters", type=int, default=3, 
                        help="Number of unrolled ADMM iterations")
    parser.add_argument("--seed", type=int, default=42, 
                        help="Random seed for reproducibility")
    
    return parser.parse_args()

# ==========================================
# 2. Dataset Definition
# ==========================================
class SPENDataset(Dataset):
    def __init__(self, data_dir):
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
# 3. Models (UNet + Unrolled ADMM)
# ==========================================
class SimpleUNet(nn.Module):
    """A standard UNet acting as the learned regularizer (Z-update)."""
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

class DataConsistencyLayer(nn.Module):
    """
    Forces the network's prediction to remain faithful to the measured physics.
    Takes a gradient descent step using the forward operator AFinal and inverse InvA.
    """
    def __init__(self):
        super(DataConsistencyLayer, self).__init__()
        self.alpha = nn.Parameter(torch.tensor(0.1)) # Learnable step size
        
    def forward(self, x, y, AFinal, InvA, rho, z_minus_u):
        complex_x = x.squeeze(1).to(dtype=torch.complex64) * 1j
        
        # Forward process: Ax
        Ax = torch.matmul(AFinal, complex_x)
        
        # Error in measurement space: Ax - y
        error = Ax - y
        
        # Map error back to image space: A^H(Ax - y)
        grad_physics = torch.matmul(InvA, error).abs().unsqueeze(1)
        
        # ADMM penalty gradient
        grad_admm = rho * (x - z_minus_u)
        
        # Gradient Descent Step
        x_new = x - self.alpha * (grad_physics + grad_admm)
        return x_new

class UnrolledADMM(nn.Module):
    """Unrolls the ADMM optimization algorithm into a trainable neural network."""
    def __init__(self, num_iterations=3):
        super(UnrolledADMM, self).__init__()
        self.num_iterations = num_iterations
        
        # Shared UNet prior across iterations to save memory
        self.unet = SimpleUNet() 
        self.rho = nn.Parameter(torch.tensor(0.05)) # Learnable penalty parameter
        self.dc_layers = nn.ModuleList([DataConsistencyLayer() for _ in range(num_iterations)])

    def forward(self, y_meas, rough_input, AFinal, InvA):
        x = rough_input
        u = torch.zeros_like(x)
        
        for i in range(self.num_iterations):
            # 1. Z-Update (Neural Network Denoising)
            z_input = x + u
            # Normalize to [0,1] range expected by Sigmoid at end of UNet
            z_input_norm = (z_input - z_input.min()) / (z_input.max() - z_input.min() + 1e-8)
            z = self.unet(z_input_norm) 
            
            # 2. X-Update (Data Consistency)
            z_minus_u = z - u
            x = self.dc_layers[i](x, y_meas, AFinal, InvA, self.rho, z_minus_u)
            
            # 3. U-Update (Dual Variable Update)
            u = u + (x - z)
            
        # Optional: final normalization to ensure outputs are bounded [0,1] for loss/viz
        x_out = (x - x.min()) / (x.max() - x.min() + 1e-8)
        return x_out

# ==========================================
# 4. Provided Visualization Function
# ==========================================
def plot_training_progress(inputs, finals, gts, input_psnrs, input_ssims, final_psnrs, final_ssims, save_path):
    import matplotlib.pyplot as plt # Ensure standard pyplot is used here
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
        
        for ax in [ax_in, ax_final, ax_gt]:
            ax.set_xticks([])
            ax.set_yticks([])
        
        ax_in.text(0.05, 0.05, f"#{i+1}", color='white', fontsize=14, fontweight='bold',
                   ha='left', va='bottom', transform=ax_in.transAxes)
                   
        if i == 0:
            ax_in.set_ylabel("Input", color='orange', fontsize=14, fontweight='bold', rotation=0, labelpad=30, va='center')
            ax_final.set_ylabel("Recons", color='yellow', fontsize=14, fontweight='bold', rotation=0, labelpad=30, va='center')
            ax_gt.set_ylabel("GT", color='cyan', fontsize=14, fontweight='bold', rotation=0, labelpad=30, va='center')

        in_metric_text = f"PSNR: {input_psnrs[i]:.2f}\nSSIM: {input_ssims[i]:.3f}"
        ax_in.text(0.05, 0.95, in_metric_text, color='orange', fontsize=11, fontweight='bold',
                    ha='left', va='top', transform=ax_in.transAxes)

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
    
    ckpt_dir = os.path.join(run_log_dir, "ckpt")
    img_dir = os.path.join(run_log_dir, "images")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(img_dir, exist_ok=True)
    
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

    # --- 2. Pre-compute SPEN Matrices ---
    logging.info("Pre-computing InvA and AFinal matrices...")
    InvA, AFinal = spen(acq_point=(96, 96)).get_InvA()
    
    if not isinstance(InvA, torch.Tensor):
        InvA = torch.tensor(InvA)
    if not isinstance(AFinal, torch.Tensor):
        AFinal = torch.tensor(AFinal)
        
    InvA = InvA.to(dtype=torch.complex64, device=device)
    AFinal = AFinal.to(dtype=torch.complex64, device=device)

    # --- 3. Model & Optimizer ---
    # NEW: Initialize the Unrolled ADMM Model instead of just SimpleUNet
    model = UnrolledADMM(num_iterations=args.admm_iters).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.L1Loss() 
    
    best_psnr = 0.0

    # --- 4. Training Loop ---
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        
        for gts in train_loader:
            gts = gts.to(device) 
            
            with torch.no_grad():
                noise_lvl = random.uniform(args.noise_min, args.noise_max)
                noise = torch.randn_like(gts) * noise_lvl
                noisy_gts = gts + noise
                
                complex_img = noisy_gts.squeeze(1).to(dtype=torch.complex64) * 1j
                
                # y_meas (degraded_data) is kept to pass into the ADMM network
                degraded_data = torch.matmul(AFinal, complex_img)
                corr_data = torch.matmul(InvA, degraded_data)
                
                rough = corr_data.abs().unsqueeze(1)
                rough = (rough - rough.min()) / (rough.max() - rough.min() + 1e-8)

            optimizer.zero_grad()
            
            # NEW: Pass both measurements and matrices to the unrolled network
            recons = model(y_meas=degraded_data, rough_input=rough, AFinal=AFinal, InvA=InvA)
            
            loss = criterion(recons, gts)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
        avg_train_loss = epoch_loss / len(train_loader)
        
        # --- 5. Validation & Evaluation ---
        model.eval()
        val_loss = 0.0
        val_psnrs, val_ssims = [], []
        
        first_batch = True 
        val_noise_lvl = (args.noise_min + args.noise_max) / 2.0
        
        with torch.no_grad():
            for gts in val_loader:
                gts = gts.to(device)
                
                noise = torch.randn_like(gts) * val_noise_lvl
                noisy_gts = gts + noise
                
                complex_img = noisy_gts.squeeze(1).to(dtype=torch.complex64) * 1j
                degraded_data = torch.matmul(AFinal, complex_img)
                corr_data = torch.matmul(InvA, degraded_data)
                
                rough = corr_data.abs().unsqueeze(1)
                rough = (rough - rough.min()) / (rough.max() - rough.min() + 1e-8)
                
                # NEW: Pass both measurements and matrices to the unrolled network
                recons = model(y_meas=degraded_data, rough_input=rough, AFinal=AFinal, InvA=InvA)
                
                val_loss += criterion(recons, gts).item()
                
                gts_np = gts.cpu().numpy().squeeze(1)     
                rough_np = rough.cpu().numpy().squeeze(1) 
                recons_np = recons.cpu().numpy().squeeze(1) 
                
                batch_rough_psnrs = []
                batch_rough_ssims = []
                batch_final_psnrs = []
                batch_final_ssims = []
                
                for i in range(gts_np.shape[0]):
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
                
                if first_batch:
                    plot_save_path = os.path.join(img_dir, f"epoch_{epoch:03d}.png") 
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
        
        avg_val_psnr = np.mean(val_psnrs)
        std_val_psnr = np.std(val_psnrs)
        avg_val_ssim = np.mean(val_ssims)
        std_val_ssim = np.std(val_ssims)
        
        # --- 6. Logging and Checkpointing ---
        log_msg = (f"Epoch [{epoch:03d}/{args.epochs}] | "
                   f"Train L1: {avg_train_loss:.4f} | Val L1: {avg_val_loss:.4f} | "
                   f"Val PSNR: {avg_val_psnr:.2f} ± {std_val_psnr:.2f} | "
                   f"Val SSIM: {avg_val_ssim:.4f} ± {std_val_ssim:.4f}")
        logging.info(log_msg)
        
        last_ckpt_path = os.path.join(ckpt_dir, "last.pth")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, last_ckpt_path)
        
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