"""
cd /home/data1/musong/workspace/python/spen_recons
source .venv/bin/activate
export PYTHONPATH=$PYTHONPATH:.
CUDA_VISIBLE_DEVICES=1 python3 script/0415_unsupervised_train.py
"""

import argparse
import glob
import logging
import os
import random
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from torch.utils.data import DataLoader, Dataset, random_split
from torchmetrics.image import PeakSignalNoiseRatio
from torchmetrics.image import StructuralSimilarityIndexMeasure as SSIM

from spenpy.spen import spen

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class RatHRDataset(Dataset):
    def __init__(self, data_dir, image_size=96):
        self.paths = sorted(glob.glob(os.path.join(data_dir, "*.png")))
        self.transform = T.Compose([
            T.Grayscale(num_output_channels=1),
            T.Resize((image_size, image_size)),
            T.ToTensor(),
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        image = Image.open(self.paths[idx]).convert("L")
        return self.transform(image)


class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
        )

    def forward(self, x):
        return x + self.net(x)


class PhasePredictor(nn.Module):
    def __init__(self, base_channels=64, num_blocks=5):
        super().__init__()
        layers = [
            nn.Conv2d(1, base_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        ]
        for _ in range(num_blocks):
            layers.append(ResBlock(base_channels))
        layers.extend([
            nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_channels, 1, kernel_size=(4, 3), stride=(2, 1), padding=(1, 1)),
        ])
        self.net = nn.Sequential(*layers)

    def forward(self, recon):
        magnitude = recon.abs()
        phase_map = np.pi * torch.tanh(self.net(magnitude))
        return phase_map


class SPENPhaseCorrectionNet(nn.Module):
    def __init__(self, inv_a, a_final):
        super().__init__()
        self.phase_predictor = PhasePredictor()
        self.register_buffer("InvA", inv_a)
        self.register_buffer("AFinal", a_final)

    def forward(self, ghosted_img):
        recon = torch.matmul(self.InvA.unsqueeze(0), ghosted_img.squeeze(1)).unsqueeze(1)
        pred_phase = self.phase_predictor(recon)
        corrected_img = apply_phase_correction(ghosted_img, pred_phase)
        reproj = torch.matmul(self.AFinal.unsqueeze(0), corrected_img.squeeze(1)).unsqueeze(1)
        return pred_phase, corrected_img, recon, reproj



def apply_phase_correction(ghosted_img, phase_map):
    odd_lines = ghosted_img[:, :, 0::2, :]
    even_lines = ghosted_img[:, :, 1::2, :]
    corrected_even = even_lines * torch.exp(-1j * phase_map.to(torch.complex64))
    return torch.stack((odd_lines, corrected_even), dim=3).reshape_as(ghosted_img)



def phase_smoothness_loss(phase_map):
    dy = phase_map[:, :, 1:, :] - phase_map[:, :, :-1, :]
    dx = phase_map[:, :, :, 1:] - phase_map[:, :, :, :-1]
    return dy.abs().mean() + dx.abs().mean()



def phase_centering_loss(phase_map):
    return phase_map.mean(dim=(-2, -1)).abs().mean()



def corrected_even_odd_loss(corrected_img):
    odd = corrected_img[:, :, 0::2, :]
    even = corrected_img[:, :, 1::2, :]
    return torch.mean(torch.abs(odd - even))



def measurement_consistency_loss(reproj, corrected_img):
    return torch.mean(torch.abs(reproj - corrected_img))



def image_focus_loss(recon_abs):
    batch = recon_abs / recon_abs.amax(dim=(-2, -1), keepdim=True).clamp_min(1e-6)
    flat = batch.reshape(batch.shape[0], -1)
    prob = flat / flat.sum(dim=1, keepdim=True).clamp_min(1e-6)
    entropy = -(prob * torch.log(prob.clamp_min(1e-6))).sum(dim=1)
    return entropy.mean()



def normalize_for_display(batch):
    batch = batch.detach().cpu()
    mins = batch.amin(dim=(-2, -1), keepdim=True)
    maxs = batch.amax(dim=(-2, -1), keepdim=True)
    return (batch - mins) / (maxs - mins + 1e-8)



def compute_batch_metrics(pred, target, psnr_metric, ssim_metric):
    pred = pred.clamp(0, 1)
    target = target.clamp(0, 1)
    return psnr_metric(pred, target).item(), ssim_metric(pred, target).item()



def compute_image_metrics(pred, target):
    pred_np = pred.detach().cpu().numpy()
    target_np = target.detach().cpu().numpy()
    psnrs, ssims = [], []
    for pred_img, target_img in zip(pred_np, target_np):
        psnrs.append(peak_signal_noise_ratio(target_img, pred_img, data_range=1.0))
        ssims.append(structural_similarity(target_img, pred_img, data_range=1.0))
    return psnrs, ssims



def plot_progress(
    ghosted, reproj, recon, good_img, gt,
    ghosted_psnrs, ghosted_ssims,
    reproj_psnrs, reproj_ssims,
    recon_psnrs, recon_ssims,
    good_psnrs, good_ssims,
    save_path,
):
    cols = min(4, ghosted.shape[0])
    fig, axes = plt.subplots(5, cols, figsize=(3 * cols, 15), gridspec_kw={"wspace": 0.02, "hspace": 0.02})
    if cols == 1:
        axes = axes[:, None]

    ghosted_display = normalize_for_display(ghosted[:cols].squeeze(1)).numpy()
    reproj_display = normalize_for_display(reproj[:cols].squeeze(1)).numpy()
    recon_display = normalize_for_display(recon[:cols].squeeze(1)).numpy()
    good_display = normalize_for_display(good_img[:cols].squeeze(1)).numpy()
    gt_display = normalize_for_display(gt[:cols].squeeze(1)).numpy()

    row_labels = ["Ghosted", "Reproj LR", "Recon HR", "Good Img", "GT"]
    row_colors = ["white", "lime", "yellow", "pink", "cyan"]
    row_metrics = [
        (ghosted_psnrs, ghosted_ssims, "white"),
        (reproj_psnrs, reproj_ssims, "lime"),
        (recon_psnrs, recon_ssims, "yellow"),
        (good_psnrs, good_ssims, "pink"),
        None,
    ]

    for i in range(cols):
        axes[0, i].imshow(ghosted_display[i], cmap="gray", vmin=0, vmax=1)
        axes[1, i].imshow(reproj_display[i], cmap="gray", vmin=0, vmax=1)
        axes[2, i].imshow(recon_display[i], cmap="gray", vmin=0, vmax=1)
        axes[3, i].imshow(good_display[i], cmap="gray", vmin=0, vmax=1)
        axes[4, i].imshow(gt_display[i], cmap="gray", vmin=0, vmax=1)

        axes[0, i].text(0.95, 0.95, f"#{i + 1}", color="white", fontsize=12, fontweight="bold", ha="right", va="top", transform=axes[0, i].transAxes)

        for row in range(4):
            psnrs, ssims, color = row_metrics[row]
            axes[row, i].text(
                0.05, 0.95, f"PSNR: {psnrs[i]:.2f}\nSSIM: {ssims[i]:.3f}",
                color=color, fontsize=10, fontweight="bold", ha="left", va="top",
                transform=axes[row, i].transAxes,
            )

        if i == 0:
            for row in range(5):
                axes[row, i].text(0.05, 0.05, row_labels[row], color=row_colors[row], fontsize=12, fontweight="bold", ha="left", va="bottom", transform=axes[row, i].transAxes)

        for row in range(5):
            axes[row, i].set_xticks([])
            axes[row, i].set_yticks([])

    plt.savefig(save_path, bbox_inches="tight", pad_inches=0.05, dpi=200)
    plt.close(fig)



def setup_logger(log_dir):
    logger = logging.getLogger("SPENUnsupervised0415")
    logger.setLevel(logging.INFO)
    if logger.hasHandlers():
        logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    file_handler = logging.FileHandler(os.path.join(log_dir, "training.log"))
    file_handler.setFormatter(formatter)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger



def parse_args():
    parser = argparse.ArgumentParser(description="Unsupervised SPEN phase-map training with measurement-cycle consistency")
    parser.add_argument("--data_dir", type=str, default="/home/data1/musong/workspace/python/spen_recons/data/0325_rat/hr")
    parser.add_argument("--image_size", type=int, default=96)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--exp_name", type=str, default="spen_unsupervised_0415")
    parser.add_argument("--log_dir", type=str, default="log")
    parser.add_argument("--dc_weight", type=float, default=0.01)       # Dropped by 100x
    parser.add_argument("--evenodd_weight", type=float, default=1.0)   # Increased by 10x
    parser.add_argument("--smoothness_weight", type=float, default=0.01) 
    parser.add_argument("--centering_weight", type=float, default=0.005)
    parser.add_argument("--focus_weight", type=float, default=0.1)     # Increased by 50x
    return parser.parse_args()



def run_epoch(model, loader, simulator, optimizer, device, weights, psnr_metric, ssim_metric, image_dir=None, epoch=None):
    is_train = optimizer is not None
    if is_train:
        model.train()
    else:
        model.eval()

    losses = []
    dc_losses = []
    evenodd_losses = []
    smooth_losses = []
    center_losses = []
    focus_losses = []
    psnrs = []
    ssims = []

    for batch_idx, hr in enumerate(loader):
        hr = hr.to(device)
        
        # Unpack the 3 outputs from your modified simulator
        sim_out = simulator.sim(hr.squeeze(1), return_phase_map=True, return_good_image=True)
        ghosted_img = sim_out[0].unsqueeze(1)
        ideal_phase = sim_out[1].unsqueeze(1)
        good_lr_img = sim_out[2].unsqueeze(1)
        
        # ghosted_img = apply_phase_correction(ghosted_img, ideal_phase)

        if is_train:
            optimizer.zero_grad()

        with torch.set_grad_enabled(is_train):
            pred_phase, corrected_img, recon, reproj = model(ghosted_img)
            recon_mag = recon.abs()
            
            # Reconstruct the ideal "good" image using the Inverse A matrix
            good_recon = torch.matmul(model.InvA.unsqueeze(0), good_lr_img.squeeze(1)).unsqueeze(1)
            good_recon_mag = good_recon.abs()

            loss_dc = measurement_consistency_loss(reproj, corrected_img)
            loss_evenodd = corrected_even_odd_loss(corrected_img)
            loss_smooth = phase_smoothness_loss(pred_phase)
            loss_center = phase_centering_loss(pred_phase)
            loss_focus = image_focus_loss(recon_mag)

            loss = (
                weights["dc"] * loss_dc
                + weights["evenodd"] * loss_evenodd
                + weights["smooth"] * loss_smooth
                + weights["center"] * loss_center
                + weights["focus"] * loss_focus
            )

            if is_train:
                loss.backward()
                optimizer.step()

        losses.append(loss.item())
        dc_losses.append(loss_dc.item())
        evenodd_losses.append(loss_evenodd.item())
        smooth_losses.append(loss_smooth.item())
        center_losses.append(loss_center.item())
        focus_losses.append(loss_focus.item())

        batch_psnr, batch_ssim = compute_batch_metrics(recon_mag, hr, psnr_metric, ssim_metric)
        psnrs.append(batch_psnr)
        ssims.append(batch_ssim)

        if (not is_train) and batch_idx == 0 and image_dir is not None and epoch is not None:
            cols = min(4, hr.shape[0])
            
            # Helper function to safely scale tensors to [0, 1] for metric comparison
            def scale_to_01(t):
                return t / t.amax(dim=(-2, -1), keepdim=True).clamp_min(1e-8)

            # Scale everything to [0, 1] before clamping
            ghosted_mag = scale_to_01(ghosted_img.abs()).clamp(0, 1)
            reproj_mag = scale_to_01(reproj.abs()).clamp(0, 1)
            recon_eval = scale_to_01(recon_mag).clamp(0, 1)
            good_eval = scale_to_01(good_recon_mag).clamp(0, 1)  
            
            # GT is already [0, 1] from ToTensor(), but we ensure it here
            gt_eval = hr.clamp(0, 1) 
            
            ghosted_psnrs, ghosted_ssims = compute_image_metrics(ghosted_mag[:cols].squeeze(1), gt_eval[:cols].squeeze(1))
            reproj_psnrs, reproj_ssims = compute_image_metrics(reproj_mag[:cols].squeeze(1), gt_eval[:cols].squeeze(1))
            recon_psnrs, recon_ssims = compute_image_metrics(recon_eval[:cols].squeeze(1), gt_eval[:cols].squeeze(1))
            good_psnrs, good_ssims = compute_image_metrics(good_eval[:cols].squeeze(1), gt_eval[:cols].squeeze(1)) 
            
            plot_progress(
                ghosted=ghosted_mag,
                reproj=reproj_mag,
                recon=recon_eval,
                good_img=good_eval,
                gt=gt_eval,
                ghosted_psnrs=ghosted_psnrs,
                ghosted_ssims=ghosted_ssims,
                reproj_psnrs=reproj_psnrs,
                reproj_ssims=reproj_ssims,
                recon_psnrs=recon_psnrs,
                recon_ssims=recon_ssims,
                good_psnrs=good_psnrs,
                good_ssims=good_ssims,
                save_path=os.path.join(image_dir, f"epoch_{epoch:03d}.png"),
            )

    return {
        "loss": float(np.mean(losses)),
        "dc": float(np.mean(dc_losses)),
        "evenodd": float(np.mean(evenodd_losses)),
        "smooth": float(np.mean(smooth_losses)),
        "center": float(np.mean(center_losses)),
        "focus": float(np.mean(focus_losses)),
        "psnr": float(np.mean(psnrs)),
        "ssim": float(np.mean(ssims)),
    }



def train(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    timestamp = datetime.now().strftime("%m%d%H%M")
    run_dir = os.path.join(args.log_dir, f"{timestamp}_{args.exp_name}")
    ckpt_dir = os.path.join(run_dir, "ckpt")
    image_dir = os.path.join(run_dir, "images")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(image_dir, exist_ok=True)

    logger = setup_logger(run_dir)
    logger.info(f"Arguments: {vars(args)}")

    dataset = RatHRDataset(args.data_dir, image_size=args.image_size)
    train_size = int(len(dataset) * args.train_ratio)
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    logger.info(f"Dataset split: {train_size} train | {val_size} val")

    simulator = spen(acq_point=(args.image_size, args.image_size), device=device)
    inv_a, a_final = simulator.get_InvA()
    inv_a = inv_a.to(device)
    a_final = a_final.to(device)

    model = SPENPhaseCorrectionNet(inv_a, a_final).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(device)
    ssim_metric = SSIM(data_range=1.0).to(device)

    weights = {
        "dc": args.dc_weight,
        "evenodd": args.evenodd_weight,
        "smooth": args.smoothness_weight,
        "center": args.centering_weight,
        "focus": args.focus_weight,
    }

    best_val_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        train_stats = run_epoch(
            model=model,
            loader=train_loader,
            simulator=simulator,
            optimizer=optimizer,
            device=device,
            weights=weights,
            psnr_metric=psnr_metric,
            ssim_metric=ssim_metric,
        )

        val_stats = run_epoch(
            model=model,
            loader=val_loader,
            simulator=simulator,
            optimizer=None,
            device=device,
            weights=weights,
            psnr_metric=psnr_metric,
            ssim_metric=ssim_metric,
            image_dir=image_dir,
            epoch=epoch,
        )

        logger.info(
            f"Epoch [{epoch:03d}/{args.epochs}] | "
            f"Train loss: {train_stats['loss']:.4f} (dc {train_stats['dc']:.4f}, eo {train_stats['evenodd']:.4f}, sm {train_stats['smooth']:.4f}, ctr {train_stats['center']:.4f}, foc {train_stats['focus']:.4f}) | "
            f"Train PSNR: {train_stats['psnr']:.4f} | Train SSIM: {train_stats['ssim']:.4f} | "
            f"Val loss: {val_stats['loss']:.4f} (dc {val_stats['dc']:.4f}, eo {val_stats['evenodd']:.4f}, sm {val_stats['smooth']:.4f}, ctr {val_stats['center']:.4f}, foc {val_stats['focus']:.4f}) | "
            f"Val PSNR: {val_stats['psnr']:.4f} | Val SSIM: {val_stats['ssim']:.4f}"
        )

        torch.save(model.state_dict(), os.path.join(ckpt_dir, "last.pth"))
        if val_stats["loss"] < best_val_loss:
            best_val_loss = val_stats["loss"]
            torch.save(model.state_dict(), os.path.join(ckpt_dir, "best.pth"))
            logger.info(f"Saved new best checkpoint with val loss {best_val_loss:.4f}")


if __name__ == "__main__":
    args = parse_args()
    train(args)