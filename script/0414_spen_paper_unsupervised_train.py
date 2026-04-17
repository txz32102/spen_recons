"""
cd /home/data1/musong/workspace/python/spen_recons
source .venv/bin/activate
export PYTHONPATH=$PYTHONPATH:.
CUDA_VISIBLE_DEVICES=6 python3 script/0414_spen_paper_unsupervised_train.py
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
    def __init__(self, base_channels=64, num_blocks=4):
        super().__init__()
        layers = [
            nn.Conv2d(2, base_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        ]
        for _ in range(num_blocks):
            layers.append(ResBlock(base_channels))
        layers.append(nn.Conv2d(base_channels, 1, kernel_size=(4, 3), stride=(2, 1), padding=(1, 1)))
        self.net = nn.Sequential(*layers)

    def forward(self, ghosted_img):
        x = torch.cat([ghosted_img.real, ghosted_img.imag], dim=1)
        return self.net(x)


class SPENPhaseCorrectionNet(nn.Module):
    def __init__(self, inv_a):
        super().__init__()
        self.phase_predictor = PhasePredictor()
        self.register_buffer("InvA", inv_a)

    def forward(self, ghosted_img):
        pred_phase = self.phase_predictor(ghosted_img)
        corrected_img = apply_phase_correction(ghosted_img, pred_phase)
        recon = torch.matmul(self.InvA, corrected_img)
        return pred_phase, corrected_img, recon


def apply_phase_correction(ghosted_img, phase_map):
    phase_factor = torch.exp(-1j * phase_map.to(torch.complex64))
    odd_lines = ghosted_img[:, :, 0::2, :]
    even_lines = ghosted_img[:, :, 1::2, :] * phase_factor
    corrected = torch.zeros_like(ghosted_img)
    corrected[:, :, 0::2, :] = odd_lines
    corrected[:, :, 1::2, :] = even_lines
    return corrected


def image_entropy_loss(recon_abs, eps=1e-8):
    flat = recon_abs.reshape(recon_abs.shape[0], -1)
    prob = flat / (flat.sum(dim=1, keepdim=True) + eps)
    entropy = -(prob * torch.log(prob + eps)).sum(dim=1)
    return entropy.mean()


def even_odd_consistency_loss(corrected_img):
    odd = corrected_img[:, :, 0::2, :].abs()
    even = corrected_img[:, :, 1::2, :].abs()
    return torch.mean(torch.abs(odd - even))


def phase_smoothness_loss(phase_map):
    dy = phase_map[:, :, 1:, :] - phase_map[:, :, :-1, :]
    dx = phase_map[:, :, :, 1:] - phase_map[:, :, :, :-1]
    return dy.abs().mean() + dx.abs().mean()


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


def plot_progress(ghosted, corrected, recon, gt, ghosted_psnrs, ghosted_ssims, recon_psnrs, recon_ssims, save_path):
    cols = min(4, ghosted.shape[0])
    fig, axes = plt.subplots(4, cols, figsize=(3 * cols, 12), gridspec_kw={"wspace": 0.02, "hspace": 0.02})
    if cols == 1:
        axes = axes[:, None]

    ghosted_display = normalize_for_display(ghosted[:cols].squeeze(1)).numpy()
    corrected_display = normalize_for_display(corrected[:cols].squeeze(1)).numpy()
    recon_display = normalize_for_display(recon[:cols].squeeze(1)).numpy()
    gt_display = normalize_for_display(gt[:cols].squeeze(1)).numpy()

    row_labels = ["Ghosted", "Corrected", "Recon", "GT"]
    row_colors = ["white", "orange", "yellow", "cyan"]

    for i in range(cols):
        axes[0, i].imshow(ghosted_display[i], cmap="gray", vmin=0, vmax=1)
        axes[1, i].imshow(corrected_display[i], cmap="gray", vmin=0, vmax=1)
        axes[2, i].imshow(recon_display[i], cmap="gray", vmin=0, vmax=1)
        axes[3, i].imshow(gt_display[i], cmap="gray", vmin=0, vmax=1)

        axes[0, i].text(0.05, 0.95, f"#{i + 1}", color="white", fontsize=12, fontweight="bold", ha="left", va="top", transform=axes[0, i].transAxes)
        axes[0, i].text(0.05, 0.20, f"PSNR: {ghosted_psnrs[i]:.2f}\nSSIM: {ghosted_ssims[i]:.3f}", color="white", fontsize=10, fontweight="bold", ha="left", va="top", transform=axes[0, i].transAxes)
        axes[2, i].text(0.05, 0.95, f"PSNR: {recon_psnrs[i]:.2f}\nSSIM: {recon_ssims[i]:.3f}", color="yellow", fontsize=10, fontweight="bold", ha="left", va="top", transform=axes[2, i].transAxes)

        if i == 0:
            for row in range(4):
                axes[row, i].text(0.05, 0.05, row_labels[row], color=row_colors[row], fontsize=12, fontweight="bold", ha="left", va="bottom", transform=axes[row, i].transAxes)

        for row in range(4):
            axes[row, i].set_xticks([])
            axes[row, i].set_yticks([])

    plt.savefig(save_path, bbox_inches="tight", pad_inches=0.05, dpi=200)
    plt.close(fig)


def setup_logger(log_dir):
    logger = logging.getLogger("SPENPaperUnsupervised")
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
    parser = argparse.ArgumentParser(description="Paper-style unsupervised SPEN phase correction")
    parser.add_argument("--data_dir", type=str, default="/home/data1/musong/workspace/python/spen_recons/data/0325_rat/hr")
    parser.add_argument("--image_size", type=int, default=96)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--exp_name", type=str, default="spen_paper_unsupervised")
    parser.add_argument("--log_dir", type=str, default="log")
    parser.add_argument("--entropy_weight", type=float, default=1.0)
    parser.add_argument("--consistency_weight", type=float, default=0.2)
    parser.add_argument("--smoothness_weight", type=float, default=0.01)
    return parser.parse_args()


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

    inv_a, _ = spen(acq_point=(args.image_size, args.image_size), device=device).get_InvA()
    inv_a = inv_a.to(device)
    simulator = spen(acq_point=(args.image_size, args.image_size), device=device)

    model = SPENPhaseCorrectionNet(inv_a).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(device)
    ssim_metric = SSIM(data_range=1.0).to(device)

    best_val_loss = float("inf")

    for epoch in range(args.epochs):
        model.train()
        train_losses = []
        train_psnrs = []
        train_ssims = []

        for hr in train_loader:
            hr = hr.to(device)
            ghosted_img, _, measured_good = simulator.sim(hr.squeeze(1), return_phase_map=True, return_good_image=True)
            ghosted_img = ghosted_img.unsqueeze(1)
            measured_good = measured_good.unsqueeze(1)

            optimizer.zero_grad()
            pred_phase, corrected_img, recon = model(ghosted_img)

            recon_mag = recon.abs()
            entropy = image_entropy_loss(recon_mag)
            consistency = even_odd_consistency_loss(corrected_img)
            smoothness = phase_smoothness_loss(pred_phase)
            dc = torch.mean(torch.abs(recon_mag - measured_good.abs()))

            loss = (
                args.entropy_weight * entropy
                + args.consistency_weight * consistency
                + args.smoothness_weight * smoothness
                + dc
            )
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

            train_psnr, train_ssim = compute_batch_metrics(recon_mag, hr, psnr_metric, ssim_metric)
            train_psnrs.append(train_psnr)
            train_ssims.append(train_ssim)

        avg_train_loss = float(np.mean(train_losses))
        avg_train_psnr = float(np.mean(train_psnrs))
        avg_train_ssim = float(np.mean(train_ssims))

        model.eval()
        val_losses = []
        val_psnrs = []
        val_ssims = []
        with torch.no_grad():
            for batch_idx, hr in enumerate(val_loader):
                hr = hr.to(device)
                ghosted_img, _, measured_good = simulator.sim(hr.squeeze(1), return_phase_map=True, return_good_image=True)
                ghosted_img = ghosted_img.unsqueeze(1)
                measured_good = measured_good.unsqueeze(1)

                pred_phase, corrected_img, recon = model(ghosted_img)
                recon_mag = recon.abs()
                entropy = image_entropy_loss(recon_mag)
                consistency = even_odd_consistency_loss(corrected_img)
                smoothness = phase_smoothness_loss(pred_phase)
                dc = torch.mean(torch.abs(recon_mag - measured_good.abs()))
                loss = (
                    args.entropy_weight * entropy
                    + args.consistency_weight * consistency
                    + args.smoothness_weight * smoothness
                    + dc
                )
                val_losses.append(loss.item())

                val_psnr, val_ssim = compute_batch_metrics(recon_mag, hr, psnr_metric, ssim_metric)
                val_psnrs.append(val_psnr)
                val_ssims.append(val_ssim)

                if batch_idx == 0:
                    cols = min(4, hr.shape[0])
                    ghosted_mag = ghosted_img.abs().clamp(0, 1)
                    corrected_mag = corrected_img.abs().clamp(0, 1)
                    recon_eval = recon_mag.clamp(0, 1)
                    gt_eval = hr.clamp(0, 1)
                    ghosted_psnrs, ghosted_ssims = compute_image_metrics(ghosted_mag[:cols].squeeze(1), gt_eval[:cols].squeeze(1))
                    recon_psnrs, recon_ssims = compute_image_metrics(recon_eval[:cols].squeeze(1), gt_eval[:cols].squeeze(1))
                    plot_progress(
                        ghosted=ghosted_mag,
                        corrected=corrected_mag,
                        recon=recon_eval,
                        gt=gt_eval,
                        ghosted_psnrs=ghosted_psnrs,
                        ghosted_ssims=ghosted_ssims,
                        recon_psnrs=recon_psnrs,
                        recon_ssims=recon_ssims,
                        save_path=os.path.join(image_dir, f"epoch_{epoch + 1:03d}.png"),
                    )

        avg_val_loss = float(np.mean(val_losses))
        avg_val_psnr = float(np.mean(val_psnrs))
        avg_val_ssim = float(np.mean(val_ssims))
        logger.info(
            f"Epoch [{epoch + 1:03d}/{args.epochs}] | "
            f"Train loss: {avg_train_loss:.4f} | Train PSNR: {avg_train_psnr:.4f} | Train SSIM: {avg_train_ssim:.4f} | "
            f"Val loss: {avg_val_loss:.4f} | Val PSNR: {avg_val_psnr:.4f} | Val SSIM: {avg_val_ssim:.4f}"
        )

        torch.save(model.state_dict(), os.path.join(ckpt_dir, "last.pth"))
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), os.path.join(ckpt_dir, "best.pth"))
            logger.info(f"Saved new best checkpoint with val loss {best_val_loss:.4f}")


if __name__ == "__main__":
    args = parse_args()
    train(args)