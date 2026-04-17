"""
cd /home/data1/musong/workspace/python/spen_recons
source .venv/bin/activate
export PYTHONPATH=$PYTHONPATH:.
CUDA_VISIBLE_DEVICES=1 python3 script/0416_supervised_phase_train.py
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
    def __init__(self, in_channels=2, base_channels=128, num_blocks=10):
        super().__init__()
        layers = [
            nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1),
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

    def forward(self, recon_features):
        return np.pi * torch.tanh(self.net(recon_features))


class SupervisedSPENPhaseNet(nn.Module):
    def __init__(self, inv_a):
        super().__init__()
        self.phase_predictor = PhasePredictor()
        self.register_buffer("InvA", inv_a)

    def reconstruct_with_inv_a(self, lowres_img):
        return torch.matmul(self.InvA.unsqueeze(0), lowres_img.squeeze(1)).unsqueeze(1)

    def forward(self, ghosted_img):
        inva_recon = self.reconstruct_with_inv_a(ghosted_img)
        recon_features = torch.cat((inva_recon.real, inva_recon.imag), dim=1)
        pred_phase = self.phase_predictor(recon_features)
        corrected_img = apply_phase_correction(ghosted_img, pred_phase)
        corrected_recon = self.reconstruct_with_inv_a(corrected_img)
        return pred_phase, corrected_img, inva_recon, corrected_recon


def apply_phase_correction(ghosted_img, phase_map):
    odd_lines = ghosted_img[:, :, 0::2, :]
    even_lines = ghosted_img[:, :, 1::2, :]
    corrected_even = even_lines * torch.exp(-1j * phase_map.to(torch.complex64))
    return torch.stack((odd_lines, corrected_even), dim=3).reshape_as(ghosted_img)


def phase_rmse(pred_phase, target_phase):
    return torch.sqrt(torch.mean((pred_phase - target_phase) ** 2))


def normalize_for_display(batch):
    batch = batch.detach().cpu()
    mins = batch.amin(dim=(-2, -1), keepdim=True)
    maxs = batch.amax(dim=(-2, -1), keepdim=True)
    return (batch - mins) / (maxs - mins + 1e-8)


def normalize_for_metrics(batch):
    magnitude = batch.abs() if torch.is_complex(batch) else batch
    mins = magnitude.amin(dim=(-2, -1), keepdim=True)
    maxs = magnitude.amax(dim=(-2, -1), keepdim=True)
    return (magnitude - mins) / (maxs - mins + 1e-8)


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
    ghosted,
    inva_recon,
    corrected_recon,
    good_img,
    gt,
    pred_phase,
    target_phase,
    ghosted_psnrs,
    ghosted_ssims,
    inva_psnrs,
    inva_ssims,
    corrected_psnrs,
    corrected_ssims,
    good_psnrs,
    good_ssims,
    phase_maes,
    phase_rmses,
    save_path,
):
    cols = min(4, ghosted.shape[0])
    fig, axes = plt.subplots(7, cols, figsize=(3 * cols, 21), gridspec_kw={"wspace": 0.02, "hspace": 0.02})
    if cols == 1:
        axes = axes[:, None]

    ghosted_display = normalize_for_display(ghosted[:cols].squeeze(1)).numpy()
    inva_display = normalize_for_display(inva_recon[:cols].squeeze(1)).numpy()
    corrected_display = normalize_for_display(corrected_recon[:cols].squeeze(1)).numpy()
    good_display = normalize_for_display(good_img[:cols].squeeze(1)).numpy()
    gt_display = normalize_for_display(gt[:cols].squeeze(1)).numpy()
    pred_phase_display = pred_phase[:cols].detach().cpu().squeeze(1).numpy()
    target_phase_display = target_phase[:cols].detach().cpu().squeeze(1).numpy()

    row_labels = ["Ghosted", "InvA Recon", "Corrected Recon", "Good Img", "GT", "Pred Phase", "GT Phase"]
    row_colors = ["white", "yellow", "lime", "pink", "cyan", "orange", "orange"]
    row_metrics = [
        (ghosted_psnrs, ghosted_ssims, "white"),
        (inva_psnrs, inva_ssims, "yellow"),
        (corrected_psnrs, corrected_ssims, "lime"),
        (good_psnrs, good_ssims, "pink"),
        None,
        None,
        None,
    ]

    for i in range(cols):
        axes[0, i].imshow(ghosted_display[i], cmap="gray", vmin=0, vmax=1)
        axes[1, i].imshow(inva_display[i], cmap="gray", vmin=0, vmax=1)
        axes[2, i].imshow(corrected_display[i], cmap="gray", vmin=0, vmax=1)
        axes[3, i].imshow(good_display[i], cmap="gray", vmin=0, vmax=1)
        axes[4, i].imshow(gt_display[i], cmap="gray", vmin=0, vmax=1)
        axes[5, i].imshow(pred_phase_display[i], cmap="twilight", vmin=-np.pi, vmax=np.pi)
        axes[6, i].imshow(target_phase_display[i], cmap="twilight", vmin=-np.pi, vmax=np.pi)

        axes[0, i].text(0.95, 0.95, f"#{i + 1}", color="white", fontsize=12, fontweight="bold", ha="right", va="top", transform=axes[0, i].transAxes)

        for row in range(4):
            psnrs, ssims, color = row_metrics[row]
            axes[row, i].text(
                0.05,
                0.95,
                f"PSNR: {psnrs[i]:.2f}\nSSIM: {ssims[i]:.3f}",
                color=color,
                fontsize=10,
                fontweight="bold",
                ha="left",
                va="top",
                transform=axes[row, i].transAxes,
            )

        axes[5, i].text(
            0.05,
            0.95,
            f"MAE: {phase_maes[i]:.4f}\nRMSE: {phase_rmses[i]:.4f}",
            color="orange",
            fontsize=10,
            fontweight="bold",
            ha="left",
            va="top",
            transform=axes[5, i].transAxes,
        )

        if i == 0:
            for row in range(7):
                axes[row, i].text(0.05, 0.05, row_labels[row], color=row_colors[row], fontsize=12, fontweight="bold", ha="left", va="bottom", transform=axes[row, i].transAxes)

        for row in range(7):
            axes[row, i].set_xticks([])
            axes[row, i].set_yticks([])

    plt.savefig(save_path, bbox_inches="tight", pad_inches=0.05, dpi=200)
    plt.close(fig)


def setup_logger(log_dir):
    logger = logging.getLogger("SPENSupervised0416")
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
    parser = argparse.ArgumentParser(description="Supervised SPEN phase-map training using invA reconstruction as input")
    parser.add_argument("--data_dir", type=str, default="/home/data1/musong/workspace/python/spen_recons/data/0325_rat/hr")
    parser.add_argument("--image_size", type=int, default=96)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--exp_name", type=str, default="spen_supervised_phase_0416")
    parser.add_argument("--log_dir", type=str, default="log")
    parser.add_argument("--phase_weight", type=float, default=1.0)
    parser.add_argument("--recon_weight", type=float, default=0.5)
    return parser.parse_args()


def run_epoch(model, loader, simulator, optimizer, device, weights, psnr_metric, ssim_metric, image_dir=None, epoch=None):
    is_train = optimizer is not None
    if is_train:
        model.train()
    else:
        model.eval()

    losses = []
    phase_losses = []
    recon_losses = []
    phase_maes = []
    phase_rmses = []
    corrected_psnrs = []
    corrected_ssims = []

    for batch_idx, hr in enumerate(loader):
        hr = hr.to(device)
        sim_out = simulator.sim(hr.squeeze(1), return_phase_map=True, return_good_image=True)
        ghosted_img = sim_out[0].unsqueeze(1)
        target_phase = sim_out[1].unsqueeze(1)
        good_lr_img = sim_out[2].unsqueeze(1)

        if is_train:
            optimizer.zero_grad()

        with torch.set_grad_enabled(is_train):
            pred_phase, corrected_img, inva_recon, corrected_recon = model(ghosted_img)
            good_recon = model.reconstruct_with_inv_a(good_lr_img)

            loss_phase = nn.functional.mse_loss(pred_phase, target_phase)
            loss_recon = nn.functional.l1_loss(corrected_recon.abs(), good_recon.abs())
            loss = weights["phase"] * loss_phase + weights["recon"] * loss_recon

            if is_train:
                loss.backward()
                optimizer.step()

        losses.append(loss.item())
        phase_losses.append(loss_phase.item())
        recon_losses.append(loss_recon.item())
        phase_maes.append(torch.mean(torch.abs(pred_phase - target_phase)).item())
        phase_rmses.append(phase_rmse(pred_phase, target_phase).item())

        corrected_eval = normalize_for_metrics(corrected_recon)
        good_eval = normalize_for_metrics(good_recon)
        batch_psnr, batch_ssim = compute_batch_metrics(corrected_eval, good_eval, psnr_metric, ssim_metric)
        corrected_psnrs.append(batch_psnr)
        corrected_ssims.append(batch_ssim)

        if (not is_train) and batch_idx == 0 and image_dir is not None and epoch is not None:
            cols = min(4, hr.shape[0])
            ghosted_eval = normalize_for_metrics(ghosted_img)
            inva_eval = normalize_for_metrics(inva_recon)
            corrected_eval_vis = normalize_for_metrics(corrected_recon)
            gt_eval = hr.clamp(0, 1)

            ghosted_psnrs, ghosted_ssims = compute_image_metrics(ghosted_eval[:cols].squeeze(1), gt_eval[:cols].squeeze(1))
            inva_psnrs, inva_ssims = compute_image_metrics(inva_eval[:cols].squeeze(1), gt_eval[:cols].squeeze(1))
            corrected_psnrs_vis, corrected_ssims_vis = compute_image_metrics(corrected_eval_vis[:cols].squeeze(1), gt_eval[:cols].squeeze(1))
            good_psnrs, good_ssims = compute_image_metrics(good_eval[:cols].squeeze(1), gt_eval[:cols].squeeze(1))
            phase_mae_images = torch.mean(torch.abs(pred_phase[:cols] - target_phase[:cols]), dim=(1, 2, 3)).detach().cpu().tolist()
            phase_rmse_images = torch.sqrt(torch.mean((pred_phase[:cols] - target_phase[:cols]) ** 2, dim=(1, 2, 3))).detach().cpu().tolist()

            plot_progress(
                ghosted=ghosted_eval,
                inva_recon=inva_eval,
                corrected_recon=corrected_eval_vis,
                good_img=good_eval,
                gt=gt_eval,
                pred_phase=pred_phase,
                target_phase=target_phase,
                ghosted_psnrs=ghosted_psnrs,
                ghosted_ssims=ghosted_ssims,
                inva_psnrs=inva_psnrs,
                inva_ssims=inva_ssims,
                corrected_psnrs=corrected_psnrs_vis,
                corrected_ssims=corrected_ssims_vis,
                good_psnrs=good_psnrs,
                good_ssims=good_ssims,
                phase_maes=phase_mae_images,
                phase_rmses=phase_rmse_images,
                save_path=os.path.join(image_dir, f"epoch_{epoch:03d}.png"),
            )

    return {
        "loss": float(np.mean(losses)),
        "phase": float(np.mean(phase_losses)),
        "recon": float(np.mean(recon_losses)),
        "phase_mae": float(np.mean(phase_maes)),
        "phase_rmse": float(np.mean(phase_rmses)),
        "corrected_psnr": float(np.mean(corrected_psnrs)),
        "corrected_ssim": float(np.mean(corrected_ssims)),
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
    inv_a, _ = simulator.get_InvA()
    inv_a = inv_a.to(device)

    model = SupervisedSPENPhaseNet(inv_a).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(device)
    ssim_metric = SSIM(data_range=1.0).to(device)

    weights = {
        "phase": args.phase_weight,
        "recon": args.recon_weight,
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
            f"Train loss: {train_stats['loss']:.4f} (phase {train_stats['phase']:.4f}, recon {train_stats['recon']:.4f}) | "
            f"Train phase MAE: {train_stats['phase_mae']:.4f} | Train phase RMSE: {train_stats['phase_rmse']:.4f} | "
            f"Train corrected PSNR/SSIM: {train_stats['corrected_psnr']:.4f}/{train_stats['corrected_ssim']:.4f} | "
            f"Val loss: {val_stats['loss']:.4f} (phase {val_stats['phase']:.4f}, recon {val_stats['recon']:.4f}) | "
            f"Val phase MAE: {val_stats['phase_mae']:.4f} | Val phase RMSE: {val_stats['phase_rmse']:.4f} | "
            f"Val corrected PSNR/SSIM: {val_stats['corrected_psnr']:.4f}/{val_stats['corrected_ssim']:.4f}"
        )

        torch.save(model.state_dict(), os.path.join(ckpt_dir, "last.pth"))
        if val_stats["loss"] < best_val_loss:
            best_val_loss = val_stats["loss"]
            torch.save(model.state_dict(), os.path.join(ckpt_dir, "best.pth"))
            logger.info(f"Saved new best checkpoint with val loss {best_val_loss:.4f}")


if __name__ == "__main__":
    args = parse_args()
    train(args)
