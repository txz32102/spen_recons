"""
cd /home/data1/musong/workspace/python/spen_recons
source .venv/bin/activate
export PYTHONPATH=$PYTHONPATH:.
CUDA_VISIBLE_DEVICES=0 python3 script/0417_supervised_train_v1.py
"""

import argparse
import logging
import os
import random
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from torch.utils.data import DataLoader, Dataset, random_split
from torchmetrics.image import PeakSignalNoiseRatio
from torchmetrics.image import StructuralSimilarityIndexMeasure

from spenpy.spen import spen


DATA_DIR = "/home/data1/musong/workspace/python/spen_recons/data/0325_rat/hr"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class RatDataset(Dataset):
    def __init__(self, data_dir: str, image_size: int) -> None:
        self.paths = sorted(Path(data_dir).glob("*.png"))
        self.transform = transforms.Compose(
            [
                transforms.Grayscale(num_output_channels=1),
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
            ]
        )

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> torch.Tensor:
        image = Image.open(self.paths[index]).convert("L")
        return self.transform(image)


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class PhaseMapNet(nn.Module):
    def __init__(self, in_channels: int = 2, base_channels: int = 96, num_blocks: int = 8) -> None:
        super().__init__()
        layers: list[nn.Module] = [
            nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        ]
        for _ in range(num_blocks):
            layers.append(ResidualBlock(base_channels))
        layers.extend(
            [
                nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(base_channels, 1, kernel_size=(4, 3), stride=(2, 1), padding=(1, 1)),
            ]
        )
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.pi * torch.tanh(self.net(x))


class SupervisedPhaseReconModel(nn.Module):
    def __init__(self, inv_a: torch.Tensor) -> None:
        super().__init__()
        self.phase_net = PhaseMapNet()
        self.register_buffer("inv_a", inv_a)

    def reconstruct(self, rofft_data: torch.Tensor) -> torch.Tensor:
        return torch.matmul(self.inv_a.unsqueeze(0), rofft_data.squeeze(1)).unsqueeze(1)

    def forward(self, ghosted_data: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        inva_recon = self.reconstruct(ghosted_data)
        features = torch.cat((inva_recon.real, inva_recon.imag), dim=1)
        pred_phase = self.phase_net(features)
        corrected_data = apply_phase_correction(ghosted_data, pred_phase)
        corrected_recon = self.reconstruct(corrected_data)
        return pred_phase, corrected_data, inva_recon, corrected_recon


def apply_phase_correction(ghosted_data: torch.Tensor, phase_map: torch.Tensor) -> torch.Tensor:
    odd_lines = ghosted_data[:, :, 0::2, :]
    even_lines = ghosted_data[:, :, 1::2, :] * torch.exp(-1j * phase_map.to(torch.complex64))
    return torch.stack((odd_lines, even_lines), dim=3).reshape_as(ghosted_data)


def normalize_real_image(batch: torch.Tensor) -> torch.Tensor:
    mins = batch.amin(dim=(-2, -1), keepdim=True)
    maxs = batch.amax(dim=(-2, -1), keepdim=True)
    return (batch - mins) / (maxs - mins + 1e-8)


def normalize_complex_magnitude(batch: torch.Tensor) -> torch.Tensor:
    return normalize_real_image(batch.abs())


def phase_rmse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(torch.mean((pred - target) ** 2))


def compute_batch_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    psnr_metric: PeakSignalNoiseRatio,
    ssim_metric: StructuralSimilarityIndexMeasure,
) -> tuple[float, float]:
    pred = pred.clamp(0, 1)
    target = target.clamp(0, 1)
    return psnr_metric(pred, target).item(), ssim_metric(pred, target).item()


def compute_image_metrics(pred: torch.Tensor, target: torch.Tensor) -> tuple[list[float], list[float]]:
    pred_np = pred.detach().cpu().numpy()
    target_np = target.detach().cpu().numpy()
    psnrs: list[float] = []
    ssims: list[float] = []
    for pred_image, target_image in zip(pred_np, target_np):
        psnrs.append(peak_signal_noise_ratio(target_image, pred_image, data_range=1.0))
        ssims.append(structural_similarity(target_image, pred_image, data_range=1.0))
    return psnrs, ssims


def annotate_panel(axis, title: str, metric_text: str | None, title_color: str = "white") -> None:
    axis.text(
        0.03,
        0.97,
        title,
        color=title_color,
        fontsize=11,
        fontweight="bold",
        ha="left",
        va="top",
        transform=axis.transAxes,
        bbox={"facecolor": "black", "alpha": 0.5, "pad": 2, "edgecolor": "none"},
    )
    if metric_text is not None:
        axis.text(
            0.97,
            0.03,
            metric_text,
            color="white",
            fontsize=10,
            fontweight="bold",
            ha="right",
            va="bottom",
            transform=axis.transAxes,
            bbox={"facecolor": "black", "alpha": 0.5, "pad": 2, "edgecolor": "none"},
        )
    axis.set_xticks([])
    axis.set_yticks([])


def save_visualization(
    ghosted: torch.Tensor,
    inva_recon: torch.Tensor,
    corrected_recon: torch.Tensor,
    good_recon: torch.Tensor,
    gt: torch.Tensor,
    pred_phase: torch.Tensor,
    target_phase: torch.Tensor,
    save_path: str,
) -> None:
    cols = min(4, ghosted.shape[0])
    rows = 6
    fig, axes = plt.subplots(rows, cols, figsize=(3.5 * cols, 18), squeeze=False)

    ghosted_display = normalize_complex_magnitude(ghosted[:cols].squeeze(1))
    inva_display = normalize_complex_magnitude(inva_recon[:cols].squeeze(1))
    corrected_display = normalize_complex_magnitude(corrected_recon[:cols].squeeze(1))
    good_display = normalize_complex_magnitude(good_recon[:cols].squeeze(1))
    gt_display = normalize_real_image(gt[:cols].squeeze(1))

    ghosted_psnr, ghosted_ssim = compute_image_metrics(ghosted_display, gt_display)
    inva_psnr, inva_ssim = compute_image_metrics(inva_display, gt_display)
    corrected_psnr, corrected_ssim = compute_image_metrics(corrected_display, gt_display)
    good_psnr, good_ssim = compute_image_metrics(good_display, gt_display)

    phase_mae = torch.mean(torch.abs(pred_phase[:cols] - target_phase[:cols]), dim=(1, 2, 3)).detach().cpu().tolist()
    phase_rmse_values = torch.sqrt(torch.mean((pred_phase[:cols] - target_phase[:cols]) ** 2, dim=(1, 2, 3))).detach().cpu().tolist()

    image_rows = [
        (ghosted_display.detach().cpu().numpy(), "Ghosted", ghosted_psnr, ghosted_ssim, "gray"),
        (inva_display.detach().cpu().numpy(), "InvA Recon", inva_psnr, inva_ssim, "gray"),
        (corrected_display.detach().cpu().numpy(), "Corrected Recon", corrected_psnr, corrected_ssim, "gray"),
        (good_display.detach().cpu().numpy(), "GT-Corrected Recon", good_psnr, good_ssim, "gray"),
        (gt_display.detach().cpu().numpy(), "Ground Truth", [None] * cols, [None] * cols, "gray"),
    ]

    for col in range(cols):
        for row, (images, title, psnrs, ssims, cmap) in enumerate(image_rows):
            axes[row, col].imshow(images[col], cmap=cmap, vmin=0, vmax=1)
            metric_text = None if psnrs[col] is None else f"PSNR {psnrs[col]:.2f}\nSSIM {ssims[col]:.3f}"
            annotate_panel(axes[row, col], title, metric_text)

        axes[5, col].imshow(pred_phase[col, 0].detach().cpu().numpy(), cmap="twilight", vmin=-np.pi, vmax=np.pi)
        annotate_panel(
            axes[5, col],
            "Pred Phase",
            f"MAE {phase_mae[col]:.4f}\nRMSE {phase_rmse_values[col]:.4f}",
            title_color="orange",
        )
        axes[5, col].contour(target_phase[col, 0].detach().cpu().numpy(), levels=8, colors="white", linewidths=0.5, alpha=0.7)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def setup_logger(log_dir: str) -> logging.Logger:
    logger = logging.getLogger("supervised_phase_0417_v1")
    logger.setLevel(logging.INFO)
    if logger.hasHandlers():
        logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    file_handler = logging.FileHandler(os.path.join(log_dir, "training.log"))
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Supervised SPEN image reconstruction with phase-map prediction")
    parser.add_argument("--data_dir", type=str, default=DATA_DIR)
    parser.add_argument("--image_size", type=int, default=96)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--exp_name", type=str, default="supervised_phase_0417_v1")
    parser.add_argument("--log_root", type=str, default="log")
    parser.add_argument("--phase_weight", type=float, default=1.0)
    parser.add_argument("--recon_weight", type=float, default=0.5)
    return parser.parse_args()


def run_epoch(
    model: SupervisedPhaseReconModel,
    loader: DataLoader,
    simulator: spen,
    optimizer: optim.Optimizer | None,
    weights: dict[str, float],
    psnr_metric: PeakSignalNoiseRatio,
    ssim_metric: StructuralSimilarityIndexMeasure,
    image_dir: str | None = None,
    epoch: int | None = None,
) -> dict[str, float]:
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    losses: list[float] = []
    phase_losses: list[float] = []
    recon_losses: list[float] = []
    phase_maes: list[float] = []
    phase_rmses: list[float] = []
    corrected_psnrs: list[float] = []
    corrected_ssims: list[float] = []

    for batch_index, hr in enumerate(loader):
        hr = hr.to(model.inv_a.device)
        ghosted_data, target_phase, good_lr = simulator.sim(
            hr.squeeze(1),
            return_phase_map=True,
            return_good_lr_image=True,
        )
        ghosted_data = ghosted_data.unsqueeze(1)
        target_phase = target_phase.unsqueeze(1)
        good_lr = good_lr.unsqueeze(1)

        if is_train:
            optimizer.zero_grad()

        with torch.set_grad_enabled(is_train):
            pred_phase, _, inva_recon, corrected_recon = model(ghosted_data)
            good_recon = model.reconstruct(good_lr)

            phase_loss = nn.functional.mse_loss(pred_phase, target_phase)
            recon_loss = nn.functional.l1_loss(corrected_recon.abs(), good_recon.abs())
            loss = weights["phase"] * phase_loss + weights["recon"] * recon_loss

            if is_train:
                loss.backward()
                optimizer.step()

        losses.append(loss.item())
        phase_losses.append(phase_loss.item())
        recon_losses.append(recon_loss.item())
        phase_maes.append(torch.mean(torch.abs(pred_phase - target_phase)).item())
        phase_rmses.append(phase_rmse(pred_phase, target_phase).item())

        corrected_eval = normalize_complex_magnitude(corrected_recon)
        good_eval = normalize_complex_magnitude(good_recon)
        batch_psnr, batch_ssim = compute_batch_metrics(corrected_eval, good_eval, psnr_metric, ssim_metric)
        corrected_psnrs.append(batch_psnr)
        corrected_ssims.append(batch_ssim)

        if not is_train and batch_index == 0 and image_dir is not None and epoch is not None:
            save_visualization(
                ghosted=ghosted_data,
                inva_recon=inva_recon,
                corrected_recon=corrected_recon,
                good_recon=good_recon,
                gt=hr,
                pred_phase=pred_phase,
                target_phase=target_phase,
                save_path=os.path.join(image_dir, f"epoch_{epoch:03d}.png"),
            )

    return {
        "loss": float(np.mean(losses)),
        "phase_loss": float(np.mean(phase_losses)),
        "recon_loss": float(np.mean(recon_losses)),
        "phase_mae": float(np.mean(phase_maes)),
        "phase_rmse": float(np.mean(phase_rmses)),
        "corrected_psnr": float(np.mean(corrected_psnrs)),
        "corrected_ssim": float(np.mean(corrected_ssims)),
    }


def train(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    timestamp = datetime.now().strftime("%m%d%H%M")
    run_dir = os.path.join(args.log_root, f"{timestamp}_{args.exp_name}")
    ckpt_dir = os.path.join(run_dir, "ckpt")
    image_dir = os.path.join(run_dir, "images")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(image_dir, exist_ok=True)

    logger = setup_logger(run_dir)
    logger.info("Arguments: %s", vars(args))

    dataset = RatDataset(args.data_dir, args.image_size)
    train_size = int(len(dataset) * args.train_ratio)
    val_size = len(dataset) - train_size
    split_generator = torch.Generator().manual_seed(args.seed)
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=split_generator)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    logger.info("Dataset split: %d train | %d val", train_size, val_size)

    simulator = spen(acq_point=[args.image_size, args.image_size], device=device)
    inv_a, _ = simulator.get_InvA()
    model = SupervisedPhaseReconModel(inv_a.to(device)).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(device)
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)

    weights = {"phase": args.phase_weight, "recon": args.recon_weight}
    best_val_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        train_stats = run_epoch(
            model=model,
            loader=train_loader,
            simulator=simulator,
            optimizer=optimizer,
            weights=weights,
            psnr_metric=psnr_metric,
            ssim_metric=ssim_metric,
        )
        val_stats = run_epoch(
            model=model,
            loader=val_loader,
            simulator=simulator,
            optimizer=None,
            weights=weights,
            psnr_metric=psnr_metric,
            ssim_metric=ssim_metric,
            image_dir=image_dir,
            epoch=epoch,
        )

        logger.info(
            "Epoch [%03d/%03d] | Train loss %.4f | phase %.4f | recon %.4f | phase MAE %.4f | phase RMSE %.4f | corrected PSNR/SSIM %.4f/%.4f | Val loss %.4f | phase %.4f | recon %.4f | phase MAE %.4f | phase RMSE %.4f | corrected PSNR/SSIM %.4f/%.4f",
            epoch,
            args.epochs,
            train_stats["loss"],
            train_stats["phase_loss"],
            train_stats["recon_loss"],
            train_stats["phase_mae"],
            train_stats["phase_rmse"],
            train_stats["corrected_psnr"],
            train_stats["corrected_ssim"],
            val_stats["loss"],
            val_stats["phase_loss"],
            val_stats["recon_loss"],
            val_stats["phase_mae"],
            val_stats["phase_rmse"],
            val_stats["corrected_psnr"],
            val_stats["corrected_ssim"],
        )

        last_ckpt_path = os.path.join(ckpt_dir, "last_ckpt.pth")
        best_ckpt_path = os.path.join(ckpt_dir, "best_ckpt.pth")
        torch.save(model.state_dict(), last_ckpt_path)
        if val_stats["loss"] < best_val_loss:
            best_val_loss = val_stats["loss"]
            torch.save(model.state_dict(), best_ckpt_path)
            logger.info("Saved best checkpoint to %s", best_ckpt_path)


if __name__ == "__main__":
    train(parse_args())
