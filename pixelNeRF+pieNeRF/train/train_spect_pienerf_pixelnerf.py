"""
Minimal training script for SpectPixelPieNeRF.

Loads SpectDataset, trains with MSE(AP/PA), logs to stdout, saves visuals and checkpoints.
"""

import argparse
import os
import sys
from pathlib import Path

import matplotlib

# Headless backend for PNG saving.
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402
from torch import nn, optim  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402


def _add_repo_paths():
    """Ensure repository root and pixel-nerf sources are importable."""
    root = Path(__file__).resolve().parents[1]
    pixelnerf_root = root / "pixel-nerf"
    pixelnerf_src = pixelnerf_root / "src"
    # Prioritize the combined project root so our local `model` wins over pixel-nerf's.
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    for p in (pixelnerf_root, pixelnerf_src):
        p_str = str(p)
        if p_str not in sys.path:
            sys.path.append(p_str)


_add_repo_paths()

from src.data.SpectDataset import SpectDataset  # noqa: E402
from model.spect_pixel_pienerf import SpectPixelPieNeRF  # noqa: E402


def _resolve_data_root(data_root: str) -> Path:
    """Resolve data_root with a few fallbacks for convenience."""
    root = Path(__file__).resolve().parents[1]
    candidates = [
        Path(data_root),
        root / data_root,
        root.parent / "pieNeRF" / "data",  # common sibling path
    ]
    for c in candidates:
        if c.exists():
            return c
    tried = "\n".join(str(c) for c in candidates)
    raise FileNotFoundError(f"Could not find data_root. Tried:\n{tried}")


def parse_args():
    parser = argparse.ArgumentParser(description="Train SpectPixelPieNeRF (simple loop).")
    parser.add_argument("--data_root", type=str, default="thesis_med/pieNeRF/data")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--vis_interval", type=int, default=100)
    parser.add_argument("--ckpt_interval", type=int, default=200)
    parser.add_argument("--run_name", type=str, default="spect_pienerf_pixelnerf")
    parser.add_argument(
        "--target_hw",
        type=int,
        nargs=2,
        default=[64, 128],
        metavar=("H", "W"),
        help="Resize AP/PA to (H,W) to control memory. Defaults to 64x128 for lighter training.",
    )
    parser.add_argument(
        "--target_depth",
        type=int,
        default=64,
        help="Resize CT/ACT depth to control memory. Default 64 for lighter training.",
    )
    return parser.parse_args()


def ensure_dirs(run_name: str):
    log_dir = Path("logs") / run_name
    ckpt_dir = Path("checkpoints") / run_name
    vis_dir = Path("visuals") / run_name
    for d in (log_dir, ckpt_dir, vis_dir):
        d.mkdir(parents=True, exist_ok=True)
    return log_dir, ckpt_dir, vis_dir


def get_dataloader(data_root: str, batch_size: int, target_hw: tuple[int, int], target_depth: int):
    resolved_root = _resolve_data_root(data_root)
    print(f"Using data_root: {resolved_root}")
    dataset = SpectDataset(
        datadir=str(resolved_root),
        stage="train",
        target_hw=target_hw,
        target_depth=target_depth,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    first = next(iter(loader))
    print(f"SpectDataset batch keys: {list(first.keys())}")
    print(f"Batch shapes: ct {tuple(first['ct'].shape)}, ap {tuple(first['ap'].shape)}, pa {tuple(first['pa'].shape)}")
    return dataset, loader, first


def make_model(first_batch, device: torch.device):
    ct = first_batch["ct"]
    # ct comes as (B, D, H, W); extract volume shape (D, H, W).
    if ct.dim() == 4:
        volume_shape = tuple(ct.shape[1:])
    elif ct.dim() == 5:
        volume_shape = tuple(ct.shape[2:])
    else:
        raise ValueError(f"Unexpected CT shape: {ct.shape}")

    model = SpectPixelPieNeRF(
        volume_shape=volume_shape,
        voxel_size=1.0,
        num_samples=volume_shape[1],
        latent_dim=512,
        mlp_width=256,
        mlp_depth=8,
        multires_xyz=6,
    ).to(device)
    return model


def save_visual(ap, pa, ap_pred, pa_pred, vis_path: Path):
    def _to_np(x):
        x = x.detach().cpu().squeeze()
        return x.numpy()

    ap_np = _to_np(ap)
    pa_np = _to_np(pa)
    ap_pred_np = _to_np(ap_pred)
    pa_pred_np = _to_np(pa_pred)

    fig, axes = plt.subplots(2, 2, figsize=(8, 8))
    axes[0, 0].imshow(ap_np, cmap="gray")
    axes[0, 0].set_title("AP GT")
    axes[0, 1].imshow(pa_np, cmap="gray")
    axes[0, 1].set_title("PA GT")
    axes[1, 0].imshow(ap_pred_np, cmap="gray")
    axes[1, 0].set_title("AP Pred")
    axes[1, 1].imshow(pa_pred_np, cmap="gray")
    axes[1, 1].set_title("PA Pred")
    for ax in axes.ravel():
        ax.axis("off")
    plt.tight_layout()
    fig.savefig(vis_path, dpi=150)
    plt.close(fig)


def train():
    args = parse_args()
    log_dir, ckpt_dir, vis_dir = ensure_dirs(args.run_name)
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    target_hw = (args.target_hw[0], args.target_hw[1])
    dataset, loader, first_batch = get_dataloader(args.data_root, args.batch_size, target_hw, args.target_depth)
    model = make_model(first_batch, device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    global_step = 0
    for epoch in range(args.epochs):
        model.train()
        for batch in loader:
            ct = batch["ct"]
            ap = batch["ap"]
            pa = batch["pa"]

            # Ensure channel dims exist.
            if ct.dim() == 4:
                ct = ct.unsqueeze(1)
            if ap.dim() == 3:
                ap = ap.unsqueeze(1)
            if pa.dim() == 3:
                pa = pa.unsqueeze(1)

            ct = ct.to(device, non_blocking=True)
            ap = ap.to(device, non_blocking=True)
            pa = pa.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            out = model(ct, ap, pa)

            ap_pred = out["ap_pred"]
            pa_pred = out["pa_pred"]

            loss_ap = criterion(ap_pred, ap)
            loss_pa = criterion(pa_pred, pa)
            loss = loss_ap + loss_pa

            loss.backward()
            optimizer.step()

            if global_step % args.log_interval == 0:
                print(
                    f"[epoch {epoch} step {global_step}] "
                    f"loss={loss.item():.6f} ap={loss_ap.item():.6f} pa={loss_pa.item():.6f}"
                )

            if global_step % args.vis_interval == 0:
                vis_path = vis_dir / f"step_{global_step:06d}.png"
                save_visual(ap[0], pa[0], ap_pred[0], pa_pred[0], vis_path)
                print(f"Saved visualization to {vis_path}")

            if global_step % args.ckpt_interval == 0 and global_step > 0:
                ckpt_path = ckpt_dir / f"step_{global_step:06d}.pt"
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "epoch": epoch,
                        "global_step": global_step,
                    },
                    ckpt_path,
                )
                print(f"Saved checkpoint to {ckpt_path}")

            global_step += 1

    print("Training completed.")


def main():
    train()


if __name__ == "__main__":
    main()
