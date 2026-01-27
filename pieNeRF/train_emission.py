"""Mini-training script for the SPECT emission NeRF."""
import argparse
import csv
import math
import logging
import signal
import subprocess
import time
import traceback
from pathlib import Path
from typing import Optional, Tuple, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

from torch.utils.data import DataLoader

from graf.config import get_data, build_models
from graf.encoders import ProjectionEncoder
from utils.ray_split import (
    PixelSplit,
    make_pixel_split_from_ap_pa,
    make_pixel_split_stratified_intensity,
    sample_train_indices,
)

__VERSION__ = "emission-train v0.3"
DEBUG_PRINTS = False  # Nur Debug-Ausgaben, keine Änderung am Verhalten
ATTEN_SCALE_DEFAULT = 25.0


def str2bool(v):
    if isinstance(v, bool):
        return v
    val = str(v).strip().lower()
    if val in {"yes", "true", "t", "1"}:
        return True
    if val in {"no", "false", "f", "0"}:
        return False
    raise argparse.ArgumentTypeError(f"Boolean value expected, got '{v}'.")


def float_or_none(v):
    if v is None:
        return None
    if isinstance(v, float):
        return v
    val = str(v).strip().lower()
    if val in {"none", "null", ""}:
        return None
    try:
        return float(val)
    except Exception as exc:
        raise argparse.ArgumentTypeError(f"Float or None expected, got '{v}'.") from exc


def parse_args():
    parser = argparse.ArgumentParser(description="Train the emission NeRF on SPECT projections.")
    parser.add_argument("--config", type=str, default="configs/spect.yaml", help="Path to the YAML config.")
    parser.add_argument(
        "--max-steps",
        type=int,
        default=2000,
        help="Number of optimisation steps (mini-batches of rays).",
    )
    parser.add_argument(
        "--rays-per-step",
        type=int,
        default=None,
        help="Number of rays per projection per optimisation step. Defaults to training.chunk in the config.",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=10,
        help="How often to print running loss statistics.",
    )
    parser.add_argument(
        "--preview-every",
        type=int,
        default=50,
        help="If >0 renders and stores full AP/PA previews every N steps (slow, full-frame render).",
    )
    parser.add_argument(
        "--depth-profile-signal-quantile",
        type=float,
        default=0.99,
        help="Quantil fuer Signal-Kandidaten in Depth-Profile-Plots (auf Target-Score).",
    )
    parser.add_argument(
        "--depth-profile-bg-quantile",
        type=float,
        default=0.10,
        help="Quantil fuer Background-Kandidaten in Depth-Profile-Plots (auf Target-Score).",
    )
    parser.add_argument(
        "--depth-profile-seed",
        type=int,
        default=123,
        help="Seed fuer deterministische Auswahl der Depth-Profile-Strahlen.",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=0,
        help="If >0 stores checkpoints every N steps in addition to the final checkpoint.",
    )
    parser.add_argument(
        "--normalize-targets",
        action="store_true",
        help="(Deprecated) Apply per-projection min/max normalisation to both targets and predictions.",
    )
    parser.add_argument(
        "--bg-weight",
        type=float,
        default=1.0,
        help="Down-weights Hintergrundstrahlen im Loss (<1 reduziert Null-Strahlen, 1 = deaktiviert).",
    )
    parser.add_argument(
        "--no-val",
        action="store_true",
        help="Deaktiviert die periodische Test-Split-Evaluation (schnellere Läufe).",
    )
    parser.add_argument(
        "--debug-prints",
        action="store_true",
        help="Aktiviert verbosere Debug-Ausgaben (keine Verhaltensänderung).",
    )
    parser.add_argument(
        "--weight-threshold",
        type=float,
        default=1e-5,
        help="Zählrate, unter der ein Strahl als Hintergrund gilt (nur relevant mit bg-weight < 1).",
    )
    parser.add_argument(
        "--bg-depth-mass-weight",
        type=float,
        default=0.0,
        help="Gewicht fuer BG depth mass loss (0 = deaktiviert).",
    )
    parser.add_argument(
        "--bg-depth-eps",
        type=float,
        default=1e-10,
        help="Schwellwert fuer Background-Kriterium (Target < eps).",
    )
    parser.add_argument(
        "--bg-depth-mode",
        type=str,
        default="integral",
        choices=["integral", "mean"],
        help="BG depth mass mode: integral (sum lambda*dz) oder mean (mean lambda).",
    )
    parser.add_argument(
        "--act-loss-weight",
        type=float,
        default=0.0,
        help="Gewicht für einen optionalen Volumen-Loss gegen act.npy (0 = deaktiviert).",
    )
    parser.add_argument(
        "--hybrid",
        action="store_true",
        help="Aktiviert den Hybrid-Ansatz: AP/PA -> Encoder -> z_enc Conditioning + Projection-Loss als Nebenloss.",
    )
    parser.add_argument(
        "--proj-loss-type",
        type=str,
        default="poisson",
        choices=["poisson", "sqrt_mse", "huber"],
        help="Projection-Loss-Typ (poisson oder sqrt_mse).",
    )
    parser.add_argument(
        "--proj-loss-weight",
        type=float,
        default=0.1,
        help="Gewicht fuer den Projection-Loss im Hybrid-Modus (Nebenloss).",
    )
    parser.add_argument(
        "--proj-warmup-steps",
        type=int,
        default=0,
        help="Optionales Warmup: Schritte 0..W nur ACT+TV, Projection-Loss danach aktiv.",
    )
    parser.add_argument(
        "--proj-weight-min",
        type=float,
        default=0.005,
        help="Unteres Limit fuer proj_loss Gewicht waehrend Warmup/Ramp.",
    )
    parser.add_argument(
        "--proj-ramp-steps",
        type=int,
        default=200,
        help="Anzahl Steps fuer lineare Ramp auf proj_loss_weight nach Warmup.",
    )
    parser.add_argument(
        "--proj-target-source",
        type=str,
        default="counts",
        choices=["counts", "norm"],
        help="Quelle fuer Projection Targets: counts (ap_counts/pa_counts) oder norm (ap/pa).",
    )
    parser.add_argument(
        "--proj-gain-source",
        type=str,
        default="z_enc",
        choices=["z_enc", "scalar", "none"],
        help="Gain g fuer counts-Projektion: z_enc-Head, lernbarer scalar oder none.",
    )
    parser.add_argument(
        "--gain-reg-weight",
        type=float,
        default=1e-4,
        help="Gewicht fuer Gain-Regularisierung (log-gain prior).",
    )
    parser.add_argument(
        "--gain-reg-scale",
        type=float,
        default=1.0,
        help="Skalierung fuer Gain-Regularisierung (1.0 = unveraendert).",
    )
    parser.add_argument(
        "--gain-prior-mode",
        type=str,
        default="ema_init",
        choices=["ema_init", "fixed"],
        help="Gain prior: EMA der ersten Schritte oder fixer Wert.",
    )
    parser.add_argument(
        "--gain-prior-value",
        type=float,
        default=1.0,
        help="Fixer Gain-Prior (nur bei gain-prior-mode=fixed).",
    )
    parser.add_argument(
        "--gain-clamp-min",
        type=float,
        default=0.05,
        help="Optionales Minimum fuer Gain (nur im proj-loss Pfad).",
    )
    parser.add_argument(
        "--gain-clamp-max",
        type=float_or_none,
        default=5.0,
        help="Optionales Maximum fuer Gain (nur im proj-loss Pfad). Setze 'none' fuer aus.",
    )
    parser.add_argument(
        "--encoder-proj-transform",
        type=str,
        default="log1p",
        choices=["log1p", "sqrt", "none"],
        help="Transform fuer Encoder-Input: log1p(y/s), sqrt(y/s) oder none.",
    )
    parser.add_argument(
        "--proj-scale-source",
        type=str,
        default="meta_p99",
        choices=["meta_p99", "compute_p99", "sumcounts", "none"],
        help="Skalenquelle fuer Encoder-Inputs: meta_p99, compute_p99, sumcounts oder none.",
    )
    parser.add_argument(
        "--act-norm-source",
        type=str,
        default="p99_scan",
        choices=["none", "p99_global", "p99_scan", "fixed"],
        help="Normierung fuer ACT-Loss: none, p99_global, p99_scan oder fixed.",
    )
    parser.add_argument(
        "--act-norm-value",
        type=float,
        default=1.0,
        help="Fixer Normierungsfaktor fuer ACT-Loss (nur bei act-norm-source=fixed).",
    )
    parser.add_argument(
        "--encoder-use-ct",
        action="store_true",
        help="Optional: CT als zusaetzlicher Encoder-Input (Mean-Projektion).",
    )
    parser.add_argument(
        "--z-enc-alpha",
        type=float,
        default=0.1,
        help="Skalierung fuer z_enc im Hybrid-Conditioning (z_latent = z_train + alpha * z_enc_proj).",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Fuehrt einen Smoke-Test mit einem Batch (Forward+Backward) aus und beendet.",
    )
    parser.add_argument(
        "--act-samples",
        type=int,
        default=None,
        help="Anzahl zufälliger Voxels zur act-Supervision pro Schritt.",
    )
    parser.add_argument(
        "--act-pos-weight",
        type=float,
        default=None,
        help="Zusatzgewicht für den ACT-Loss in aktiven Voxeln (>0).",
    )
    parser.add_argument(
        "--act-pos-fraction",
        type=float,
        default=0.5,
        help="Anteil positiver ACT-Samples pro Batch.",
    )
    parser.add_argument(
        "--act-pos-threshold",
        type=float,
        default=1e-8,
        help="Threshold für positives ACT-Sampling.",
    )
    parser.add_argument(
        "--act-only",
        action="store_true",
        help="Deaktiviert Projektionsteil (Forward + Loss); nur ACT + Regularizer.",
    )
    parser.add_argument(
        "--debug-act",
        action="store_true",
        help="Einmalige Debug-Logs für ACT-Targets/Normierung/Pred/Grad (Step 1).",
    )
    parser.add_argument(
        "--z-reg-weight",
        type=float,
        default=0.0,
        help="L2-Regularisierung auf dem latenten Code z.",
    )
    parser.add_argument(
        "--ct-loss-weight",
        type=float,
        default=0.0,
        help="Gewicht für den CT-Glättungs-Loss entlang z-Konstanten.",
    )
    parser.add_argument(
        "--ct-threshold",
        type=float,
        default=0.05,
        help="Gradienten-Schwelle in ct.npy, unterhalb derer ein Segment als konstant gilt.",
    )
    parser.add_argument(
        "--ct-samples",
        type=int,
        default=8192,
        help="Anzahl CT-Segmentpaare pro Schritt für den Glättungs-Loss.",
    )
    parser.add_argument(
        "--tv-weight",
        type=float,
        default=0.001,
        help="Gewicht für den 1D-TV-Loss entlang der Rays (0 = deaktiviert).",
    )
    parser.add_argument(
        "--ray-tv-weight",
        type=float,
        default=0.0,
        help="Gewicht für den Ray-1D-TV-Prior entlang der Samples (0 = deaktiviert).",
    )
    parser.add_argument(
        "--ray-tv-edge-aware",
        type=str2bool,
        default=False,
        nargs="?",
        const=True,
        help="Aktiviere edge-aware Ray-TV (benoetigt ray_tv_weight > 0 und ray_tv_alpha > 0).",
    )
    parser.add_argument(
        "--ray-tv-alpha",
        type=float,
        default=0.0,
        help="Alpha fuer edge-aware Ray-TV (Gewicht exp(-alpha*|delta_mu|)).",
    )
    parser.add_argument(
        "--ray-tv-w-clamp-min",
        type=float,
        default=0.0,
        help="Optionales Minimum fuer edge-aware TV-Gewichte (z.B. 0.2).",
    )
    parser.add_argument(
        "--depth-sanity-every",
        type=int,
        default=50,
        help="Depth-Checks/Abbruch alle N Schritte (0 = deaktiviert).",
    )
    parser.add_argument(
        "--proj-collapse-patience",
        type=int,
        default=0,
        help="Abbruch erst nach N aufeinanderfolgenden Projection-Collapses (0 = nie aborten).",
    )
    parser.add_argument(
        "--grad-stats-every",
        type=int,
        default=0,
        help="Falls >0: Gradienten-Normen je Loss-Term alle N Schritte (nur z-Latent, retain_graph).",
    )
    parser.add_argument(
        "--atten-scale",
        type=float,
        default=ATTEN_SCALE_DEFAULT,
        help="Globaler Längenskalenfaktor für die Attenuation (μ in 1/cm, Bounding Box ~1).",
    )
    parser.add_argument(
        "--ct-padding-mode",
        type=str,
        default="border",
        choices=["border", "zeros"],
        help="Padding-Mode fuer CT grid_sample (border|zeros).",
    )
    parser.add_argument(
        "--ray-split",
        type=float,
        default=0.8,
        help="Anteil der Rays pro Bild für das Training (Rest = Test) beim stratifizierten Split.",
    )
    parser.add_argument(
        "--ray-split-mode",
        type=str,
        default="tile_random",
        choices=["tile_random", "stratified_intensity"],
        help="Split-Modus: tile_random (Tile-permutiert) oder stratified_intensity (FG/BG stratifiziert).",
    )
    parser.add_argument(
        "--ray-split-seed",
        type=int,
        default=123,
        help="Seed für den stratifizierten Ray-Split.",
    )
    parser.add_argument(
        "--ray-split-tile",
        type=int,
        default=32,
        help="Tile-Kantenlänge in Pixeln für den Ray-Split.",
    )
    parser.add_argument(
        "--ray-fg-thr",
        type=str,
        default="0.0",
        help="Schwellwert für Vordergrund (target>thr). Zahl oder 'quantile'.",
    )
    parser.add_argument(
        "--ray-fg-quantile",
        type=float,
        default=0.90,
        help="Quantil q für FG-Definition, falls ray-fg-thr<=0 oder 'quantile'.",
    )
    parser.add_argument(
        "--ray-train-fg-frac",
        type=float,
        default=0.5,
        help="Anteil Vordergrund-Rays beim Training-Sampling (Rest Hintergrund, mit Fallback).",
    )
    parser.add_argument(
        "--ray-split-enable",
        type=str2bool,
        default=True,
        nargs="?",
        const=True,
        help="Aktiviere stratifizierten Ray-Split (False => Legacy-Uniform-Split).",
    )
    parser.add_argument(
        "--pa-xflip",
        type=str2bool,
        default=False,
        nargs="?",
        const=True,
        help="Spiegle PA in x-Richtung, um Pixel zu AP zu mappen.",
    )
    parser.add_argument(
        "--log-quantiles-final-only",
        type=str2bool,
        default=True,
        nargs="?",
        const=True,
        help="Logge p50/p80/p95/p99 der rohen AP/PA-Projektionen (Pred + Target) nur am finalen Step.",
    )
    parser.add_argument(
        "--debug-proj-stats",
        action="store_true",
        help="Einmalige AP/PA-Min/Max/p99.9-Statistiken direkt nach dem Laden loggen.",
    )
    parser.add_argument(
        "--log-proj-metrics-physical",
        action="store_true",
        help="Logge PSNR/MAE/Quantiles zusaetzlich im physikalischen Massstab (re-skaliert).",
    )
    parser.add_argument(
        "--export-vol-res",
        type=int,
        default=128,
        help="Grid-Aufloesung fuer Export des finalen Aktivitaetsvolumens (z. B. 128 oder 256).",
    )
    parser.add_argument(
        "--export-vol-every",
        type=int,
        default=0,
        help="Optional: Exportiere Aktivitaetsvolumen alle N Schritte (0 = aus).",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    return parser.parse_args()


def set_seed(seed: int):
    # deterministische Seeds für Torch + NumPy, damit Runs reproduzierbar bleiben
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)


def save_img(arr, path, title=None):
    """Robust PNG visualisation with optional logarithmic stretch."""
    import matplotlib.pyplot as plt

    # Nan/Inf-Fälle abfangen, damit matplotlib nicht abstürzt
    if not np.isfinite(arr).all():
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

    # Wertebereich auf 0..1 strecken, notfalls via Log-Scaling
    a_min, a_max = float(np.min(arr)), float(np.max(arr))
    if np.isclose(a_min, a_max):
        img = np.zeros_like(arr) if a_max == 0 else arr / (a_max + 1e-8)
    else:
        arr_shift = arr - a_min
        arr_log = np.log1p(arr_shift)
        img = (arr_log - arr_log.min()) / (arr_log.max() - arr_log.min() + 1e-8)

    plt.figure(figsize=(6, 4))
    plt.imshow(img, cmap="gray")
    if title:
        plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def log_projection_quantiles(ap_pred, pa_pred, ap_target=None, pa_target=None, tag="final"):
    quantiles = [0.5, 0.8, 0.95, 0.99]

    def _q(arr):
        data = arr.detach().float().cpu().numpy().ravel()
        return np.quantile(data, quantiles)

    ap_pred_q = _q(ap_pred)
    pa_pred_q = _q(pa_pred)
    msg = (
        f"[quantiles][{tag}] pred_ap p50={ap_pred_q[0]:.3e} p80={ap_pred_q[1]:.3e} "
        f"p95={ap_pred_q[2]:.3e} p99={ap_pred_q[3]:.3e} | "
        f"pred_pa p50={pa_pred_q[0]:.3e} p80={pa_pred_q[1]:.3e} "
        f"p95={pa_pred_q[2]:.3e} p99={pa_pred_q[3]:.3e}"
    )
    if ap_target is not None and pa_target is not None:
        ap_t_q = np.quantile(ap_target.ravel(), quantiles)
        pa_t_q = np.quantile(pa_target.ravel(), quantiles)
        msg += (
            f" | target_ap p50={ap_t_q[0]:.3e} p80={ap_t_q[1]:.3e} "
            f"p95={ap_t_q[2]:.3e} p99={ap_t_q[3]:.3e} | "
            f"target_pa p50={pa_t_q[0]:.3e} p80={pa_t_q[1]:.3e} "
            f"p95={pa_t_q[2]:.3e} p99={pa_t_q[3]:.3e}"
        )
    print(msg, flush=True)


def log_projection_quantiles_scaled(
    ap_pred,
    pa_pred,
    ap_target=None,
    pa_target=None,
    tag="final",
    ap_scale: float = 1.0,
    pa_scale: float = 1.0,
):
    ap_scale = float(ap_scale)
    pa_scale = float(pa_scale)
    log_projection_quantiles(
        ap_pred * ap_scale,
        pa_pred * pa_scale,
        ap_target=None if ap_target is None else ap_target * ap_scale,
        pa_target=None if pa_target is None else pa_target * pa_scale,
        tag=tag,
    )


def export_activity_volume(generator, z_latent, out_path: Path, res: int, device: torch.device):
    radius = generator.radius
    if isinstance(radius, tuple):
        radius = radius[1]
    radius = float(radius)
    res = int(res)
    if res <= 0:
        raise ValueError("export-vol-res must be > 0")

    x_coords = idx_to_coord(torch.arange(res, device=device), res, radius)
    y_coords = idx_to_coord(torch.arange(res, device=device), res, radius)
    y_grid, x_grid = torch.meshgrid(y_coords, x_coords, indexing="ij")
    x_flat = x_grid.reshape(-1)
    y_flat = y_grid.reshape(-1)

    target_points = 262144
    chunk_depth = max(1, min(res, target_points // (res * res) if res * res > 0 else 1))

    vol = np.empty((res, res, res), dtype=np.float32)
    with torch.no_grad():
        for z_start in range(0, res, chunk_depth):
            z_end = min(res, z_start + chunk_depth)
            z_idx = torch.arange(z_start, z_end, device=device)
            z_coords = idx_to_coord(z_idx, res, radius)
            z_rep = z_coords.repeat_interleave(x_flat.numel())
            x_rep = x_flat.repeat(z_coords.numel())
            y_rep = y_flat.repeat(z_coords.numel())
            coords = torch.stack((x_rep, y_rep, z_rep), dim=1)
            pred = query_emission_at_points(generator, z_latent, coords)
            pred = pred.view(z_coords.numel(), res, res).detach().cpu().numpy().astype(np.float32)
            vol[z_start:z_end, :, :] = pred

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, vol)


def poisson_nll(
    pred: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-8,
    clamp_max: float = 1e6,
    weight: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Poisson Negative Log-Likelihood Loss für Emissions- oder Zähl-Daten.
    Erwartet nichtnegative 'pred' und 'target' (z. B. Intensitäten).
    Falls projizierte Zählraten global skaliert werden, müssen pred/target
    konsistent dieselbe Skalierung durchlaufen – der Loss bleibt physikalisch
    äquivalent (nur numerische Reskalierung).
    """
    # Stabilisierung über clamping, damit log() definiert bleibt
    pred = pred.clamp_min(eps).clamp_max(clamp_max)
    nll = pred - target * torch.log(pred)
    if weight is not None:
        nll = nll * weight
    return nll.mean()


def sqrt_mse_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-8,
    weight: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Sqrt-MSE: ||sqrt(pred) - sqrt(target)||^2, stabilisiert via clamp."""
    pred_s = torch.sqrt(pred.clamp_min(eps))
    target_s = torch.sqrt(target.clamp_min(eps))
    diff2 = (pred_s - target_s) ** 2
    if weight is not None:
        diff2 = diff2 * weight
    return torch.mean(diff2)


def huber_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    delta: float = 1.0,
    weight: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    diff = pred - target
    abs_diff = torch.abs(diff)
    quad = torch.minimum(abs_diff, torch.tensor(delta, device=pred.device, dtype=pred.dtype))
    lin = abs_diff - quad
    loss = 0.5 * quad * quad + delta * lin
    if weight is not None:
        loss = loss * weight
    return torch.mean(loss)


def _extract_meta_scalar(meta, key: str) -> Optional[float]:
    if not isinstance(meta, dict):
        return None
    val = meta.get(key)
    if isinstance(val, (list, tuple)):
        val = val[0] if val else None
    if torch.is_tensor(val):
        if val.numel() == 0:
            return None
        return float(val.detach().view(-1)[0].item())
    if val is None:
        return None
    try:
        return float(val)
    except Exception:
        return None


def compute_joint_p99(ap: torch.Tensor, pa: torch.Tensor) -> torch.Tensor:
    """Joint p99 über AP+PA pro Batch-Sample (returns [B])."""
    if ap.dim() > 2:
        ap_flat = ap.reshape(ap.shape[0], -1)
    else:
        ap_flat = ap.unsqueeze(0)
    if pa.dim() > 2:
        pa_flat = pa.reshape(pa.shape[0], -1)
    else:
        pa_flat = pa.unsqueeze(0)
    joint = torch.cat([ap_flat, pa_flat], dim=1)
    return torch.quantile(joint.float(), 0.99, dim=1)


def compute_proj_scale(
    ap: torch.Tensor,
    pa: torch.Tensor,
    source: str,
    meta: Optional[dict] = None,
) -> torch.Tensor:
    """Bestimmt Skalenfaktor s pro Sample fuer Encoder-Inputs."""
    source = str(source or "none")
    B = ap.shape[0] if ap.dim() >= 3 else 1
    device = ap.device
    if source == "none":
        return torch.ones((B,), device=device)
    if source == "meta_p99":
        meta_scale = _extract_meta_scalar(meta, "proj_scale_joint_p99")
        if meta_scale is not None and math.isfinite(meta_scale) and meta_scale > 0:
            return torch.full((B,), float(meta_scale), device=device)
        # Fallback: compute p99 on the fly
        return compute_joint_p99(ap, pa)
    if source == "compute_p99":
        return compute_joint_p99(ap, pa)
    if source == "sumcounts":
        ap_sum = ap.reshape(B, -1).sum(dim=1)
        pa_sum = pa.reshape(B, -1).sum(dim=1)
        return ap_sum + pa_sum
    raise ValueError(f"Unknown proj_scale_source: {source}")


def apply_proj_transform(
    proj: torch.Tensor,
    scale: torch.Tensor,
    transform: str,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Skaliert proj mit s und wendet Transform an (log1p/sqrt/none)."""
    transform = str(transform or "none")
    while scale.dim() < proj.dim():
        scale = scale.view(-1, *([1] * (proj.dim() - 1)))
    scaled = proj / torch.clamp(scale, min=eps)
    scaled = torch.clamp(scaled, min=0.0)
    if transform == "log1p":
        return torch.log1p(scaled)
    if transform == "sqrt":
        return torch.sqrt(scaled + eps)
    if transform == "none":
        return scaled
    raise ValueError(f"Unknown encoder_proj_transform: {transform}")


def compute_act_norm_factor(
    act_vol: Optional[torch.Tensor],
    source: str,
    fixed_value: float,
    cached_global: Optional[float],
) -> Tuple[float, Optional[float]]:
    """Ermittelt Normierungsfaktor fuer ACT-Loss; gibt ggf. neuen globalen Cache zurueck."""
    source = str(source or "p99_scan")
    if source == "none":
        return 1.0, cached_global
    if source == "fixed":
        val = float(fixed_value)
        return (val if val > 0 else 1.0), cached_global
    if act_vol is None or act_vol.numel() == 0:
        return 1.0, cached_global
    def _approx_quantile(t: torch.Tensor, q: float, max_samples: int = 1_000_000) -> float:
        flat = t.reshape(-1)
        if flat.numel() <= max_samples:
            return float(torch.quantile(flat, q).item())
        # subsample to keep quantile fast/robust on large volumes
        idx = torch.randint(0, flat.numel(), (max_samples,), device=flat.device)
        sample = flat[idx]
        return float(torch.quantile(sample, q).item())

    if source == "p99_global":
        if cached_global is not None and cached_global > 0:
            return cached_global, cached_global
        p99 = _approx_quantile(act_vol.float(), 0.99)
        p99 = p99 if p99 > 0 else 1.0
        return p99, p99
    if source == "p99_scan":
        p99 = _approx_quantile(act_vol.float(), 0.99)
        return (p99 if p99 > 0 else 1.0), cached_global
    raise ValueError(f"Unknown act_norm_source: {source}")


def nonfinite_fraction(t: Optional[torch.Tensor]) -> float:
    if t is None or t.numel() == 0:
        return 0.0
    return float((~torch.isfinite(t)).float().mean().item())


def tensor_stats(t: Optional[torch.Tensor]) -> Optional[dict]:
    if t is None or t.numel() == 0:
        return None
    t = t.detach().float()
    flat = t.reshape(-1)
    # Quantile on very large tensors can error; subsample for robust stats.
    if flat.numel() > 1_000_000:
        idx = torch.randint(0, flat.numel(), (1_000_000,), device=flat.device)
        flat = flat[idx]
    return {
        "min": float(flat.min().item()),
        "mean": float(flat.mean().item()),
        "p95": float(torch.quantile(flat, 0.95).item()),
        "max": float(flat.max().item()),
    }


def fmt_stats(stats: Optional[dict]) -> str:
    if stats is None:
        return "min/mean/p95/max=nan/nan/nan/nan"
    return (
        "min/mean/p95/max="
        f"{stats['min']:.3e}/{stats['mean']:.3e}/{stats['p95']:.3e}/{stats['max']:.3e}"
    )


def build_encoder_input(
    ap: torch.Tensor,
    pa: torch.Tensor,
    ct_vol: Optional[torch.Tensor],
    scale: torch.Tensor,
    transform: str,
    use_ct: bool,
) -> torch.Tensor:
    """Baut den Encoder-Input als [B,C,H,W] aus AP/PA (+optional CT)."""
    ap_enc = apply_proj_transform(ap, scale, transform)
    pa_enc = apply_proj_transform(pa, scale, transform)
    inputs = [ap_enc, pa_enc]
    if use_ct:
        if ct_vol is None or ct_vol.numel() == 0:
            # CT fehlt: Dummy-Channel mit 0
            zeros = torch.zeros_like(ap_enc)
            inputs.append(zeros)
        else:
            # ct_vol: [B,D,H,W] -> Mean-Projektion [B,1,H,W]
            if ct_vol.dim() == 3:
                ct = ct_vol.unsqueeze(0)
            else:
                ct = ct_vol
            ct_mean = ct.mean(dim=1, keepdim=True)
            # an AP/PA-Auflösung anpassen
            if ct_mean.shape[-2:] != ap_enc.shape[-2:]:
                ct_mean = F.interpolate(ct_mean, size=ap_enc.shape[-2:], mode="bilinear", align_corners=False)
            # einfache Standardisierung pro Sample
            ct_flat = ct_mean.reshape(ct_mean.shape[0], -1)
            ct_mu = ct_flat.mean(dim=1).view(-1, 1, 1, 1)
            ct_std = ct_flat.std(dim=1).view(-1, 1, 1, 1)
            ct_norm = (ct_mean - ct_mu) / (ct_std + 1e-6)
            inputs.append(ct_norm)
    return torch.cat(inputs, dim=1)


def build_hwfr_from_config(data_cfg: dict) -> list:
    """Fallback HWFR fuer Smoke-Tests ohne Datenzugriff."""
    imsize = data_cfg.get("imsize") or data_cfg.get("H") or 128
    H = int(imsize)
    W = int(data_cfg.get("W") or H)
    fov = float(data_cfg.get("fov", 60.0))
    focal = W / 2.0 * 1.0 / np.tan(0.5 * fov * np.pi / 180.0)
    radius = data_cfg.get("radius", 0.5)
    render_radius = radius
    if isinstance(radius, str):
        radius = tuple(float(r) for r in radius.split(","))
        render_radius = max(radius)
    return [H, W, focal, render_radius]


def build_synthetic_batch(
    H: int,
    W: int,
    device: torch.device,
    with_ct: bool = True,
    with_act: bool = True,
) -> dict:
    """Erzeugt ein synthetisches Batch fuer Smoke-Tests (ohne I/O)."""
    B = 1
    ap = torch.rand((B, 1, H, W), device=device) * 5.0
    pa = torch.rand((B, 1, H, W), device=device) * 5.0
    ap_counts = ap * 1000.0
    pa_counts = pa * 1000.0
    D = int(min(32, H))
    ct = torch.rand((D, H, W), device=device) if with_ct else torch.empty(0, device=device)
    act = torch.rand((D, H, W), device=device) if with_act else torch.empty(0, device=device)
    meta = {"proj_scale_joint_p99": float(torch.quantile(torch.cat([ap.reshape(-1), pa.reshape(-1)]), 0.99).item())}
    return {
        "ap": ap,
        "pa": pa,
        "ap_counts": ap_counts,
        "pa_counts": pa_counts,
        "ct": ct,
        "act": act,
        "meta": meta,
    }


def compute_lambda_and_attenuation_stats(
    extras_list,
    atten_scale: float,
    clamp_max: float = 60.0,
) -> Tuple[Optional[dict], Optional[dict], Optional[dict], Optional[float], Optional[float], float, float]:
    """Aggregiert lambda/attenuation-Stats aus Extras (raw/mu/dists)."""
    lambda_vals = []
    atten_vals = []
    mu_vals = []
    for extras in extras_list:
        if not isinstance(extras, dict):
            continue
        raw_out = extras.get("raw")
        if raw_out is None:
            continue
        lambda_vals.append(F.softplus(raw_out[..., 0]))
        mu = extras.get("mu")
        dists = extras.get("dists")
        if mu is None or dists is None:
            continue
        if mu.shape != dists.shape:
            continue
        mu = torch.clamp(mu, min=0.0)
        mu_vals.append(mu)
        mu_dists = mu * dists
        attenuation = torch.cumsum(mu_dists, dim=-1) * float(atten_scale)
        attenuation = F.pad(attenuation[..., :-1], (1, 0), mode="constant", value=0.0)
        attenuation = torch.clamp(attenuation, min=0.0, max=clamp_max)
        atten_vals.append(attenuation)
    lambda_stats = None
    atten_stats = None
    frac_gt20 = None
    frac_clamp = None
    if lambda_vals:
        lambda_all = torch.cat([lv.reshape(-1) for lv in lambda_vals], dim=0)
        lambda_stats = tensor_stats(lambda_all)
    mu_stats = None
    if mu_vals:
        mu_all = torch.cat([mv.reshape(-1) for mv in mu_vals], dim=0)
        mu_stats = tensor_stats(mu_all)
    if atten_vals:
        atten_all = torch.cat([av.reshape(-1) for av in atten_vals], dim=0)
        atten_stats = tensor_stats(atten_all)
        frac_gt20 = float((atten_all > 20.0).float().mean().item())
        frac_clamp = float((atten_all >= clamp_max - 1e-6).float().mean().item())
    if lambda_vals:
        lambda_flat = torch.cat([lv.reshape(-1) for lv in lambda_vals], dim=0)
        nonfinite_lambda = nonfinite_fraction(lambda_flat)
    else:
        nonfinite_lambda = 0.0
    if atten_vals:
        atten_flat = torch.cat([av.reshape(-1) for av in atten_vals], dim=0)
        nonfinite_atten = nonfinite_fraction(atten_flat)
    else:
        nonfinite_atten = 0.0
    return lambda_stats, mu_stats, atten_stats, frac_gt20, frac_clamp, nonfinite_lambda, nonfinite_atten


def build_ray_split(num_pixels: int, split_ratio: float, device: torch.device) -> Dict[str, torch.Tensor]:
    """
    Erzeuge einen festen Train/Test-Split über alle Rays einer Ansicht.
    Split ist reproduzierbar, weil der globale Seed (set_seed) bereits gesetzt wurde.

    Rückgabe: {"train": train_idx, "test": test_idx} (jeweils torch.long auf device)
    """
    ratio = float(split_ratio)
    ratio = 0.0 if ratio < 0 else (1.0 if ratio > 1.0 else ratio)
    perm = torch.randperm(num_pixels, device=device)
    n_train = int(math.ceil(num_pixels * ratio))
    train_idx = perm[:n_train]
    test_idx = perm[n_train:]
    return {"train": train_idx, "test": test_idx}


def sample_split_indices(split_tensor: torch.Tensor, count: int) -> torch.Tensor:
    """Ziehe zufällige Indizes aus einem vorgegebenen Split (keine neuen Rays von der Gegenseite)."""
    if split_tensor.numel() <= count:
        return split_tensor
    rand_idx = torch.randint(0, split_tensor.numel(), (count,), device=split_tensor.device)
    return split_tensor[rand_idx]


def map_pa_indices_torch(idx: torch.Tensor, W: int, do_flip: bool) -> torch.Tensor:
    if not do_flip:
        return idx
    y = idx // W
    x = idx % W
    return y * W + (W - 1 - x)


def grad_norm_of(loss_term: torch.Tensor, params) -> float:
    """L2-Norm der Gradienten eines Loss-Terms bezogen auf gegebene Parameter (z. B. z_latent)."""
    if loss_term is None or not loss_term.requires_grad:
        return 0.0
    grads = torch.autograd.grad(loss_term, params, retain_graph=True, allow_unused=True)
    grads = [g for g in grads if g is not None]
    if not grads:
        return 0.0
    flat = torch.cat([g.reshape(-1) for g in grads])
    return float(flat.norm().detach().cpu().item())


def grad_norm_of_module(loss_term: torch.Tensor, module: Optional[nn.Module]) -> float:
    if module is None:
        return 0.0
    params = [p for p in module.parameters() if p.requires_grad]
    if not params:
        return 0.0
    return grad_norm_of(loss_term, params)


def global_grad_norm(params) -> float:
    """L2-Norm ueber alle vorhandenen Gradienten (logging only)."""
    total = 0.0
    for p in params:
        if p is None or p.grad is None:
            continue
        g = p.grad.detach()
        total += float(g.norm().item()) ** 2
    return math.sqrt(total) if total > 0.0 else 0.0


def module_grad_mean_abs(module: Optional[nn.Module]) -> float:
    if module is None:
        return 0.0
    vals = [p.grad.detach().abs().mean() for p in module.parameters() if p.grad is not None]
    if not vals:
        return 0.0
    return float(torch.stack(vals).mean().item())


def safe_git_rev() -> str:
    """Versucht den aktuellen Git-Commit (kurz) zu lesen, fällt andernfalls auf 'unknown' zurück."""
    try:
        out = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=Path(__file__).resolve().parent)
        return out.decode().strip()
    except Exception as exc:  # noqa: BLE001 – bewusst breit, nur Debug-Info
        return f"unknown ({exc.__class__.__name__})"


def log_effective_config(outdir: Path, config: dict, args):
    """Einmalige Ausgabe der effektiv genutzten Konfiguration (nach YAML+CLI-Merge)."""
    nerf_cfg = config.get("nerf", {})
    data_cfg = config.get("data", {})
    training_cfg = config.get("training", {})
    git_rev = safe_git_rev()
    print(f"[cfg] git_rev={git_rev} | expname={config.get('expname', 'n/a')} | outdir={outdir}", flush=True)
    print(
        f"[cfg][data] act_scale={data_cfg.get('act_scale')} | near={data_cfg.get('near')} | far={data_cfg.get('far')} "
        f"| orthographic={data_cfg.get('orthographic')}",
        flush=True,
    )
    print(
        f"[cfg][nerf] N_samples={nerf_cfg.get('N_samples')} | N_importance={nerf_cfg.get('N_importance')} "
        f"| perturb={nerf_cfg.get('perturb')} | atten_scale={nerf_cfg.get('atten_scale')} "
        f"| use_attenuation={nerf_cfg.get('use_attenuation')}",
        flush=True,
    )
    print(
        f"[cfg][training] lr_g={training_cfg.get('lr_g')} | tv_weight={training_cfg.get('tv_weight')} "
        f"| act_loss_weight={args.act_loss_weight} | act_samples={args.act_samples} "
        f"| act_pos_weight={args.act_pos_weight} | act_pos_fraction={args.act_pos_fraction} "
        f"| act_pos_threshold={args.act_pos_threshold} "
        f"| ct_loss_weight={args.ct_loss_weight} | ct_threshold={args.ct_threshold} | z_reg_weight={args.z_reg_weight} "
        f"| ray_tv_weight={args.ray_tv_weight} | ray_tv_edge_aware={args.ray_tv_edge_aware} | ray_tv_alpha={args.ray_tv_alpha} "
        f"| ray_tv_w_clamp_min={args.ray_tv_w_clamp_min} | ct_padding_mode={args.ct_padding_mode} "
        f"| bg_depth_mass_weight={args.bg_depth_mass_weight} | bg_depth_eps={args.bg_depth_eps} | bg_depth_mode={args.bg_depth_mode}",
        flush=True,
    )
    if getattr(args, "hybrid", False):
        print(
            f"[cfg][hybrid] proj_loss_type={args.proj_loss_type} | proj_loss_weight={args.proj_loss_weight} "
            f"| proj_warmup_steps={args.proj_warmup_steps} | proj_weight_min={args.proj_weight_min} "
            f"| proj_ramp_steps={args.proj_ramp_steps} | proj_target_source={args.proj_target_source} "
            f"| proj_gain_source={args.proj_gain_source} | gain_reg_weight={args.gain_reg_weight} "
            f"| gain_reg_scale={args.gain_reg_scale} "
            f"| gain_prior_mode={args.gain_prior_mode} | gain_prior_value={args.gain_prior_value} "
            f"| gain_clamp_min={args.gain_clamp_min} | gain_clamp_max={args.gain_clamp_max} "
            f"| encoder_proj_transform={args.encoder_proj_transform} "
            f"| proj_scale_source={args.proj_scale_source} | act_norm_source={args.act_norm_source} "
            f"| act_norm_value={args.act_norm_value} | encoder_use_ct={args.encoder_use_ct} "
            f"| z_enc_alpha={args.z_enc_alpha}",
            flush=True,
        )


def build_loss_weights(target: torch.Tensor, bg_weight: float, threshold: float) -> Optional[torch.Tensor]:
    """Erzeuge optionale Strahl-Gewichte, die Null-Strahlen abschwächen."""
    if bg_weight >= 1.0:
        return None
    weights = torch.ones_like(target)
    weights = weights.masked_fill(target <= threshold, bg_weight)
    return weights


def build_pose_rays(generator, pose):
    """Pre-compute all rays for a fixed pose and keep them on the target device."""
    # Ortho-Kamera nutzt ortho_size statt focal
    focal_or_size = generator.ortho_size if generator.orthographic else generator.focal
    rays_full, _, _ = generator.val_ray_sampler(generator.H, generator.W, focal_or_size, pose)
    return rays_full.to(generator.device, non_blocking=True)


def slice_rays(rays_full: torch.Tensor, ray_idx: torch.Tensor) -> torch.Tensor:
    """Select a subset of rays (by linear indices) for a mini-batch."""
    # rays_full hat Form (2, HW, 3) -> mit Indexliste extrahieren
    return torch.stack(
        (
            rays_full[0, ray_idx],
            rays_full[1, ray_idx],
        ),
        dim=0,
    )


def render_minibatch(generator, z_latent, rays_subset, ct_context=None, return_raw: bool = False):
    """Render a mini-batch of rays from a fixed pose while keeping training kwargs."""
    # train/test kwargs werden durch use_test_kwargs umgeschaltet
    render_kwargs = generator.render_kwargs_train if not generator.use_test_kwargs else generator.render_kwargs_test
    render_kwargs = dict(render_kwargs)
    render_kwargs["features"] = z_latent
    if ct_context is not None:
        render_kwargs["ct_context"] = ct_context
    elif render_kwargs.get("use_attenuation"):
        render_kwargs["use_attenuation"] = False
    if return_raw:
        render_kwargs["retraw"] = True
    if DEBUG_PRINTS:
        render_kwargs["debug_prints"] = True
    proj_map, _, _, extras = generator.render(rays=rays_subset, **render_kwargs)
    return proj_map.view(z_latent.shape[0], -1), extras


def compute_ray_tv(
    raw: torch.Tensor,
    mu_vals: Optional[torch.Tensor] = None,
    edge_aware: bool = False,
    alpha: float = 0.0,
    w_clamp_min: float = 0.0,
    mu_thresh: float = 1e-3,
    return_stats: bool = False,
) -> Tuple[torch.Tensor, Optional[dict]]:
    """Berechnet den (edge-aware) 1D-Total-Variation-Prior entlang jedes Rays."""
    lambda_vals = F.softplus(raw[..., 0])  # [N_rays, N_samples]
    diffs = torch.abs(lambda_vals[..., 1:] - lambda_vals[..., :-1])
    if edge_aware and mu_vals is not None and alpha > 0.0:
        if mu_vals.shape != lambda_vals.shape:
            raise ValueError(f"CT samples have wrong shape {mu_vals.shape}, expected {lambda_vals.shape}.")
        mu = torch.clamp(mu_vals, min=0.0)
        mu_diffs = torch.abs(mu[..., 1:] - mu[..., :-1])
        weights = torch.exp(-alpha * mu_diffs)
        if w_clamp_min > 0.0:
            weights = torch.clamp(weights, min=w_clamp_min)
        tv_per_ray = torch.sum(weights * diffs, dim=-1)
        tv = torch.mean(tv_per_ray)
        if return_stats:
            stats = {
                "w_mean": weights.mean(),
                "w_min": weights.min(),
                "w_max": weights.max(),
            }
            if mu_thresh is not None and mu.numel() > 0:
                mask = mu > mu_thresh
                valid = mask.any(dim=-1)
                if valid.any():
                    first_idx = mask.float().argmax(dim=-1)
                    depths = first_idx[valid].float() / max(1.0, float(mu.shape[-1] - 1))
                    stats["ct_boundary_depth_mean"] = depths.mean()
                    stats["ct_boundary_depth_median"] = depths.median()
            return tv, stats
        return tv, None
    tv_per_ray = torch.sum(diffs, dim=-1)
    tv = torch.mean(tv_per_ray)
    if return_stats:
        return tv, None
    return tv, None


def maybe_render_preview(
    step,
    args,
    generator,
    z_eval,
    outdir,
    ct_volume=None,
    act_volume=None,
    ct_context=None,
    target_ap=None,
    target_pa=None,
    target_ap_counts=None,
    target_pa_counts=None,
):
    # Volle AP/PA-Renderings sind teuer; nur alle N Schritte ausführen
    if args.preview_every <= 0 or (step % args.preview_every) != 0:
        return
    prev_flag = generator.use_test_kwargs
    generator.eval()
    generator.use_test_kwargs = True
    ctx = ct_context or generator.build_ct_context(ct_volume, padding_mode=args.ct_padding_mode)
    with torch.no_grad():
        proj_ap, _, _, _ = generator.render_from_pose(z_eval, generator.pose_ap, ct_context=ctx)
        proj_pa, _, _, _ = generator.render_from_pose(z_eval, generator.pose_pa, ct_context=ctx)
    generator.train()
    generator.use_test_kwargs = prev_flag or False
    H, W = generator.H, generator.W
    ap_np = proj_ap[0].reshape(H, W).detach().cpu().numpy()
    pa_np = proj_pa[0].reshape(H, W).detach().cpu().numpy()
    out_dir = outdir / "preview"
    out_dir.mkdir(parents=True, exist_ok=True)
    save_img(ap_np, out_dir / f"step_{step:05d}_AP.png", title=f"AP @ step {step}")
    save_img(pa_np, out_dir / f"step_{step:05d}_PA.png", title=f"PA @ step {step}")
    save_depth_profile(
        step,
        generator,
        z_eval,
        ct_volume,
        act_volume,
        out_dir,
        proj_ap=proj_ap,
        proj_pa=proj_pa,
        target_ap=target_ap,
        target_pa=target_pa,
        target_ap_counts=target_ap_counts,
        target_pa_counts=target_pa_counts,
        signal_quantile=args.depth_profile_signal_quantile,
        bg_quantile=args.depth_profile_bg_quantile,
        seed=args.depth_profile_seed,
    )
    print("🖼️ Preview gespeichert:", flush=True)
    print("   ", (out_dir / f"step_{step:05d}_AP.png").resolve(), flush=True)
    print("   ", (out_dir / f"step_{step:05d}_PA.png").resolve(), flush=True)


def init_log_file(path: Path):
    # CSV-Header nur einmal schreiben
    if path.exists():
        return
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "step",
                "loss",
                "loss_ap",
                "loss_pa",
                "loss_act",
                "loss_ct",
                "ray_tv",
                "ray_tv_w",
                "bg_depth_mass",
                "bg_depth_mass_w",
                "bg_depth_frac",
                "loss_tv",
                "zreg",
                "mae_ap",
                "mae_pa",
                "psnr_ap",
                "psnr_pa",
                "pred_mean_ap",
                "pred_mean_pa",
                "pred_std_ap",
                "pred_std_pa",
                "loss_test_all",
                "loss_test_ap",
                "loss_test_pa",
                "psnr_test_all",
                "psnr_test_ap",
                "psnr_test_pa",
                "mae_test_all",
                "mae_test_ap",
                "mae_test_pa",
                "loss_test_fg",
                "psnr_test_fg",
                "mae_test_fg",
                "loss_test_top10",
                "psnr_test_top10",
                "mae_test_top10",
                "iter_ms",
                "lr",
                "ray_tv_mode",
                "ray_tv_w_mean",
            ]
        )


def append_log(path: Path, row):
    with path.open("a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(row)


def init_hybrid_log_file(path: Path):
    if path.exists():
        try:
            header = path.read_text().splitlines()[0]
            if "gain" not in header or "mu_min" not in header:
                print(
                    f"[hybrid][warn] existing hybrid_stats.csv has old header; "
                    f"consider deleting {path} to get new columns.",
                    flush=True,
                )
        except Exception:
            pass
        return
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "step",
                "proj_weight",
                "loss_proj",
                "loss_ap",
                "loss_pa",
                "loss_act",
                "loss_gain",
                "gain_prior",
                "act_norm_factor",
                "loss_total",
                "proj_scale_enc",
                "target_ap_min",
                "target_ap_mean",
                "target_ap_p95",
                "target_ap_max",
                "target_pa_min",
                "target_pa_mean",
                "target_pa_p95",
                "target_pa_max",
                "pred_ap_min",
                "pred_ap_mean",
                "pred_ap_p95",
                "pred_ap_max",
                "pred_pa_min",
                "pred_pa_mean",
                "pred_pa_p95",
                "pred_pa_max",
                "lambda_ray_min",
                "lambda_ray_mean",
                "lambda_ray_p95",
                "lambda_ray_max",
                "mu_min",
                "mu_mean",
                "mu_p95",
                "mu_max",
                "atten_min",
                "atten_mean",
                "atten_p95",
                "atten_max",
                "atten_frac_gt20",
                "atten_frac_clamp60",
                "gain",
                "nonfinite_pred",
                "nonfinite_lambda",
                "nonfinite_atten",
                "grad_norm_global",
                "grad_norm_gen",
                "clip_event",
                "z_train_l2",
                "z_enc_l2",
                "z_latent_l2",
            ]
        )


def append_hybrid_log(path: Path, row):
    with path.open("a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(row)


def save_checkpoint(step, generator, z_train, optimizer, scaler, ckpt_dir: Path, encoder=None, z_fuser=None, gain_head=None, gain_param=None):
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    # Minimal-Checkpoint: coarse/fine Netze, Optimizer, AMP-Scaler
    state = {
        "step": step,
        "z_train": z_train.detach().cpu(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "generator_coarse": generator.render_kwargs_train["network_fn"].state_dict(),
        "generator_fine": None,
    }
    if generator.render_kwargs_train["network_fine"] is not None:
        state["generator_fine"] = generator.render_kwargs_train["network_fine"].state_dict()
    if encoder is not None:
        state["encoder"] = encoder.state_dict()
    if z_fuser is not None:
        state["z_fuser"] = z_fuser.state_dict()
    if gain_head is not None:
        state["gain_head"] = gain_head.state_dict()
    if gain_param is not None:
        state["gain_param"] = gain_param.detach().cpu()
    ckpt_path = ckpt_dir / f"checkpoint_step{step:05d}.pt"
    torch.save(state, ckpt_path)
    print(f"💾 Checkpoint gespeichert: {ckpt_path}", flush=True)


def compute_psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
    mse = torch.mean((pred - target) ** 2).item()
    if mse <= 0:
        return float("inf")
    return -10.0 * math.log10(mse + 1e-12)


def save_depth_profile(
    step,
    generator,
    z_latent,
    ct_vol,
    act_vol,
    outdir: Path,
    proj_ap=None,
    proj_pa=None,
    target_ap=None,
    target_pa=None,
    target_ap_counts=None,
    target_pa_counts=None,
    signal_quantile: float = 0.99,
    bg_quantile: float = 0.10,
    seed: int = 123,
):
    """
    Speichert Tiefenprofile (λ/μ/Prediction) entlang ausgewählter Strahlen für Analyse/Debugging.
    """
    H, W = generator.H, generator.W

    ap_img = proj_ap.detach().view(H, W).cpu().numpy() if proj_ap is not None else None
    pa_img = proj_pa.detach().view(H, W).cpu().numpy() if proj_pa is not None else None

    def proj_to_img(t: Optional[torch.Tensor]):
        if t is None:
            return None
        t = t.detach()
        if t.numel() == H * W:
            t = t.view(H, W)
        elif t.dim() >= 2 and t.shape[-2:] == (H, W):
            t = t.reshape(-1, H, W)[0]
        else:
            return None
        return t.cpu().numpy()

    ap_counts_img = proj_to_img(target_ap_counts)
    pa_counts_img = proj_to_img(target_pa_counts)
    ap_target_img = proj_to_img(target_ap)
    pa_target_img = proj_to_img(target_pa)

    use_counts_targets = ap_counts_img is not None and pa_counts_img is not None
    if use_counts_targets:
        target_ap_img = ap_counts_img
        target_pa_img = pa_counts_img
        target_kind = "counts"
    else:
        target_ap_img = ap_target_img
        target_pa_img = pa_target_img
        target_kind = "norm" if (target_ap_img is not None or target_pa_img is not None) else "none"

    if target_ap_img is not None and target_pa_img is not None:
        score_map = np.maximum(target_ap_img, target_pa_img)
    elif target_ap_img is not None:
        score_map = target_ap_img.copy()
    elif target_pa_img is not None:
        score_map = target_pa_img.copy()
    else:
        score_map = None

    ap_title_img = target_ap_img
    pa_title_img = target_pa_img
    if ap_title_img is None:
        ap_title_img = proj_to_img(proj_ap)
    if pa_title_img is None:
        pa_title_img = proj_to_img(proj_pa)

    signal_quantile = float(np.clip(signal_quantile, 0.0, 1.0))
    bg_quantile = float(np.clip(bg_quantile, 0.0, 1.0))
    rng = np.random.default_rng(int(seed))

    act_data = act_vol.detach().cpu().numpy() if act_vol is not None else None
    act_masks = None
    if act_data is not None:
        if act_data.ndim == 4:
            act_data = act_data.squeeze(0)
        act_zero = act_data < 1e-6
        act_nonzero = act_data > 1e-6
        act_masks = (act_zero.max(axis=0), act_nonzero.max(axis=0))

    def extract_curve(vol: torch.Tensor, y_idx: int, x_idx: int):
        if vol is None:
            return None, None
        vol = vol.detach()
        if vol.dim() == 4:
            vol = vol.squeeze(0)
        if vol.dim() != 3:
            return None, None
        D, H_loc, W_loc = vol.shape[-3:]
        if not (0 <= y_idx < H_loc and 0 <= x_idx < W_loc):
            return None, None
        curve = vol[:, y_idx, x_idx].cpu().numpy()
        z_coords = idx_to_coord(torch.arange(D, device=vol.device), D, generator.radius if not isinstance(generator.radius, tuple) else generator.radius[1])
        return curve, z_coords

    def pick_ray_indices(num_zero: int = 1, num_active: int = 3):
        chosen = []

        def add_unique(idx):
            if idx is None:
                return False
            if idx in chosen:
                return False
            chosen.append(idx)
            return True

        def dist(a, b):
            return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)

        def is_far_enough(idx):
            return all(dist(idx, c) > 8 for c in chosen)

        # Deterministische, target-basierte Auswahl: 1x Background (niedrig),
        # 3x Signal (oberstes Quantil), bevorzugt mit Count-Targets.
        if score_map is not None and np.isfinite(score_map).any():
            valid_mask = np.isfinite(score_map)
            scores_valid = score_map[valid_mask]
            if scores_valid.size > 0:
                q_sig = float(np.quantile(scores_valid, signal_quantile))
                q_bg = float(np.quantile(scores_valid, bg_quantile))

                zero_mask = (score_map == 0) & valid_mask
                valid_count = int(scores_valid.size)
                zero_count = int(zero_mask.sum())
                bg_needed = max(1, int(math.ceil(bg_quantile * valid_count)))
                many_zeros = zero_count >= bg_needed or q_bg <= 0.0
                bg_mask = zero_mask if many_zeros else ((score_map <= q_bg) & valid_mask)
                sig_mask = (score_map >= q_sig) & valid_mask

                def sample_from_coords(coords: np.ndarray):
                    if coords.size == 0:
                        return None
                    idx = int(rng.integers(0, len(coords)))
                    y, x = coords[idx]
                    return int(y), int(x)

                bg_idx = None
                bg_coords = np.argwhere(bg_mask)
                if bg_coords.size > 0:
                    bg_scores = score_map[bg_mask]
                    median_val = float(np.median(bg_scores))
                    diffs = np.abs(bg_scores - median_val)
                    min_diff = float(diffs.min())
                    near_idx = np.flatnonzero(np.isclose(diffs, min_diff, rtol=0.0, atol=1e-12))
                    bg_idx = sample_from_coords(bg_coords[near_idx]) if near_idx.size > 0 else sample_from_coords(bg_coords)
                if bg_idx is None:
                    min_val = float(np.min(scores_valid))
                    min_mask = valid_mask & np.isclose(score_map, min_val, rtol=0.0, atol=1e-12)
                    bg_idx = sample_from_coords(np.argwhere(min_mask if min_mask.any() else valid_mask))
                add_unique(bg_idx)

                sig_coords = np.argwhere(sig_mask)
                if sig_coords.size > 0:
                    sig_scores = score_map[sig_mask]
                    order = np.argsort(sig_scores)[::-1]
                    for idx in order:
                        if len(chosen) >= num_zero + num_active:
                            break
                        y, x = sig_coords[int(idx)]
                        add_unique((int(y), int(x)))

                if len(chosen) < num_zero + num_active:
                    valid_coords = np.argwhere(valid_mask)
                    valid_scores = score_map[valid_mask]
                    order_global = np.argsort(valid_scores)[::-1]
                    for idx in order_global:
                        if len(chosen) >= num_zero + num_active:
                            break
                        y, x = valid_coords[int(idx)]
                        add_unique((int(y), int(x)))

                attempts = 0
                while len(chosen) < num_zero + num_active and attempts < 128:
                    attempts += 1
                    cand = sample_from_coords(np.argwhere(valid_mask))
                    add_unique(cand)

                if chosen:
                    return chosen[: num_zero + num_active]

        def pick_from_mask(mask, prefer_high: bool):
            if mask is None:
                return None
            mask = mask.copy()
            chosen_mask = np.zeros_like(mask, dtype=bool)
            for y, x in chosen:
                if 0 <= y < H and 0 <= x < W:
                    chosen_mask[y, x] = True
            mask = mask & (~chosen_mask)
            if not mask.any():
                return None
            coords = np.argwhere(mask)
            if coords.size == 0:
                return None

            weight_map = None
            if ap_img is not None and pa_img is not None:
                weight_map = ap_img + pa_img
            if weight_map is not None:
                weights = weight_map[mask]
                if weights.size == 0:
                    weights = None
                else:
                    if prefer_high:
                        weights = weights - weights.min() + 1e-6
                    else:
                        weights = weights.max() - weights + 1e-6
                    if not np.isfinite(weights).any() or np.sum(weights) <= 0:
                        weights = None
            if weights is None:
                np.random.shuffle(coords)
                for y, x in coords:
                    if is_far_enough((int(y), int(x))):
                        return int(y), int(x)
                y, x = coords[0]
                return int(y), int(x)

            for _ in range(min(len(coords), 64)):
                idx = np.random.choice(len(coords), p=weights / weights.sum())
                y, x = coords[idx]
                if is_far_enough((int(y), int(x))):
                    return int(y), int(x)
            y, x = coords[np.argmax(weights)]
            return int(y), int(x)

        def pick_proj_extreme(func):
            if ap_img is None or pa_img is None:
                return None
            combo = (ap_img + pa_img).copy()
            for y, x in chosen:
                if 0 <= y < H and 0 <= x < W:
                    combo[y, x] = np.nan
            try:
                y, x = np.unravel_index(func(combo), combo.shape)
            except ValueError:
                return None
            return int(y), int(x)

        ct_pos_mask = None
        if ct_vol is not None:
            ct_data = ct_vol.detach()
            if ct_data.dim() == 4:
                ct_data = ct_data.squeeze(0)
            if ct_data.dim() == 3:
                ct_depth_max = ct_data.max(dim=0).values.cpu().numpy()
                ct_pos_mask = ct_depth_max > 1e-8

        def combine_mask(base_mask, require_ct: bool):
            if base_mask is None:
                return None
            mask = base_mask.astype(bool)
            if ct_pos_mask is not None and require_ct:
                mask = mask & ct_pos_mask
            return mask

        zero_mask = nonzero_mask = None
        if act_data is not None and act_masks is not None:
            zero_mask, nonzero_mask = act_masks

        zero_needed = max(num_zero, 0)
        active_needed = max(num_active, 0)

        if zero_needed > 0:
            for mask in (combine_mask(zero_mask, True), zero_mask):
                if zero_needed <= 0:
                    break
                if add_unique(pick_from_mask(mask, prefer_high=False)):
                    zero_needed -= 1

        if active_needed > 0:
            for _ in range(active_needed):
                idx = pick_from_mask(combine_mask(nonzero_mask, True), prefer_high=True)
                if not add_unique(idx):
                    break
                active_needed -= 1
            while active_needed > 0:
                idx = pick_from_mask(nonzero_mask, prefer_high=True)
                if idx is None:
                    break
                if add_unique(idx):
                    active_needed -= 1

        if zero_needed > 0:
            if add_unique(pick_proj_extreme(np.nanargmin)):
                zero_needed -= 1

        while active_needed > 0:
            idx = pick_proj_extreme(np.nanargmax)
            if idx is None:
                break
            if add_unique(idx):
                active_needed -= 1

        fixed_coords = [(72, 428), (69, 336)]
        for y_raw, x_raw in fixed_coords:
            if len(chosen) >= num_zero + num_active:
                break
            y = int(np.clip(y_raw, 0, H - 1))
            x = int(np.clip(x_raw, 0, W - 1))
            add_unique((y, x))
        rel_coords = [(0.5, 0.5), (0.25, 0.75), (0.75, 0.25)]
        for ry, rx in rel_coords:
            if len(chosen) >= num_zero + num_active:
                break
            y = int(np.clip(round((H - 1) * ry), 0, H - 1))
            x = int(np.clip(round((W - 1) * rx), 0, W - 1))
            add_unique((y, x))

        if ap_img is not None and pa_img is not None and len(chosen) < num_zero + num_active:
            max_y, max_x = np.unravel_index(np.argmax(ap_img + pa_img), ap_img.shape)
            add_unique((int(max_y), int(max_x)))

        while len(chosen) < num_zero + num_active:
            y = int(np.random.randint(0, H))
            x = int(np.random.randint(0, W))
            add_unique((y, x))

        return chosen[: num_zero + num_active]

    def first_shape(vol_a, vol_b):
        for v in (vol_a, vol_b):
            if v is None:
                continue
            data = v.squeeze(0).detach().cpu().numpy() if v.dim() == 4 else v.detach().cpu().numpy()
            if data.ndim == 3:
                return data.shape
        return None

    target_shape = first_shape(ct_vol, act_vol)
    if target_shape is None:
        return

    D = target_shape[0]
    radius = generator.radius
    if isinstance(radius, tuple):
        radius = radius[1]

    num_zero, num_active = 1, 3
    target_total = max(num_zero + num_active, 1)
    cache_attr = "_depth_profile_rays_cache"
    cache = getattr(generator, cache_attr, None)
    ray_indices_cache = None
    if isinstance(cache, dict):
        cached_indices = cache.get("indices")
        cached_shape = cache.get("shape")
        cached_total = cache.get("total")
        cached_sig_q = cache.get("signal_quantile")
        cached_bg_q = cache.get("bg_quantile")
        cached_seed = cache.get("seed")
        cached_target_kind = cache.get("target_kind")
        if (
            cached_indices
            and cached_shape == (generator.H, generator.W)
            and cached_total == target_total
            and cached_sig_q == signal_quantile
            and cached_bg_q == bg_quantile
            and cached_seed == int(seed)
            and cached_target_kind == target_kind
        ):
            ray_indices_cache = cached_indices

    if ray_indices_cache is None:
        ray_indices = pick_ray_indices(num_zero=num_zero, num_active=num_active)
        setattr(
            generator,
            cache_attr,
            {
                "indices": list(ray_indices),
                "shape": (generator.H, generator.W),
                "total": target_total,
                "signal_quantile": signal_quantile,
                "bg_quantile": bg_quantile,
                "seed": int(seed),
                "target_kind": target_kind,
            },
        )
    else:
        ray_indices = ray_indices_cache

    depth_idx = torch.arange(D, device=generator.device)
    z_coords = idx_to_coord(depth_idx, D, radius)
    depth_axis = np.linspace(0.0, 1.0, D)
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(ray_indices), figsize=(4 * len(ray_indices), 4), sharex=True, sharey=True)
    if not isinstance(axes, np.ndarray):
        axes = [axes]

    for ax, (y_idx, x_idx) in zip(axes, ray_indices):
        curves = []
        labels = []
        curve_ct = extract_curve(ct_vol, y_idx, x_idx) if ct_vol is not None else (None, None)
        curve_act = extract_curve(act_vol, y_idx, x_idx) if act_vol is not None else (None, None)

        if curve_ct[0] is not None:
            curves.append(normalize_curve(curve_ct[0].copy()))
            labels.append("μ (CT)")
        if curve_act[0] is not None:
            curves.append(normalize_curve(curve_act[0].copy()))
            labels.append("Aktivität (GT)")

        x_coord = idx_to_coord(torch.tensor(x_idx, device=generator.device), target_shape[2], radius)
        y_coord = idx_to_coord(torch.tensor(y_idx, device=generator.device), target_shape[1], radius)
        coords = torch.stack((x_coord.repeat(D), y_coord.repeat(D), z_coords), dim=1)
        pred = query_emission_at_points(generator, z_latent, coords).detach().cpu().numpy()
        curves.append(normalize_curve(pred.copy()))
        labels.append("Aktivität (NeRF)")

        for curve, label in zip(curves, labels):
            ax.plot(depth_axis, curve, label=label)

        title_extra = []
        if ap_title_img is not None:
            title_extra.append(f"I_AP={ap_title_img[y_idx, x_idx]:.2e}")
        if pa_title_img is not None:
            title_extra.append(f"I_PA={pa_title_img[y_idx, x_idx]:.2e}")
        aux = " | ".join(title_extra)
        ax.set_title(f"({y_idx},{x_idx})" + (f"\n{aux}" if aux else ""))
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.2)
        ax.legend(loc="upper right", fontsize=8)

    axes[0].set_ylabel("normierte Intensität")
    for ax in axes:
        ax.set_xlabel("Tiefe (anterior → posterior)")
    fig.suptitle(f"Depth-Profile @ step {step:05d}")
    fig.tight_layout()
    outdir.mkdir(parents=True, exist_ok=True)
    fig.savefig(outdir / f"depth_profile_step_{step:05d}.png", dpi=150)
    plt.close(fig)


def evaluate_pixel_subsets(
    generator,
    z_latent,
    rays_cache,
    subsets: Dict[str, Optional[torch.Tensor]],
    ap_flat_proc: torch.Tensor,
    pa_flat_proc: torch.Tensor,
    rays_per_eval: Optional[int],
    bg_weight: float,
    weight_threshold: float,
    pa_xflip: bool,
    ct_context=None,
    W: int = None,
    scale_ap: Optional[float] = None,
    scale_pa: Optional[float] = None,
    loss_fn=poisson_nll,
    pred_scale: float = 1.0,
    gain: Optional[torch.Tensor] = None,
):
    """Evaluiert Loss/PSNR/MAE auf gemeinsamen Pixel-Indizes für AP+PA (Loss gemittelt über Views)."""
    prev_flag = generator.use_test_kwargs
    generator.eval()
    generator.use_test_kwargs = True
    results = {}
    with torch.no_grad():
        for name, idx_all in subsets.items():
            if idx_all is None or idx_all.numel() == 0:
                results[name] = None
                continue
            n_sel = idx_all.numel() if rays_per_eval is None else min(idx_all.numel(), rays_per_eval)
            idx_ap = idx_all if rays_per_eval is None else sample_split_indices(idx_all, n_sel)
            idx_pa = map_pa_indices_torch(idx_ap, W, pa_xflip)

            ray_batch_ap = slice_rays(rays_cache["ap"], idx_ap)
            ray_batch_pa = slice_rays(rays_cache["pa"], idx_pa)

            pred_ap_raw, _ = render_minibatch(generator, z_latent, ray_batch_ap, ct_context=ct_context)
            pred_pa_raw, _ = render_minibatch(generator, z_latent, ray_batch_pa, ct_context=ct_context)

            if loss_fn == poisson_nll:
                lambda_ap_used = F.softplus(pred_ap_raw) - math.log(2.0)
                lambda_pa_used = F.softplus(pred_pa_raw) - math.log(2.0)
                lambda_ap_used = lambda_ap_used.clamp_min(1e-6)
                lambda_pa_used = lambda_pa_used.clamp_min(1e-6)
            else:
                lambda_ap_used = pred_ap_raw
                lambda_pa_used = pred_pa_raw

            if pred_scale != 1.0:
                lambda_ap_used = lambda_ap_used * float(pred_scale)
                lambda_pa_used = lambda_pa_used * float(pred_scale)
            if gain is not None:
                lambda_ap_used = lambda_ap_used * gain
                lambda_pa_used = lambda_pa_used * gain
            pred_ap = lambda_ap_used
            pred_pa = lambda_pa_used

            target_ap = ap_flat_proc[0, idx_ap].unsqueeze(0)
            target_pa = pa_flat_proc[0, idx_pa].unsqueeze(0)

            if (target_ap < 0).any() or (target_pa < 0).any():
                print("[WARN] Negative projection targets detected in eval.", flush=True)
            if loss_fn == poisson_nll:
                if not torch.isfinite(pred_ap).all() or not torch.isfinite(pred_pa).all():
                    raise RuntimeError("Non-finite lambda in Poisson projection loss (eval).")
                if (pred_ap <= 0).any() or (pred_pa <= 0).any():
                    raise RuntimeError("Non-positive lambda in Poisson projection loss (eval).")

            weight_ap = build_loss_weights(target_ap, bg_weight, weight_threshold)
            weight_pa = build_loss_weights(target_pa, bg_weight, weight_threshold)

            # ---------------------------------------------------------------------------
            loss_ap = loss_fn(pred_ap, target_ap, weight=weight_ap)
            loss_pa = loss_fn(pred_pa, target_pa, weight=weight_pa)
            loss_total = 0.5 * (loss_ap + loss_pa)

            psnr_ap = compute_psnr(pred_ap, target_ap)
            psnr_pa = compute_psnr(pred_pa, target_pa)
            mae_ap = torch.mean(torch.abs(pred_ap - target_ap)).item()
            mae_pa = torch.mean(torch.abs(pred_pa - target_pa)).item()

            phys_metrics = None
            if scale_ap is not None and scale_pa is not None:
                scale_ap_f = float(scale_ap)
                scale_pa_f = float(scale_pa)
                pred_ap_phys = pred_ap * scale_ap_f
                pred_pa_phys = pred_pa * scale_pa_f
                target_ap_phys = target_ap * scale_ap_f
                target_pa_phys = target_pa * scale_pa_f
                psnr_ap_phys = compute_psnr(pred_ap_phys, target_ap_phys)
                psnr_pa_phys = compute_psnr(pred_pa_phys, target_pa_phys)
                mae_ap_phys = torch.mean(torch.abs(pred_ap_phys - target_ap_phys)).item()
                mae_pa_phys = torch.mean(torch.abs(pred_pa_phys - target_pa_phys)).item()
                phys_metrics = {
                    "psnr": 0.5 * (psnr_ap_phys + psnr_pa_phys),
                    "mae": 0.5 * (mae_ap_phys + mae_pa_phys),
                    "view": {
                        "ap": {"psnr": psnr_ap_phys, "mae": mae_ap_phys},
                        "pa": {"psnr": psnr_pa_phys, "mae": mae_pa_phys},
                    },
                }

            results[name] = {
                "loss": loss_total.item(),
                "loss_ap": loss_ap.item(),
                "loss_pa": loss_pa.item(),
                "psnr": 0.5 * (psnr_ap + psnr_pa),
                "mae": 0.5 * (mae_ap + mae_pa),
                "pred_mean": ((float(pred_ap.mean()), float(pred_pa.mean()))),
                "target_mean": ((float(target_ap.mean()), float(target_pa.mean()))),
                "view": {
                    "ap": {"loss": loss_ap.item(), "psnr": psnr_ap, "mae": mae_ap},
                    "pa": {"loss": loss_pa.item(), "psnr": psnr_pa, "mae": mae_pa},
                },
                "phys": phys_metrics,
            }

    if prev_flag:
        generator.eval()
    else:
        generator.train()

    return results


def sample_act_points(
    act: torch.Tensor, nsamples: int, radius: float, pos_fraction: float = 0.5, pos_threshold: float = 1e-8
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Ziehe zufällige Voxel (coords, values) aus act.npy, halb aus aktiven Voxeln (ACT>0), halb global.
    Gibt zusätzlich einen Bool-Flag pro Sample zurück, der anzeigt, ob es aus ACT>0 stammt.
    """
    if act is None:
        raise ValueError("act tensor missing despite act-loss-weight > 0.")
    # Unterscheide (1,D,H,W) vs (D,H,W)
    if act.dim() == 4:
        act = act.squeeze(0)
    D, H, W = act.shape[-3:]
    flat = act.view(-1)
    nsamples = min(nsamples, flat.numel())
    if nsamples <= 0:
        empty = torch.zeros((0,), device=act.device)
        return empty.reshape(0, 3), empty, empty.bool()

    # Split: pos_fraction aus ACT>pos_threshold, Rest uniform
    num_pos = int(round(float(nsamples) * float(pos_fraction)))
    num_pos = max(0, min(num_pos, nsamples))
    num_all = nsamples - num_pos

    pos_mask = flat > pos_threshold
    pos_idx = pos_mask.nonzero(as_tuple=False).squeeze(-1)

    idx_parts = []
    flag_parts = []

    if num_pos > 0 and pos_idx.numel() > 0:
        perm = torch.randint(0, pos_idx.numel(), (num_pos,), device=act.device)
        idx_pos = pos_idx[perm]
        idx_parts.append(idx_pos)
        flag_parts.append(torch.ones_like(idx_pos, dtype=torch.bool))
    else:
        # Fallback: keine aktiven Voxeln gefunden -> alles aus globalem Sampling ziehen
        num_all = nsamples

    if num_all > 0:
        idx_all = torch.randint(0, flat.numel(), (num_all,), device=act.device)
        idx_parts.append(idx_all)
        flag_parts.append(torch.zeros_like(idx_all, dtype=torch.bool))

    idx = torch.cat(idx_parts, dim=0)
    pos_flags = torch.cat(flag_parts, dim=0)
    values = flat[idx]

    hw = H * W
    z_idx = idx // hw
    y_idx = (idx % hw) // W
    x_idx = idx % W

    coords = torch.stack(
        (
            idx_to_coord(x_idx, W, radius),
            idx_to_coord(y_idx, H, radius),
            idx_to_coord(z_idx, D, radius),
        ),
        dim=1,
    )
    return coords, values, pos_flags


def query_emission_at_points(
    generator, z_latent, coords: torch.Tensor, return_raw: bool = False
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Fragt das NeRF an frei gewählten Koordinaten ab (ohne Integration)."""
    if coords.numel() == 0:
        empty = torch.tensor([], device=coords.device)
        return (empty, empty) if return_raw else empty
    render_kwargs = generator.render_kwargs_train
    network_fn = render_kwargs["network_fn"]
    network_query_fn = render_kwargs["network_query_fn"]
    pts = coords.unsqueeze(0)
    raw = network_query_fn(pts, None, network_fn, features=z_latent)
    raw = raw.view(-1, raw.shape[-1])
    pred = F.softplus(raw[:, 0])
    if return_raw:
        return pred, raw[:, 0]
    return pred


def idx_to_coord(idx: torch.Tensor, size: int, radius: float) -> torch.Tensor:
    if size <= 1:
        return torch.zeros_like(idx, dtype=torch.float32)
    return ((idx.float() / (size - 1)) - 0.5) * 2.0 * radius


def normalize_curve(arr: np.ndarray) -> np.ndarray:
    arr = arr - np.min(arr)
    maxv = np.max(arr)
    if maxv > 1e-8:
        arr = arr / maxv
    return arr



def sample_ct_pairs(ct: torch.Tensor, nsamples: int, thresh: float, radius: float):
    """Wählt Voxel-Paare (z,z+1) mit geringer CT-Änderung entlang der Tiefe."""
    if ct.dim() == 4:
        ct = ct.squeeze(0)
    D, H, W = ct.shape[-3:]
    if D < 2:
        return None
    # Differenz entlang z, kleine Gradienten => weiches Gewebe -> Loss erzwingt glatte Emission
    diff = torch.abs(ct[1:, :, :] - ct[:-1, :, :])
    ct_max = torch.max(ct)
    rel_diff = diff / (ct_max + 1e-8) if ct_max > 0 else diff
    mask = diff < thresh
    mask = mask | (rel_diff < thresh)
    valid_idx = mask.nonzero(as_tuple=False)
    if valid_idx.numel() == 0:
        return None
    nsamples = min(nsamples, valid_idx.shape[0])
    perm = torch.randperm(valid_idx.shape[0], device=ct.device)[:nsamples]
    sel = valid_idx[perm]
    z = sel[:, 0]
    y = sel[:, 1]
    x = sel[:, 2]
    z_next = z + 1

    coords1 = torch.stack(
        (idx_to_coord(x, W, radius), idx_to_coord(y, H, radius), idx_to_coord(z, D, radius)),
        dim=1,
    )
    coords2 = torch.stack(
        (idx_to_coord(x, W, radius), idx_to_coord(y, H, radius), idx_to_coord(z_next, D, radius)),
        dim=1,
    )
    weights = torch.clamp(1.0 - diff[sel[:, 0], sel[:, 1], sel[:, 2]], min=0.0)
    return coords1, coords2, weights


def train():
    print(f"▶ {__VERSION__} – starte Training", flush=True)
    args = parse_args()
    global DEBUG_PRINTS
    DEBUG_PRINTS = bool(args.debug_prints)
    hybrid_enabled = bool(args.hybrid)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA device required – please launch on a GPU node.")

    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True
    set_seed(args.seed)

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    nerf_cfg = config.setdefault("nerf", {})
    nerf_cfg.setdefault("atten_scale", ATTEN_SCALE_DEFAULT)
    if args.atten_scale != ATTEN_SCALE_DEFAULT:
        nerf_cfg["atten_scale"] = float(args.atten_scale)

    data_cfg = config.setdefault("data", {})
    if args.normalize_targets:
        print("⚠️ --normalize-targets ist veraltet und wird ignoriert.", flush=True)
    data_cfg.setdefault("act_scale", 1.0)
    data_cfg["debug_proj_stats"] = bool(args.debug_proj_stats)
    data_cfg["ray_split_ratio"] = float(args.ray_split)
    training_cfg = config.setdefault("training", {})
    training_cfg.setdefault("val_interval", 0)
    training_cfg.setdefault("tv_weight", 0.001)
    training_cfg["tv_weight"] = args.tv_weight
    training_cfg.setdefault("ray_tv_weight", 0.0)
    training_cfg["ray_tv_weight"] = args.ray_tv_weight
    training_cfg.setdefault("ray_tv_edge_aware", False)
    training_cfg["ray_tv_edge_aware"] = bool(args.ray_tv_edge_aware)
    training_cfg.setdefault("ray_tv_alpha", 0.0)
    training_cfg["ray_tv_alpha"] = float(args.ray_tv_alpha)
    training_cfg.setdefault("ray_tv_w_clamp_min", 0.0)
    training_cfg["ray_tv_w_clamp_min"] = float(args.ray_tv_w_clamp_min)
    training_cfg.setdefault("bg_depth_mass_weight", 0.0)
    training_cfg["bg_depth_mass_weight"] = float(args.bg_depth_mass_weight)
    training_cfg.setdefault("bg_depth_eps", 1e-10)
    training_cfg["bg_depth_eps"] = float(args.bg_depth_eps)
    training_cfg.setdefault("bg_depth_mode", "integral")
    training_cfg["bg_depth_mode"] = str(args.bg_depth_mode)
    training_cfg.setdefault("act_samples", 16384)
    training_cfg.setdefault("act_pos_weight", 2.0)
    training_cfg.setdefault("act_pos_fraction", 0.5)
    training_cfg.setdefault("act_pos_threshold", 1e-8)
    if args.act_samples is None:
        args.act_samples = int(training_cfg.get("act_samples", 16384))
    else:
        training_cfg["act_samples"] = args.act_samples
    if args.act_pos_weight is None:
        args.act_pos_weight = float(training_cfg.get("act_pos_weight", 2.0))
    else:
        training_cfg["act_pos_weight"] = args.act_pos_weight
    if args.act_pos_fraction is None:
        args.act_pos_fraction = float(training_cfg.get("act_pos_fraction", 0.5))
    else:
        training_cfg["act_pos_fraction"] = args.act_pos_fraction
    if args.act_pos_threshold is None:
        args.act_pos_threshold = float(training_cfg.get("act_pos_threshold", 1e-8))
    else:
        training_cfg["act_pos_threshold"] = args.act_pos_threshold
    training_cfg.setdefault("ct_loss_weight", 0.0)
    training_cfg.setdefault("ct_threshold", 0.05)
    training_cfg.setdefault("ct_samples", 8192)
    training_cfg["ct_loss_weight"] = args.ct_loss_weight
    training_cfg["ct_threshold"] = args.ct_threshold
    training_cfg["ct_samples"] = args.ct_samples
    training_cfg.setdefault("z_reg_weight", 0.0)
    training_cfg["z_reg_weight"] = args.z_reg_weight
    if hybrid_enabled and args.act_loss_weight <= 0.0:
        print("[WARN] Hybrid aktiv, aber --act-loss-weight <= 0: ACT-Hauptloss ist deaktiviert.", flush=True)
    if hybrid_enabled and "ct_prefer_raw" not in data_cfg:
        data_cfg["ct_prefer_raw"] = True

    print(f"📂 CWD: {Path.cwd().resolve()}", flush=True)
    outdir = Path(config.get("training", {}).get("outdir", "./results_spect")).expanduser().resolve()
    (outdir / "preview").mkdir(parents=True, exist_ok=True)
    print(f"🗂️ Output-Ordner: {outdir}", flush=True)
    log_effective_config(outdir, config, args)
    ckpt_dir = outdir / "checkpoints"
    log_path = outdir / "train_log.csv"
    init_log_file(log_path)
    hybrid_log_path = None
    if hybrid_enabled:
        hybrid_log_path = outdir / "hybrid_stats.csv"
        init_hybrid_log_file(hybrid_log_path)

    dataset = None
    try:
        dataset, hwfr, _ = get_data(config)
    except Exception as exc:
        if args.smoke_test:
            print(f"[smoke-test] get_data failed ({exc.__class__.__name__}): using synthetic batch.", flush=True)
            hwfr = build_hwfr_from_config(data_cfg)
        else:
            raise
    config["data"]["hwfr"] = hwfr

    batch_size = config["training"]["batch_size"]
    if batch_size != 1:
        raise ValueError("This mini-training script currently assumes batch_size == 1.")

    dataloader = None
    if dataset is not None:
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=config["training"]["nworkers"],
            pin_memory=True,
            drop_last=False,
        )

    act_global_scale = float(data_cfg.get("act_scale", 1.0))
    if act_global_scale != 1.0:
        print(f"ℹ️ ACT/λ globaler Faktor (im Loader angewandt): x{act_global_scale}", flush=True)
    if DEBUG_PRINTS:
        print(f"[DEBUG] act_scale={act_global_scale}", flush=True)
    ray_split_ratio = float(data_cfg.get("ray_split_ratio", 0.8))
    ray_split_enabled = bool(args.ray_split_enable)
    ray_split_mode = str(args.ray_split_mode)
    ray_split_seed = int(args.ray_split_seed)
    ray_split_tile = int(max(1, args.ray_split_tile))
    ray_fg_thr = args.ray_fg_thr
    ray_fg_quantile = float(args.ray_fg_quantile)
    pa_xflip = bool(args.pa_xflip)
    ray_train_fg_frac = float(np.clip(args.ray_train_fg_frac, 0.0, 1.0))
    log_proj_metrics_physical = bool(args.log_proj_metrics_physical)
    val_interval = int(training_cfg.get("val_interval", 0) or 0)
    tv_weight = float(training_cfg.get("tv_weight", 0.0))
    ray_tv_weight = float(training_cfg.get("ray_tv_weight", 0.0))
    ray_tv_edge_aware = bool(training_cfg.get("ray_tv_edge_aware", False))
    ray_tv_alpha = float(training_cfg.get("ray_tv_alpha", 0.0))
    ray_tv_w_clamp_min = float(training_cfg.get("ray_tv_w_clamp_min", 0.0))
    depth_sanity_every = int(max(0, args.depth_sanity_every))
    depth_checks_active = depth_sanity_every > 0
    depth_grad_zero_streak = 0

    generator = build_models(config)
    generator.to(device)
    generator.train()
    generator.use_test_kwargs = False  # enforce training kwargs

    def _depth_grad_norm(loss_term: torch.Tensor, module_candidate) -> float:
        def _local_grad_norm(loss_term_local: torch.Tensor, module_local: torch.nn.Module) -> float:
            if loss_term_local is None or not loss_term_local.requires_grad:
                return 0.0
            params = [p for p in module_local.parameters() if p.requires_grad]
            if not params:
                return 0.0
            grads = torch.autograd.grad(loss_term_local, params, retain_graph=True, allow_unused=True)
            grads = [g for g in grads if g is not None]
            if not grads:
                return 0.0
            flat = torch.cat([g.reshape(-1) for g in grads])
            return float(flat.norm().detach().cpu().item())

        target = (
            module_candidate
            if isinstance(module_candidate, torch.nn.Module)
            else (generator if isinstance(generator, torch.nn.Module) else None)
        )
        if target is None:
            print("[WARN][depth] Grad-Target ist kein nn.Module; skippe Depth-Grad-Norm.", flush=True)
            return 0.0
        if "grad_norm_of_module" in globals():
            try:
                return grad_norm_of_module(loss_term, target)
            except Exception as exc:
                print(
                    f"[WARN][depth] grad_norm_of_module failed ({exc.__class__.__name__}); nutze Fallback.",
                    flush=True,
                )
        return _local_grad_norm(loss_term, target)

    # always provide AP/PA fallback poses if not already configured
    generator.set_fixed_ap_pa(radius=hwfr[3])

    z_dim = config["z_dist"]["dim"]
    z_train = torch.nn.Parameter(torch.zeros(1, z_dim, device=device))
    torch.nn.init.normal_(z_train, mean=0.0, std=1.0)
    encoder = None
    z_fuser = None
    z_enc_alpha = float(args.z_enc_alpha)
    gain_head = None
    gain_param = None
    if hybrid_enabled:
        enc_in_ch = 2 + (1 if args.encoder_use_ct else 0)
        encoder = ProjectionEncoder(in_ch=enc_in_ch, z_dim=z_dim, base_ch=32).to(device)
        z_fuser = nn.Sequential(nn.Linear(z_dim, z_dim), nn.LayerNorm(z_dim)).to(device)
        if args.proj_target_source == "counts":
            if args.proj_gain_source == "z_enc":
                gain_head = nn.Linear(z_dim, 1).to(device)
            elif args.proj_gain_source == "scalar":
                gain_param = nn.Parameter(torch.zeros(1, device=device))
        encoder.train()
        print(
            f"[hybrid] Encoder init: in_ch={enc_in_ch}, z_dim={z_dim} | z_enc_alpha={z_enc_alpha}",
            flush=True,
        )

    # --- Sofortiger Smoke-Test ---
    # Einmal vor dem eigentlichen Training rendern, um Setup/NaNs zu prüfen
    with torch.no_grad():
        generator.eval()
        generator.use_test_kwargs = True
        z_smoke = z_train.detach()
        proj_ap, _, _, _ = generator.render_from_pose(z_smoke, generator.pose_ap)
        proj_pa, _, _, _ = generator.render_from_pose(z_smoke, generator.pose_pa)
        generator.train()
        generator.use_test_kwargs = False

    H, W = generator.H, generator.W
    ap_np = proj_ap[0].reshape(H, W).detach().cpu().numpy()
    pa_np = proj_pa[0].reshape(H, W).detach().cpu().numpy()
    smoke_dir = outdir / "preview"
    smoke_dir.mkdir(parents=True, exist_ok=True)
    save_img(ap_np, smoke_dir / "smoke_AP.png", title="Smoke AP")
    save_img(pa_np, smoke_dir / "smoke_PA.png", title="Smoke PA")
    print("✅ Smoke-Test gespeichert:", flush=True)

    rays_cache = {
        "ap": build_pose_rays(generator, generator.pose_ap),
        "pa": build_pose_rays(generator, generator.pose_pa),
    }
    # Gesamtzahl der Pixel bestimmt die Maximalzahl möglicher Strahlen
    num_pixels = generator.H * generator.W
    pixel_split_np: Optional[PixelSplit] = None
    ray_indices: Dict[str, Dict[str, Optional[torch.Tensor]]] = {}

    def parse_fg_threshold(raw_thr) -> Tuple[float, bool]:
        try:
            return float(raw_thr), False
        except Exception:
            if isinstance(raw_thr, str) and raw_thr.strip().lower() == "quantile":
                return 0.0, True
            raise

    ray_fg_thr_value, ray_fg_force_quantile = parse_fg_threshold(ray_fg_thr)
    if ray_split_mode not in ("tile_random", "stratified_intensity"):
        raise ValueError(f"Unknown ray_split_mode: {ray_split_mode}")

    def map_pa_indices_torch(idx: torch.Tensor, W: int, do_flip: bool) -> torch.Tensor:
        if not do_flip:
            return idx
        y = idx // W
        x = idx % W
        return y * W + (W - 1 - x)

    def _to_torch(arr: Optional[np.ndarray]):
        if arr is None:
            return None
        return torch.from_numpy(arr.astype(np.int64)).long().to(device, non_blocking=True)

    def _log_split(split: PixelSplit, score_img: np.ndarray, mode: str):
        fg_total = split.train_idx_fg.size + split.test_idx_fg.size
        bg_total = split.train_idx_bg.size + split.test_idx_bg.size
        fg_ratio = fg_total / float(num_pixels) if num_pixels > 0 else 0.0
        bg_ratio = bg_total / float(num_pixels) if num_pixels > 0 else 0.0
        test_total = float(split.test_idx_all.size or 1)
        test_fg_ratio = split.test_idx_fg.size / test_total if test_total > 0 else 0.0
        test_bg_ratio = split.test_idx_bg.size / test_total if test_total > 0 else 0.0
        top10_count = split.test_idx_top10.size if split.test_idx_top10 is not None else 0
        print(
            f"🔀 Pixel split: train={split.train_idx_all.size} | test={split.test_idx_all.size} "
            f"| fg={fg_total} ({fg_ratio:.3f}) | bg={bg_total} ({bg_ratio:.3f}) "
            f"| test_fg={split.test_idx_fg.size} ({test_fg_ratio:.3f}) | test_bg={split.test_idx_bg.size} ({test_bg_ratio:.3f}) "
            f"| test_top10={top10_count} | mode={mode} | tile={ray_split_tile} | thr={split.thr_used:.3e} | seed={ray_split_seed}",
            flush=True,
        )
        score_flat = score_img.reshape(-1)
        fg_all = np.concatenate([split.train_idx_fg, split.test_idx_fg]) if fg_total > 0 else np.array([], dtype=np.int64)
        if fg_all.size > 0:
            top_k = min(5, fg_all.size)
            top_order = np.argpartition(-score_flat[fg_all], top_k - 1)[:top_k]
            top_fg = fg_all[top_order]
            dbg_entries = []
            for idx in top_fg:
                y = int(idx // W)
                x = int(idx % W)
                dbg_entries.append(f"({x},{y},{score_flat[idx]:.3e})")
            print(f"   [pixel-split-debug] FG top-{top_k} (x,y,score): " + ", ".join(dbg_entries), flush=True)
        else:
            print("   [pixel-split-debug] FG top-k: none (no FG pixels).", flush=True)

    if ray_split_enabled and dataset is None:
        print("[smoke-test] dataset missing; disabling ray split.", flush=True)
        ray_split_enabled = False

    if ray_split_enabled:
        ref_sample = dataset[0]
        ap_target_np = ref_sample["ap"].squeeze(0).numpy()
        pa_target_np = ref_sample["pa"].squeeze(0).numpy()
        if ap_target_np.shape != (H, W) or pa_target_np.shape != (H, W):
            raise ValueError(f"Unexpected target shape: AP {ap_target_np.shape}, PA {pa_target_np.shape}, expected {(H, W)}")

        if ray_split_mode == "stratified_intensity":
            fg_thr_value = ray_fg_thr_value if not ray_fg_force_quantile else 0.0
            pixel_split_np = make_pixel_split_stratified_intensity(
                ap_target_np,
                pa_target_np,
                train_frac=ray_split_ratio,
                fg_threshold=fg_thr_value,
                fg_quantile=ray_fg_quantile,
                seed=ray_split_seed,
                pa_xflip=pa_xflip,
                topk_frac=0.10,
            )
        else:
            fg_thr_value = ray_fg_thr_value
            if ray_fg_force_quantile:
                fg_thr_value = -abs(ray_fg_quantile)
            pixel_split_np = make_pixel_split_from_ap_pa(
                ap_target_np,
                pa_target_np,
                train_frac=ray_split_ratio,
                tile=ray_split_tile,
                thr=fg_thr_value,
                seed=ray_split_seed,
                pa_xflip=pa_xflip,
                topk_frac=0.10,
            )
        score_img = np.maximum(ap_target_np, pa_target_np[:, ::-1] if pa_xflip else pa_target_np)
        _log_split(pixel_split_np, score_img, ray_split_mode)

        np.savez(
            outdir / "pixel_split.npz",
            train_idx_all=pixel_split_np.train_idx_all,
            test_idx_all=pixel_split_np.test_idx_all,
            train_idx_fg=pixel_split_np.train_idx_fg,
            train_idx_bg=pixel_split_np.train_idx_bg,
            test_idx_fg=pixel_split_np.test_idx_fg,
            test_idx_bg=pixel_split_np.test_idx_bg,
            test_idx_top10=pixel_split_np.test_idx_top10 if pixel_split_np.test_idx_top10 is not None else np.array([], dtype=np.int64),
            meta=np.array(
                [
                    {
                        "H": H,
                        "W": W,
                        "train_frac": ray_split_ratio,
                        "tile": ray_split_tile,
                        "seed": ray_split_seed,
                        "threshold": pixel_split_np.thr_used,
                        "mode": ray_split_mode,
                        "pa_xflip": pa_xflip,
                    }
                ],
                dtype=object,
            ),
        )

        ray_indices["pixel"] = {
            "train_idx_all": _to_torch(pixel_split_np.train_idx_all),
            "test_idx_all": _to_torch(pixel_split_np.test_idx_all),
            "train_idx_fg": _to_torch(pixel_split_np.train_idx_fg),
            "train_idx_bg": _to_torch(pixel_split_np.train_idx_bg),
            "test_idx_fg": _to_torch(pixel_split_np.test_idx_fg),
            "test_idx_bg": _to_torch(pixel_split_np.test_idx_bg),
            "test_idx_top10": _to_torch(pixel_split_np.test_idx_top10) if pixel_split_np.test_idx_top10 is not None else None,
        }
    else:
        split_uniform = build_ray_split(num_pixels, ray_split_ratio, device)
        ray_indices["pixel"] = {
            "train_idx_all": split_uniform["train"],
            "test_idx_all": split_uniform["test"],
            "train_idx_fg": None,
            "train_idx_bg": None,
            "test_idx_fg": None,
            "test_idx_bg": None,
            "test_idx_top10": None,
        }
        print(
            f"🔀 Legacy Pixel-Split: train={ray_indices['pixel']['train_idx_all'].numel()} / "
            f"test={ray_indices['pixel']['test_idx_all'].numel()} (ratio={ray_split_ratio})",
            flush=True,
        )

    rng_train = np.random.default_rng(ray_split_seed + 12345) if ray_split_enabled else None

    rays_per_proj = args.rays_per_step or config["training"]["chunk"]
    if rays_per_proj <= 0:
        raise ValueError("rays-per-step must be > 0.")
    rays_per_proj = min(rays_per_proj, num_pixels)

    opt_params = list(generator.parameters()) + [z_train]
    if hybrid_enabled and encoder is not None:
        opt_params += list(encoder.parameters())
    if hybrid_enabled and z_fuser is not None:
        opt_params += list(z_fuser.parameters())
    if hybrid_enabled and gain_head is not None:
        opt_params += list(gain_head.parameters())
    if hybrid_enabled and gain_param is not None:
        opt_params += [gain_param]
    optimizer = torch.optim.Adam(
        opt_params,
        lr=config["training"]["lr_g"],
    )
    # Projection-Loss
    proj_loss_type = args.proj_loss_type
    if hybrid_enabled:
        if args.proj_target_source == "counts" and proj_loss_type != "poisson":
            print("[WARN] counts target -> set proj_loss_type=poisson.", flush=True)
            proj_loss_type = "poisson"
        if args.proj_target_source == "norm" and proj_loss_type == "poisson":
            print("[WARN] norm target -> set proj_loss_type=sqrt_mse.", flush=True)
            proj_loss_type = "sqrt_mse"
    if proj_loss_type == "poisson":
        loss_fn = poisson_nll
    elif proj_loss_type == "huber":
        loss_fn = huber_loss
    else:
        loss_fn = sqrt_mse_loss

    amp_enabled = bool(config["training"].get("use_amp", False))
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    if args.smoke_test:
        batch = None
        if dataloader is not None:
            try:
                batch = next(iter(dataloader))
            except Exception as exc:
                print(f"[smoke-test] dataloader failed ({exc.__class__.__name__}); using synthetic batch.", flush=True)
        if batch is None:
            batch = build_synthetic_batch(generator.H, generator.W, device=device)
            ap = batch["ap"]
            pa = batch["pa"]
            meta = batch.get("meta")
            act_vol = batch.get("act")
            ct_vol = batch.get("ct")
        else:
            ap = batch["ap"].to(device, non_blocking=True).float()
            pa = batch["pa"].to(device, non_blocking=True).float()
            meta = batch.get("meta")
            act_vol = batch.get("act")
            if act_vol is not None and act_vol.numel() > 0:
                act_vol = act_vol.to(device, non_blocking=True)
            ct_vol = batch.get("ct")
            if ct_vol is not None and ct_vol.numel() > 0:
                ct_vol = ct_vol.to(device, non_blocking=True).float()
        if act_vol is not None and act_vol.numel() == 0:
            act_vol = None
        if ct_vol is not None and ct_vol.numel() == 0:
            ct_vol = None
        ct_context = generator.build_ct_context(ct_vol, padding_mode=args.ct_padding_mode) if ct_vol is not None else None

        z_base = z_train
        if z_base.shape[0] != ap.shape[0]:
            z_base = z_base.expand(ap.shape[0], -1)
        z_enc = None
        if hybrid_enabled and encoder is not None:
            proj_scale_enc = compute_proj_scale(ap, pa, args.proj_scale_source, meta)
            proj_scale_enc = torch.clamp(proj_scale_enc, min=1e-6)
            enc_input = build_encoder_input(
                ap,
                pa,
                ct_vol,
                proj_scale_enc,
                args.encoder_proj_transform,
                args.encoder_use_ct,
            )
            z_enc = encoder(enc_input)
            if z_enc.shape[0] != z_base.shape[0]:
                z_base = z_base.expand(z_enc.shape[0], -1)
            z_enc_proj = z_fuser(z_enc) if z_fuser is not None else z_enc
            z_latent = z_base + (z_enc_alpha * z_enc_proj)
        else:
            z_latent = z_base

        scale_joint_used = 1.0
        if isinstance(meta, dict):
            meta_scale = meta.get("proj_scale_joint_p99")
            if torch.is_tensor(meta_scale):
                meta_scale = meta_scale.item() if meta_scale.numel() > 0 else None
            if isinstance(meta_scale, (int, float)) and math.isfinite(meta_scale) and meta_scale > 0:
                scale_joint_used = float(meta_scale)

        idx_ap = torch.randperm(num_pixels, device=device)[:rays_per_proj]
        idx_pa = map_pa_indices_torch(idx_ap, W, pa_xflip)
        ray_batch_ap = slice_rays(rays_cache["ap"], idx_ap)
        ray_batch_pa = slice_rays(rays_cache["pa"], idx_pa)

        proj_weight = 1.0
        if hybrid_enabled:
            proj_weight = float(args.proj_loss_weight)

        with torch.cuda.amp.autocast(enabled=amp_enabled):
            pred_ap_raw, _ = render_minibatch(generator, z_latent, ray_batch_ap, ct_context=ct_context)
            pred_pa_raw, _ = render_minibatch(generator, z_latent, ray_batch_pa, ct_context=ct_context)
            ap_counts = batch.get("ap_counts")
            pa_counts = batch.get("pa_counts")
            use_counts = ap_counts is not None and ap_counts.numel() > 0 and pa_counts is not None and pa_counts.numel() > 0
            if use_counts:
                ap_counts = ap_counts.to(device, non_blocking=True).float()
                pa_counts = pa_counts.to(device, non_blocking=True).float()
                target_ap = ap_counts.reshape(ap_counts.shape[0], -1)[0, idx_ap].unsqueeze(0)
                target_pa = pa_counts.reshape(pa_counts.shape[0], -1)[0, idx_pa].unsqueeze(0)
            else:
                target_ap = ap.reshape(ap.shape[0], -1)[0, idx_ap].unsqueeze(0)
                target_pa = pa.reshape(pa.shape[0], -1)[0, idx_pa].unsqueeze(0)

            if proj_loss_type == "poisson":
                pred_ap = F.softplus(pred_ap_raw) + 1e-6
                pred_pa = F.softplus(pred_pa_raw) + 1e-6
            else:
                pred_ap = pred_ap_raw
                pred_pa = pred_pa_raw
            if use_counts:
                pred_ap = pred_ap * float(scale_joint_used)
                pred_pa = pred_pa * float(scale_joint_used)
                if gain_head is not None and z_enc is not None:
                    gain_val = F.softplus(gain_head(z_enc))
                    pred_ap = pred_ap * gain_val
                    pred_pa = pred_pa * gain_val
                elif gain_param is not None:
                    gain_val = F.softplus(gain_param)
                    pred_ap = pred_ap * gain_val
                    pred_pa = pred_pa * gain_val
            loss_ap = loss_fn(pred_ap, target_ap)
            loss_pa = loss_fn(pred_pa, target_pa)
            loss_proj = 0.5 * (loss_ap + loss_pa)
            if hybrid_enabled:
                loss = proj_weight * loss_proj
            else:
                loss = loss_proj
            loss_act = torch.tensor(0.0, device=device)
            if args.act_loss_weight > 0.0 and act_vol is not None:
                radius = generator.radius
                if isinstance(radius, tuple):
                    radius = radius[1]
                coords, act_samples, pos_flags = sample_act_points(
                    act_vol,
                    args.act_samples,
                    radius=radius,
                    pos_fraction=args.act_pos_fraction,
                    pos_threshold=args.act_pos_threshold,
                )
                pred_act, pred_act_raw = query_emission_at_points(generator, z_latent, coords, return_raw=True)
                if pred_act.numel() > 0:
                    act_norm_factor, _ = compute_act_norm_factor(
                        act_vol, args.act_norm_source, args.act_norm_value, None
                    )
                    act_norm_factor = float(act_norm_factor)
                    pred_pos = pred_act_raw.clamp_min(0.0) / max(act_norm_factor, 1e-8)
                    act_pos = act_samples.clamp_min(0.0) / max(act_norm_factor, 1e-8)
                    pred_log = torch.log1p(pred_pos)
                    act_log = torch.log1p(act_pos)
                    weights_act = torch.where(
                        pos_flags,
                        torch.full_like(pred_log, args.act_pos_weight),
                        torch.ones_like(pred_log),
                    )
                    diff = F.smooth_l1_loss(pred_log, act_log, reduction="none")
                    loss_act = torch.mean(weights_act * diff)
                    loss = loss + args.act_loss_weight * loss_act

        scaler.scale(loss).backward()
        print(
            f"[smoke-test] loss={loss.item():.6f} | proj={loss_proj.item():.6f} | act={loss_act.item():.6f} "
            f"| pred_ap shape={tuple(pred_ap.shape)} pred_pa shape={tuple(pred_pa.shape)} "
            f"| finite_pred={torch.isfinite(pred_ap).all().item() and torch.isfinite(pred_pa).all().item()}",
            flush=True,
        )
        return

    data_iter = iter(dataloader)
    ct_context = None
    max_steps_cfg = None
    if isinstance(training_cfg, dict):
        max_steps_cfg = training_cfg.get("max_steps")
    if max_steps_cfg is None:
        max_steps_cfg = config.get("max_steps")
    if max_steps_cfg is not None and args.max_steps is not None and args.max_steps > 0:
        if int(max_steps_cfg) != int(args.max_steps):
            print(
                f"[cfg] max_steps in config={int(max_steps_cfg)} -> CLI override (using {int(args.max_steps)})",
                flush=True,
            )
    max_steps = int(args.max_steps)
    last_step = 0
    exit_reason = "normal"

    def _signal_handler(signum, frame):
        nonlocal exit_reason, last_step
        exit_reason = "signal"
        try:
            signame = signal.Signals(signum).name
        except Exception:
            signame = str(signum)
        print(
            f"[signal] received {signame} last_step={last_step} max_steps={max_steps}",
            flush=True,
        )
        raise SystemExit(128 + int(signum))

    signal.signal(signal.SIGTERM, _signal_handler)
    signal.signal(signal.SIGINT, _signal_handler)
    print(
        f"🚀 Starting emission-NeRF training | steps={max_steps} | rays/proj={rays_per_proj} "
        f"| image={generator.H}x{generator.W} | chunk={generator.chunk}"
    )
    scale_ap_used = 1.0
    scale_pa_used = 1.0
    scale_joint_used = 1.0
    scale_missing_warned = False
    counts_missing_warned = False
    act_norm_global = None
    last_z_latent = z_train
    last_gain_val = None
    gain_prior_ema = None
    gain_prior_final = None
    gain_prior_decay = 0.9
    gain_prior_steps = 50
    proj_collapse_count = 0

    print(
        f"[sanity] max_steps={max_steps} proj_warmup_steps={args.proj_warmup_steps} "
        f"depth_sanity_every={args.depth_sanity_every}",
        flush=True,
    )
    try:
        for step in range(1, max_steps + 1):
            last_step = step
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                batch = next(data_iter)
    
            ap = batch["ap"].to(device, non_blocking=True).float()
            pa = batch["pa"].to(device, non_blocking=True).float()
            meta = batch.get("meta")
            meta_scale = None
            meta_missing = False
            if isinstance(meta, dict):
                meta_scale = meta.get("proj_scale_joint_p99")
                meta_missing = meta.get("proj_scale_joint_p99_missing", False)
                if isinstance(meta_scale, (list, tuple)):
                    meta_scale = meta_scale[0] if meta_scale else None
                if torch.is_tensor(meta_scale):
                    meta_scale = meta_scale.item() if meta_scale.numel() > 0 else None
                if torch.is_tensor(meta_missing):
                    meta_missing = bool(meta_missing.item()) if meta_missing.numel() > 0 else False
            if meta_scale is None or (isinstance(meta_scale, float) and math.isnan(meta_scale)) or meta_missing:
                scale_joint_used = 1.0
                if not scale_missing_warned:
                    print(
                        "[WARN] proj_scale_joint_p99 fehlt im manifest; physikalische Metriken sind bedeutungslos.",
                        flush=True,
                    )
                    scale_missing_warned = True
            else:
                scale_joint_used = float(meta_scale)
            if step == 1:
                print(
                    f"[scale] projections_on_disk_normalized_with_joint_p99: proj_scale_joint_p99={scale_joint_used:.3e}",
                    flush=True,
                )
            act_vol = batch.get("act")
            if act_vol is not None:
                if act_vol.numel() == 0:
                    act_vol = None
                else:
                    act_vol = act_vol.to(device, non_blocking=True)
            ct_vol = batch.get("ct")
            if ct_vol is not None:
                ct_vol = ct_vol.to(device, non_blocking=True).float()
            ct_context = generator.build_ct_context(ct_vol, padding_mode=args.ct_padding_mode) if ct_vol is not None else None
    
            # Wichtig: Flatten-Order ist (y * W + x), identisch zu den Ray-Indizes aus make_stratified_tile_split.
            # Keine permute/transpose zwischen (H, W) und reshape(-1), damit Target/Predict exakt die gleiche Reihenfolge teilen.
            ap_flat = ap.reshape(batch_size, -1)
            pa_flat = pa.reshape(batch_size, -1)
            ap_counts = batch.get("ap_counts")
            pa_counts = batch.get("pa_counts")
            if ap_counts is not None and ap_counts.numel() > 0:
                ap_counts = ap_counts.to(device, non_blocking=True).float()
            else:
                ap_counts = None
            if pa_counts is not None and pa_counts.numel() > 0:
                pa_counts = pa_counts.to(device, non_blocking=True).float()
            else:
                pa_counts = None
    
            use_counts = (
                (ap_counts is not None)
                and (pa_counts is not None)
                and (ap_counts.numel() > 0)
                and (pa_counts.numel() > 0)
            )
            if not use_counts and not counts_missing_warned:
                print("[WARN] Projection-Loss wuerde normierte AP/PA nutzen (Counts fehlen).", flush=True)
                counts_missing_warned = True
            if proj_loss_type == "poisson" and not use_counts:
                raise RuntimeError("Poisson projection loss requires count targets (ap_counts/pa_counts).")
            if use_counts and proj_loss_type != "poisson":
                raise RuntimeError("Counts targets require Poisson projection loss.")
    
            target_ap_full = ap_counts if use_counts else ap
            target_pa_full = pa_counts if use_counts else pa
            ap_flat_proc = target_ap_full.reshape(batch_size, -1)
            pa_flat_proc = target_pa_full.reshape(batch_size, -1)
    
            pred_to_counts_scale = scale_joint_used if use_counts else 1.0
            if use_counts:
                scale_ap_used = 1.0
                scale_pa_used = 1.0
            else:
                scale_ap_used = scale_joint_used
                scale_pa_used = scale_joint_used
    
            z_base = z_train
            if z_base.shape[0] != ap.shape[0]:
                z_base = z_base.expand(ap.shape[0], -1)
            z_enc = None
            z_enc_proj = None
            proj_scale_enc = None
            if hybrid_enabled and encoder is not None:
                proj_scale_enc = compute_proj_scale(ap, pa, args.proj_scale_source, meta)
                proj_scale_enc = torch.clamp(proj_scale_enc, min=1e-6)
                enc_input = build_encoder_input(
                    ap,
                    pa,
                    ct_vol,
                    proj_scale_enc,
                    args.encoder_proj_transform,
                    args.encoder_use_ct,
                )
                z_enc = encoder(enc_input)
                if z_enc.shape[0] != z_base.shape[0]:
                    z_base = z_base.expand(z_enc.shape[0], -1)
                if z_fuser is not None:
                    z_enc_proj = z_fuser(z_enc)
                else:
                    z_enc_proj = z_enc
                z_latent = z_base + (z_enc_alpha * z_enc_proj)
            else:
                z_latent = z_base
            last_z_latent = z_latent
    
            skip_proj = bool(args.act_only)
            debug_act_step = bool(args.debug_act and step == 1)
            proj_metrics_enabled = (not skip_proj) and not (hybrid_enabled and args.proj_loss_weight <= 0.0)
    
            optimizer.zero_grad(set_to_none=True)
            t0 = time.perf_counter()
    
            need_ray_tv = (not skip_proj) and ray_tv_weight != 0.0
            need_bg_depth = (not skip_proj) and args.bg_depth_mass_weight > 0.0
            need_raw_stats = proj_metrics_enabled and hybrid_enabled and args.log_every > 0 and (step % args.log_every == 0 or step == 1)
            need_raw = need_ray_tv or need_bg_depth or need_raw_stats or depth_checks_active
    
            idx_ap = None
            idx_pa = None
            ray_batch_ap = None
            ray_batch_pa = None
            proj_weight = 0.0
            if not skip_proj:
                if ray_split_enabled and pixel_split_np is not None and rng_train is not None:
                    idx_np = sample_train_indices(pixel_split_np, rays_per_proj, ray_train_fg_frac, rng_train)
                    idx_ap = torch.from_numpy(idx_np).long().to(device, non_blocking=True)
                    idx_pa = map_pa_indices_torch(idx_ap, W, pa_xflip)
                else:
                    idx_ap = sample_split_indices(ray_indices["pixel"]["train_idx_all"], rays_per_proj)
                    idx_pa = map_pa_indices_torch(idx_ap, W, pa_xflip)
    
                ray_batch_ap = slice_rays(rays_cache["ap"], idx_ap)
                ray_batch_pa = slice_rays(rays_cache["pa"], idx_pa)
    
                proj_weight = 1.0
                if hybrid_enabled:
                    proj_weight_min = float(args.proj_weight_min)
                    proj_weight_max = float(args.proj_loss_weight)
                    if args.proj_warmup_steps > 0 and step <= args.proj_warmup_steps:
                        proj_weight = proj_weight_min
                    else:
                        ramp_steps = max(1, int(args.proj_ramp_steps))
                        ramp_t = min(1.0, max(0.0, (step - max(args.proj_warmup_steps, 0)) / float(ramp_steps)))
                        proj_weight = proj_weight_min + ramp_t * (proj_weight_max - proj_weight_min)
    
            loss = torch.tensor(0.0, device=device)
            loss_ap = torch.tensor(0.0, device=device)
            loss_pa = torch.tensor(0.0, device=device)
            loss_proj = torch.tensor(0.0, device=device)
            pred_ap = None
            pred_pa = None
            pred_ap_raw = None
            pred_pa_raw = None
            target_ap = None
            target_pa = None
            extras_ap = None
            extras_pa = None
            gain_val = None
    
            with torch.cuda.amp.autocast(enabled=amp_enabled):
                if not skip_proj:
                    pred_ap, extras_ap = render_minibatch(
                        generator, z_latent, ray_batch_ap, ct_context=ct_context, return_raw=need_raw
                    )
                    pred_pa, extras_pa = render_minibatch(
                        generator, z_latent, ray_batch_pa, ct_context=ct_context, return_raw=need_raw
                    )
    
                    target_ap = ap_flat_proc[0, idx_ap].unsqueeze(0)
                    target_pa = pa_flat_proc[0, idx_pa].unsqueeze(0)
    
                    pred_ap_raw = pred_ap
                    pred_pa_raw = pred_pa
    
                    if proj_loss_type == "poisson":
                        lambda_ap_used = F.softplus(pred_ap_raw) - math.log(2.0)
                        lambda_pa_used = F.softplus(pred_pa_raw) - math.log(2.0)
                        lambda_ap_used = lambda_ap_used.clamp_min(1e-6)
                        lambda_pa_used = lambda_pa_used.clamp_min(1e-6)
                    else:
                        lambda_ap_used = pred_ap_raw
                        lambda_pa_used = pred_pa_raw
    
                    gain_val_raw = None
                    gain_val = None
                    if use_counts:
                        lambda_ap_used = lambda_ap_used * float(pred_to_counts_scale)
                        lambda_pa_used = lambda_pa_used * float(pred_to_counts_scale)
                        if gain_head is not None and z_enc is not None:
                            gain_raw = gain_head(z_enc)
                            gain_val_raw = F.softplus(gain_raw)
                        elif gain_param is not None:
                            gain_val_raw = F.softplus(gain_param)
                        if gain_val_raw is not None:
                            g_min = float(args.gain_clamp_min) if args.gain_clamp_min is not None else None
                            g_max = args.gain_clamp_max
                            if g_min is not None or g_max is not None:
                                gmin = g_min if g_min is not None else -float("inf")
                                gmax = g_max if g_max is not None else float("inf")
                                gain_val = torch.clamp(gain_val_raw, min=gmin, max=gmax)
                            else:
                                gain_val = gain_val_raw
                            lambda_ap_used = lambda_ap_used * gain_val
                            lambda_pa_used = lambda_pa_used * gain_val
                    pred_ap = lambda_ap_used
                    pred_pa = lambda_pa_used
                    last_gain_val = gain_val
    
                    if (target_ap < 0).any() or (target_pa < 0).any():
                        print("[WARN] Negative projection targets detected.", flush=True)
                    if proj_loss_type == "poisson":
                        if not torch.isfinite(pred_ap).all() or not torch.isfinite(pred_pa).all():
                            raise RuntimeError("Non-finite lambda in Poisson projection loss.")
                        if (pred_ap <= 0).any() or (pred_pa <= 0).any():
                            raise RuntimeError("Non-positive lambda in Poisson projection loss.")
                        if use_counts:
                            if (target_ap <= 1.0).all() and (target_pa <= 1.0).all():
                                print("[WARN] Count targets appear in [0,1] range; check scaling.", flush=True)
                            if step == 1:
                                p95_ap = float(torch.quantile(pred_ap, 0.95).item()) if pred_ap.numel() > 0 else float("nan")
                                p95_pa = float(torch.quantile(pred_pa, 0.95).item()) if pred_pa.numel() > 0 else float("nan")
                                if p95_ap < 5 or p95_ap > 1e6 or p95_pa < 5 or p95_pa > 1e6:
                                    print("[WARN] Lambda p95 outside expected count scale (5..1e6).", flush=True)
                        if step % 50 == 0:
                            pred_std = float(pred_ap.std().item())
                            target_std = float(target_ap.std().item())
                            if pred_std < 1e-6 and target_std > 1e-3:
                                proj_collapse_count += 1
                                pred_mean = float(pred_ap.mean().item())
                                pred_min = float(pred_ap.min().item())
                                pred_max = float(pred_ap.max().item())
                                target_mean = float(target_ap.mean().item())
                                gain_pre = float(gain_val_raw.mean().item()) if gain_val_raw is not None else float("nan")
                                gain_post = float(gain_val.mean().item()) if gain_val is not None else float("nan")
                                print(
                                    "[WARN][proj] Projection lambda collapsed: pred std ~0 while target std is large.",
                                    flush=True,
                                )
                                print(
                                    f"[WARN][proj] pred_std={pred_std:.3e} target_std={target_std:.3e} "
                                    f"| pred_mean={pred_mean:.3e} pred_min/max=({pred_min:.3e},{pred_max:.3e}) "
                                    f"| target_mean={target_mean:.3e} "
                                    f"| pred_to_counts_scale={pred_to_counts_scale:.3e} "
                                    f"| gain_pre={gain_pre:.3e} gain_post={gain_post:.3e} "
                                    f"| collapse_streak={proj_collapse_count} patience={args.proj_collapse_patience}",
                                    flush=True,
                                )
                                if args.proj_collapse_patience > 0 and proj_collapse_count >= args.proj_collapse_patience:
                                    raise RuntimeError(
                                        "Projection lambda collapsed: pred std ~0 while target std is large "
                                        "(patience exceeded)."
                                    )
                            else:
                                proj_collapse_count = 0
    
                    weight_ap = build_loss_weights(target_ap, args.bg_weight, args.weight_threshold)
                    weight_pa = build_loss_weights(target_pa, args.bg_weight, args.weight_threshold)
    
                    if step in (1, 50):
                        pred_raw_mean = float(pred_ap_raw.mean().item())
                        pred_raw_std = float(pred_ap_raw.std().item())
                        pred_mean = float(pred_ap.mean().item())
                        pred_std = float(pred_ap.std().item())
                        target_mean = float(target_ap.mean().item())
                        target_std = float(target_ap.std().item())
                        gain_pre = float(gain_val_raw.mean().item()) if gain_val_raw is not None else float("nan")
                        gain_post = float(gain_val.mean().item()) if gain_val is not None else float("nan")
                        print(
                            f"[DEBUG][proj][step {step}] pred_raw_mean={pred_raw_mean:.3e} pred_raw_std={pred_raw_std:.3e} "
                            f"| pred_mean={pred_mean:.3e} pred_std={pred_std:.3e} "
                            f"| target_mean={target_mean:.3e} target_std={target_std:.3e} "
                            f"| sum_pred={pred_ap.sum().item():.3e} sum_target={target_ap.sum().item():.3e} "
                            f"| pred_to_counts_scale={pred_to_counts_scale:.3e} "
                            f"| gain_pre={gain_pre:.3e} gain_post={gain_post:.3e}",
                            flush=True,
                        )
    
                    loss_ap = loss_fn(pred_ap, target_ap, weight=weight_ap)
                    loss_pa = loss_fn(pred_pa, target_pa, weight=weight_pa)
                    loss_proj = 0.5 * (loss_ap + loss_pa)
                    if step == 1:
                        tmin = float(target_ap.min().item()) if target_ap.numel() > 0 else float("nan")
                        tmax = float(target_ap.max().item()) if target_ap.numel() > 0 else float("nan")
                        lmin = float(pred_ap.min().item()) if pred_ap.numel() > 0 else float("nan")
                        lmax = float(pred_ap.max().item()) if pred_ap.numel() > 0 else float("nan")
                        print(
                            f"[DEBUG][proj][step 1] use_counts={use_counts} "
                            f"| target_min/max=({tmin:.3e},{tmax:.3e}) "
                            f"| lambda_min/max=({lmin:.3e},{lmax:.3e})",
                            flush=True,
                        )
                    if hybrid_enabled:
                        if proj_weight > 0.0:
                            loss = loss + proj_weight * loss_proj
                    else:
                        loss = loss_proj
                    if DEBUG_PRINTS and (step % 50 == 0):
                        print(
                            f"[DEBUG][step {step}] TARGET AP min/max: {target_ap.min().item():.3e}/{target_ap.max().item():.3e} | "
                            f"PRED AP min/max: {pred_ap.min().item():.3e}/{pred_ap.max().item():.3e} | "
                            f"TARGET PA min/max: {target_pa.min().item():.3e}/{target_pa.max().item():.3e} | "
                            f"PRED PA min/max: {pred_pa.min().item():.3e}/{pred_pa.max().item():.3e}",
                            flush=True,
                        )
    
                bg_depth_mass = torch.tensor(0.0, device=device)
                bg_depth_mass_w = torch.tensor(0.0, device=device)
                bg_depth_frac_t = torch.tensor(0.0, device=device)
                if (not skip_proj) and args.bg_depth_mass_weight > 0.0:
                    bg_mask = None
                    if target_ap is not None and target_pa is not None:
                        bg_mask = (target_ap < args.bg_depth_eps) & (target_pa < args.bg_depth_eps)
                    elif target_ap is not None:
                        bg_mask = target_ap < args.bg_depth_eps
                    elif target_pa is not None:
                        bg_mask = target_pa < args.bg_depth_eps
                    if bg_mask is not None:
                        bg_mask_flat = bg_mask.reshape(-1)
                        bg_depth_frac_t = bg_mask_flat.float().mean()
                        bg_terms = []
                        for extras in (extras_ap, extras_pa):
                            if not isinstance(extras, dict):
                                continue
                            raw_out = extras.get("raw")
                            if raw_out is None:
                                continue
                            lambda_vals = F.softplus(raw_out[..., 0])
                            if args.bg_depth_mode == "integral":
                                dists = extras.get("dists")
                                if dists is None:
                                    z_vals = extras.get("z_vals")
                                    if z_vals is not None:
                                        dists = z_vals[..., 1:] - z_vals[..., :-1]
                                        dists = torch.cat([dists, dists[..., -1:].clone()], dim=-1)
                                if dists is None:
                                    continue
                                m_ray = torch.sum(lambda_vals * dists, dim=-1)
                            else:
                                m_ray = torch.mean(lambda_vals, dim=-1)
                            if m_ray.shape[0] != bg_mask_flat.shape[0]:
                                continue
                            if bg_mask_flat.any():
                                bg_terms.append(m_ray[bg_mask_flat].mean())
                        if bg_terms:
                            bg_depth_mass = torch.stack(bg_terms).mean()
                            bg_depth_mass_w = bg_depth_mass * args.bg_depth_mass_weight
                            loss = loss + bg_depth_mass_w
    
                loss_act = torch.tensor(0.0, device=device)
                act_norm_factor = 1.0
                if args.act_loss_weight > 0.0 and act_vol is not None:
                    radius = generator.radius
                    if isinstance(radius, tuple):
                        radius = radius[1]
                    # Stichprobe aus act.npy und direkte Dichteabfrage im NeRF
                    coords, act_samples, pos_flags = sample_act_points(
                        act_vol,
                        args.act_samples,
                        radius=radius,
                        pos_fraction=args.act_pos_fraction,
                        pos_threshold=args.act_pos_threshold,
                    )
                    pred_act_raw = None
                    pred_act = None
                    pred_act_log = None
                    if debug_act_step:
                        pred_act, pred_act_raw = query_emission_at_points(
                            generator, z_latent, coords, return_raw=True
                        )
                    else:
                        pred_act, pred_act_raw = query_emission_at_points(
                            generator, z_latent, coords, return_raw=True
                        )
                    if pred_act.numel() > 0:
                        act_norm_factor, act_norm_global = compute_act_norm_factor(
                            act_vol, args.act_norm_source, args.act_norm_value, act_norm_global
                        )
                        act_norm_factor = float(act_norm_factor)
                        pred_pos = pred_act_raw.clamp_min(0.0) / max(act_norm_factor, 1e-8)
                        act_pos = act_samples.clamp_min(0.0) / max(act_norm_factor, 1e-8)
                        pred_act_log = torch.log1p(pred_pos)
                        act_log = torch.log1p(act_pos)
                        weights_act = torch.where(
                            pos_flags,
                            torch.full_like(pred_act_log, args.act_pos_weight),
                            torch.ones_like(pred_act_log),
                        )
                        diff = F.smooth_l1_loss(pred_act_log, act_log, reduction="none")
                        loss_act = torch.mean(weights_act * diff)
                        loss = loss + args.act_loss_weight * loss_act
                        if debug_act_step:
                            act_vol_stats = tensor_stats(act_vol)
                            pos_vol_frac = float((act_vol > 1e-8).float().mean().item()) if act_vol is not None else float("nan")
                            act_stats_pre = tensor_stats(act_samples)
                            act_stats_norm = tensor_stats(act_pos)
                            pred_stats = tensor_stats(pred_act)
                            pred_raw_stats = tensor_stats(pred_act_raw)
                            zero_frac = float((act_samples == 0).float().mean().item())
                            tiny_frac = float((act_samples < 1e-6).float().mean().item())
                            pos_frac = float(pos_flags.float().mean().item()) if pos_flags.numel() > 0 else float("nan")
                            print(
                                f"[DEBUG][ACT][step {step}] act_vol={fmt_stats(act_vol_stats)} | pos_vol_frac={pos_vol_frac:.3f} "
                                f"| act_gt={fmt_stats(act_stats_pre)} "
                                f"| act_gt_norm={fmt_stats(act_stats_norm)} "
                                f"| zero_frac={zero_frac:.3f} | lt1e-6_frac={tiny_frac:.3f} | pos_frac={pos_frac:.3f}",
                                flush=True,
                            )
                            print(
                                f"[DEBUG][ACT][step {step}] pred_raw={fmt_stats(pred_raw_stats)} "
                                f"| pred_act={fmt_stats(pred_stats)} "
                                f"| act_norm_source={args.act_norm_source} act_norm_factor={act_norm_factor:.3e} "
                                f"act_norm_global={'set' if act_norm_global is not None else 'none'}",
                                flush=True,
                            )
                            print(
                                f"[DEBUG][ACT][step {step}] requires_grad: pred_act={pred_act_raw.requires_grad} "
                                f"z_latent={z_latent.requires_grad} act_samples={act_samples.requires_grad}",
                                flush=True,
                            )
    
                loss_gain = torch.tensor(0.0, device=device)
                gain_prior = None
                if hybrid_enabled and gain_val is not None and args.gain_reg_weight > 0.0:
                    gain_mean = float(gain_val.detach().mean().item())
                    if args.gain_prior_mode == "fixed":
                        gain_prior = float(args.gain_prior_value)
                    else:
                        # EMA over first N steps
                        if gain_prior_ema is None:
                            gain_prior_ema = gain_mean
                        else:
                            gain_prior_ema = gain_prior_decay * gain_prior_ema + (1.0 - gain_prior_decay) * gain_mean
                        if step <= gain_prior_steps:
                            gain_prior = gain_prior_ema
                        else:
                            if gain_prior_final is None:
                                gain_prior_final = gain_prior_ema
                            gain_prior = gain_prior_final
                    if gain_prior is not None and gain_prior > 0:
                        log_gain = torch.log(gain_val.clamp_min(1e-12))
                        log_prior = math.log(max(gain_prior, 1e-12))
                        loss_gain = ((log_gain - log_prior) ** 2).mean()
                        loss_gain = loss_gain * float(args.gain_reg_scale)
                        loss = loss + float(args.gain_reg_weight) * loss_gain
    
                loss_ct = torch.tensor(0.0, device=device)
                ct_pairs_valid = False
                if args.ct_loss_weight > 0.0 and ct_vol is not None:
                    radius = generator.radius
                    if isinstance(radius, tuple):
                        radius = radius[1]
                    ct_pairs = sample_ct_pairs(ct_vol, args.ct_samples, args.ct_threshold, radius=radius)
                    if ct_pairs is not None:
                        ct_pairs_valid = True
                        coords1, coords2, weights = ct_pairs
                        pred1 = query_emission_at_points(generator, z_latent, coords1)
                        pred2 = query_emission_at_points(generator, z_latent, coords2)
                        # Loss zwingt Emission auf flachen CT-Strecken zur Konstanz
                        loss_ct = torch.mean(torch.abs(pred1 - pred2) * weights)
                        loss = loss + args.ct_loss_weight * loss_ct
    
                loss_reg = torch.tensor(0.0, device=device)
                if args.z_reg_weight > 0.0:
                    loss_reg = z_train.pow(2).mean()
                    loss = loss + args.z_reg_weight * loss_reg
    
                tv_base_loss = torch.tensor(0.0, device=device)
                loss_tv = torch.tensor(0.0, device=device)
                loss_ray_tv = torch.tensor(0.0, device=device)
                loss_ray_tv_w = torch.tensor(0.0, device=device)
                ray_tv_mode = "plain"
                ray_tv_w_mean = None
                ray_tv_w_min = None
                ray_tv_w_max = None
                ct_boundary_depth_mean = None
                ct_boundary_depth_median = None
    
                tv_base_terms = []
                if isinstance(extras_ap, dict):
                    base_val = extras_ap.get("tv_base_loss") or extras_ap.get("tv_loss")
                    if base_val is not None:
                        tv_base_terms.append(base_val)
                if isinstance(extras_pa, dict):
                    base_val = extras_pa.get("tv_base_loss") or extras_pa.get("tv_loss")
                    if base_val is not None:
                        tv_base_terms.append(base_val)
    
                if tv_base_terms:
                    tv_base_loss = torch.stack(tv_base_terms).mean()
    
                if tv_weight != 0.0:
                    loss_tv = tv_weight * tv_base_loss
                    loss = loss + loss_tv
    
                if ray_tv_weight != 0.0:
                    edge_aware_active = ray_tv_edge_aware and ray_tv_alpha > 0.0
                    ray_tv_terms = []
                    ray_tv_w_terms = []
                    for extras in (extras_ap, extras_pa):
                        if not isinstance(extras, dict):
                            continue
                        raw_out = extras.get("raw")
                        if raw_out is None:
                            continue
                        if edge_aware_active:
                            mu_out = extras.get("mu")
                            tv_val, w_stats = compute_ray_tv(
                                raw_out,
                                mu_vals=mu_out,
                                edge_aware=True,
                                alpha=ray_tv_alpha,
                                w_clamp_min=ray_tv_w_clamp_min,
                                return_stats=True,
                            )
                            if isinstance(w_stats, dict):
                                w_mean = w_stats.get("w_mean")
                                if w_mean is not None:
                                    ray_tv_w_terms.append(w_mean)
                                ray_tv_w_min = w_stats.get("w_min")
                                ray_tv_w_max = w_stats.get("w_max")
                                ct_boundary_depth_mean = w_stats.get("ct_boundary_depth_mean")
                                ct_boundary_depth_median = w_stats.get("ct_boundary_depth_median")
                            ray_tv_terms.append(tv_val)
                        else:
                            tv_val, _ = compute_ray_tv(raw_out)
                            ray_tv_terms.append(tv_val)
                    if ray_tv_terms:
                        loss_ray_tv = torch.stack(ray_tv_terms).mean()
                        loss_ray_tv_w = loss_ray_tv * ray_tv_weight
                        loss = loss + loss_ray_tv_w
                    if edge_aware_active and ray_tv_w_terms:
                        ray_tv_w_mean = torch.stack(ray_tv_w_terms).mean().item()
                        ray_tv_mode = "edgeaware"
    
                if depth_checks_active and (step % depth_sanity_every == 0 or step == 1):
                    # Single-Phantom ist inhaltlich stabil, wenn:
                    # - Projektionen plateauieren,
                    # - Depth-Regularizer nicht trivial sind,
                    # - lambda-Std entlang Rays stabil > 1e-4,
                    # - Gain im physikalischen Bereich bleibt.
                    proj_loss_active = (not skip_proj) and (not hybrid_enabled or proj_weight > 0.0)
                    atten_flag = bool(generator.render_kwargs_train.get("use_attenuation", False))
                    atten_active = atten_flag and (ct_context is not None)
                    if proj_loss_active and not atten_active:
                        reason = "use_attenuation=False" if not atten_flag else "ct_context=None"
                        print(
                            f"[WARN][depth] Attenuation inaktiv bei aktivem Projection-Loss -> Depth unterbestimmt. "
                            f"Ursache: {reason}.",
                            flush=True,
                        )
                    if proj_loss_active and atten_active:
                        mu_terms = []
                        atten_terms = []
                        atten_default = globals().get("ATTEN_SCALE_DEFAULT", 1.0)
                        atten_scale = float(generator.render_kwargs_train.get("atten_scale", atten_default))
                        for extras in (extras_ap, extras_pa):
                            if not isinstance(extras, dict):
                                continue
                            mu_out = extras.get("mu")
                            if mu_out is not None and mu_out.dim() > 0 and mu_out.shape[-1] == 1:
                                mu_out = mu_out.squeeze(-1)
                            dists = extras.get("dists")
                            if mu_out is not None:
                                mu_terms.append(mu_out.reshape(-1))
                            if mu_out is not None and dists is not None and mu_out.shape == dists.shape:
                                mu_clamped = torch.clamp(mu_out, min=0.0)
                                mu_dists = mu_clamped * dists
                                attenuation = torch.cumsum(mu_dists, dim=-1) * atten_scale
                                attenuation = F.pad(attenuation[..., :-1], (1, 0), mode="constant", value=0.0)
                                attenuation = torch.clamp(attenuation, min=0.0, max=60.0)
                                atten_terms.append(attenuation.reshape(-1))
                        if not mu_terms:
                            print(
                                "[WARN][depth] Attenuation aktiv, aber mu fehlt in Extras (retraw/ct_context prüfen).",
                                flush=True,
                            )
                        else:
                            mu_all = torch.cat(mu_terms)
                            mu_mean = float(mu_all.mean().item())
                            if mu_mean <= 0.0:
                                print(
                                    "[WARN][depth] mu_mean <= 0 -> Attenuation hat praktisch keinen Einfluss.",
                                    flush=True,
                                )
                        if atten_terms:
                            atten_all = torch.cat(atten_terms)
                            atten_mean = float(atten_all.mean().item())
                            if atten_mean < 1e-3:
                                print(
                                    f"[WARN][depth] Attenuation-Mittelwert sehr klein ({atten_mean:.3e}) -> geringer Einfluss.",
                                    flush=True,
                                )
    
                    depth_zero = []
                    if ray_tv_weight != 0.0 and float(loss_ray_tv.item()) <= 0.0:
                        depth_zero.append("ray_tv")
                    if args.ct_loss_weight > 0.0:
                        if ct_vol is None:
                            depth_zero.append("ct_missing")
                        elif not ct_pairs_valid:
                            depth_zero.append("ct_pairs")
                    if depth_zero:
                        print(
                            f"[WARN][depth] Depth-Terms ohne Signal: {', '.join(depth_zero)}. "
                            "Bitte CT/Ray-TV/BG-Depth pruefen.",
                            flush=True,
                        )
    
                    lam_std_terms = []
                    for extras in (extras_ap, extras_pa):
                        if not isinstance(extras, dict):
                            continue
                        raw_out = extras.get("raw")
                        if raw_out is None:
                            continue
                        lam = F.softplus(raw_out[..., 0]) - math.log(2.0)
                        lam = lam.clamp_min(1e-6)
                        lam_std_terms.append(lam.std(dim=-1).mean())
                    if lam_std_terms:
                        lam_std_mean = float(torch.stack(lam_std_terms).mean().item())
                        if lam_std_mean < 1e-6:
                            print(
                                f"[WARN][depth] Depth-Profil kollabiert (lambda std entlang Ray ~ {lam_std_mean:.3e}).",
                                flush=True,
                            )
                        if lam_std_mean < 1e-4:
                            print(
                                f"[WARN][depth] lambda std entlang Ray sehr klein: {lam_std_mean:.3e}",
                                flush=True,
                            )
    
                    depth_reg_loss = loss_ray_tv_w + (args.ct_loss_weight * loss_ct) + bg_depth_mass_w
                    depth_reg_total = float(depth_reg_loss.detach().item())
                    net_module = generator.render_kwargs_train.get("network_fn")
                    depth_grad_net = _depth_grad_norm(depth_reg_loss, net_module)
                    print(
                        f"[depth][grad] ||g_depth||_net={depth_grad_net:.3e} | depth_reg_total={depth_reg_total:.3e}",
                        flush=True,
                    )
                    if depth_grad_net < 1e-8:
                        depth_grad_zero_streak += 1
                    else:
                        depth_grad_zero_streak = 0
                    if depth_grad_zero_streak >= 3:
                        print(
                            "[WARN][depth] Depth-Gradient ~0 ueber mehrere Checks -> Depth lernt nicht.",
                            flush=True,
                        )
    
                    if gain_val is not None:
                        gain_mean = float(gain_val.detach().mean().item())
                        gain_std = float(gain_val.detach().std().item())
                        print(
                            f"[gain][depth] gain_mean={gain_mean:.3f} gain_std={gain_std:.3f} "
                            f"| depth_reg_total={depth_reg_total:.3e}",
                            flush=True,
                        )
                        if depth_reg_total < 1e-6 and abs(gain_mean - 1.0) > 0.1:
                            print(
                                f"[WARN][gain] gain_mean={gain_mean:.3f} bei depth_reg_total={depth_reg_total:.3e} "
                                "-> Gain kann Strukturfreiheit kompensieren.",
                                flush=True,
                            )
            proj_loss_for_grad = proj_weight * 0.5 * (loss_ap + loss_pa) if not skip_proj else torch.tensor(0.0, device=device)
    
            if debug_act_step:
                net_module = generator.render_kwargs_train.get("network_fn")
                grad_act_net = grad_norm_of_module(args.act_loss_weight * loss_act, net_module)
                grad_proj_net = grad_norm_of_module(proj_loss_for_grad, net_module)
                grad_act_enc = grad_norm_of_module(args.act_loss_weight * loss_act, encoder)
                grad_proj_enc = grad_norm_of_module(proj_loss_for_grad, encoder)
                print(
                    f"[DEBUG][ACT][gradcomp][step {step}] ||g_act||_net={grad_act_net:.3e} "
                    f"||g_proj||_net={grad_proj_net:.3e} ||g_act||_enc={grad_act_enc:.3e} "
                    f"||g_proj||_enc={grad_proj_enc:.3e}",
                    flush=True,
                )
    
            if args.grad_stats_every > 0 and (step % args.grad_stats_every) == 0:
                grad_stats = {
                    "proj": grad_norm_of(proj_loss_for_grad, [z_latent]),
                    "act": grad_norm_of(args.act_loss_weight * loss_act, [z_latent]) if args.act_loss_weight > 0 else 0.0,
                    "ct": grad_norm_of(args.ct_loss_weight * loss_ct, [z_latent]) if args.ct_loss_weight > 0 else 0.0,
                    "zreg": grad_norm_of(args.z_reg_weight * loss_reg, [z_latent]) if args.z_reg_weight > 0 else 0.0,
                }
                print(
                    f"[grad][step {step:05d}] ||g_proj||={grad_stats['proj']:.3e} "
                    f"| ||g_act||={grad_stats['act']:.3e} | ||g_ct||={grad_stats['ct']:.3e} "
                    f"| ||g_zreg||={grad_stats['zreg']:.3e}",
                    flush=True,
                )
    
            scaler.scale(loss).backward()
            if debug_act_step:
                net_module = generator.render_kwargs_train.get("network_fn")
                gain_param_grad = 0.0
                if gain_param is not None and gain_param.grad is not None:
                    gain_param_grad = float(gain_param.grad.detach().abs().mean().item())
                print(
                    f"[DEBUG][ACT][grads][step {step}] net={module_grad_mean_abs(net_module):.3e} "
                    f"encoder={module_grad_mean_abs(encoder):.3e} z_fuser={module_grad_mean_abs(z_fuser):.3e} "
                    f"gain_head={module_grad_mean_abs(gain_head):.3e} gain_param={gain_param_grad:.3e}",
                    flush=True,
                )
            grad_norm_global = global_grad_norm(opt_params) if hybrid_enabled else 0.0
            grad_norm_gen = torch.nn.utils.clip_grad_norm_(generator.parameters(), max_norm=1.0)
            clip_event = float(grad_norm_gen) > 1.0
            scaler.step(optimizer)
            scaler.update()
    
            torch.cuda.synchronize()
            iter_ms = (time.perf_counter() - t0) * 1000.0
    
            with torch.no_grad():
                if not proj_metrics_enabled:
                    mae_ap = float("nan")
                    mae_pa = float("nan")
                    pred_mean = (float("nan"), float("nan"))
                    pred_std = (float("nan"), float("nan"))
                    pred_mean_raw = (float("nan"), float("nan"))
                    pred_std_raw = (float("nan"), float("nan"))
                    psnr_ap = float("nan")
                    psnr_pa = float("nan")
                    psnr_ap_phys = None
                    psnr_pa_phys = None
                    mae_ap_phys = None
                    mae_pa_phys = None
                else:
                    mae_ap = torch.mean(torch.abs(pred_ap - target_ap)).item()
                    mae_pa = torch.mean(torch.abs(pred_pa - target_pa)).item()
                    pred_mean = (pred_ap.mean().item(), pred_pa.mean().item())              # skaliert gemäß Projektnorm
                    pred_std = (pred_ap.std().item(), pred_pa.std().item())
                    pred_mean_raw = (pred_ap_raw.mean().item(), pred_pa_raw.mean().item())  # physikalischer Maßstab
                    pred_std_raw = (pred_ap_raw.std().item(), pred_pa_raw.std().item())
                    psnr_ap = compute_psnr(pred_ap, target_ap)
                    psnr_pa = compute_psnr(pred_pa, target_pa)
                    psnr_ap_phys = None
                    psnr_pa_phys = None
                    mae_ap_phys = None
                    mae_pa_phys = None
                    if log_proj_metrics_physical:
                        pred_ap_phys = pred_ap * float(scale_ap_used)
                        pred_pa_phys = pred_pa * float(scale_pa_used)
                        target_ap_phys = target_ap * float(scale_ap_used)
                        target_pa_phys = target_pa * float(scale_pa_used)
                        psnr_ap_phys = compute_psnr(pred_ap_phys, target_ap_phys)
                        psnr_pa_phys = compute_psnr(pred_pa_phys, target_pa_phys)
                        mae_ap_phys = torch.mean(torch.abs(pred_ap_phys - target_ap_phys)).item()
                        mae_pa_phys = torch.mean(torch.abs(pred_pa_phys - target_pa_phys)).item()
                bg_depth_frac = float(bg_depth_frac_t.detach().cpu().item())
                if hybrid_enabled and hybrid_log_path is not None and need_raw_stats:
                    target_ap_stats = tensor_stats(target_ap)
                    target_pa_stats = tensor_stats(target_pa)
                    pred_ap_stats = tensor_stats(pred_ap)
                    pred_pa_stats = tensor_stats(pred_pa)
                    atten_scale = float(generator.render_kwargs_train.get("atten_scale", ATTEN_SCALE_DEFAULT))
                    (
                        lambda_stats,
                        mu_stats,
                        atten_stats,
                        atten_frac_gt20,
                        atten_frac_clamp,
                        nonfinite_lambda,
                        nonfinite_atten,
                    ) = compute_lambda_and_attenuation_stats([extras_ap, extras_pa], atten_scale=atten_scale)
                    nonfinite_pred = nonfinite_fraction(torch.cat([pred_ap, pred_pa], dim=1))
                    proj_scale_enc_val = (
                        float(proj_scale_enc.mean().item()) if torch.is_tensor(proj_scale_enc) else float("nan")
                    )
                    gain_log = float(gain_val.mean().item()) if gain_val is not None else float("nan")
                    z_train_l2 = float(z_train.detach().norm().item())
                    z_enc_l2 = (
                        float(z_enc_proj.detach().norm(dim=1).mean().item()) if z_enc_proj is not None else float("nan")
                    )
                    z_latent_l2 = float(z_latent.detach().norm(dim=1).mean().item())
    
                    def _stat(stats, key):
                        return float(stats[key]) if stats is not None and key in stats else float("nan")
    
                    print(
                        f"[hybrid][step {step:05d}] "
                        f"t_ap(min/mean/p95/max)=({_stat(target_ap_stats,'min'):.3e},"
                        f"{_stat(target_ap_stats,'mean'):.3e},{_stat(target_ap_stats,'p95'):.3e},"
                        f"{_stat(target_ap_stats,'max'):.3e}) "
                        f"t_pa(min/mean/p95/max)=({_stat(target_pa_stats,'min'):.3e},"
                        f"{_stat(target_pa_stats,'mean'):.3e},{_stat(target_pa_stats,'p95'):.3e},"
                        f"{_stat(target_pa_stats,'max'):.3e}) "
                        f"p_ap(min/mean/p95/max)=({_stat(pred_ap_stats,'min'):.3e},"
                        f"{_stat(pred_ap_stats,'mean'):.3e},{_stat(pred_ap_stats,'p95'):.3e},"
                        f"{_stat(pred_ap_stats,'max'):.3e}) "
                        f"p_pa(min/mean/p95/max)=({_stat(pred_pa_stats,'min'):.3e},"
                        f"{_stat(pred_pa_stats,'mean'):.3e},{_stat(pred_pa_stats,'p95'):.3e},"
                        f"{_stat(pred_pa_stats,'max'):.3e}) "
                        f"lambda_ray(min/mean/p95/max)=({_stat(lambda_stats,'min'):.3e},"
                        f"{_stat(lambda_stats,'mean'):.3e},{_stat(lambda_stats,'p95'):.3e},"
                        f"{_stat(lambda_stats,'max'):.3e}) "
                        f"mu(min/mean/p95/max)=({_stat(mu_stats,'min'):.3e},"
                        f"{_stat(mu_stats,'mean'):.3e},{_stat(mu_stats,'p95'):.3e},"
                        f"{_stat(mu_stats,'max'):.3e}) "
                        f"atten(min/mean/p95/max)=({_stat(atten_stats,'min'):.3e},"
                        f"{_stat(atten_stats,'mean'):.3e},{_stat(atten_stats,'p95'):.3e},"
                        f"{_stat(atten_stats,'max'):.3e}) "
                        f"gain={gain_log:.3e} "
                        f"atten>20={atten_frac_gt20 if atten_frac_gt20 is not None else float('nan'):.3f} "
                        f"atten=60={atten_frac_clamp if atten_frac_clamp is not None else float('nan'):.3f} "
                        f"nonfinite(pred/lambda/atten)=({nonfinite_pred:.3e},{nonfinite_lambda:.3e},{nonfinite_atten:.3e}) "
                        f"grad_norm={grad_norm_global:.3e} clip={int(clip_event)}",
                        flush=True,
                    )
    
                    append_hybrid_log(
                        hybrid_log_path,
                        [
                            step,
                            proj_weight,
                            float(loss_proj.item()),
                            float(loss_ap.item()),
                            float(loss_pa.item()),
                            float(loss_act.item()),
                            float(loss_gain.item()),
                            float(gain_prior) if gain_prior is not None else float("nan"),
                            float(act_norm_factor),
                            float(loss.item()),
                            proj_scale_enc_val,
                            _stat(target_ap_stats, "min"),
                            _stat(target_ap_stats, "mean"),
                            _stat(target_ap_stats, "p95"),
                            _stat(target_ap_stats, "max"),
                            _stat(target_pa_stats, "min"),
                            _stat(target_pa_stats, "mean"),
                            _stat(target_pa_stats, "p95"),
                            _stat(target_pa_stats, "max"),
                            _stat(pred_ap_stats, "min"),
                            _stat(pred_ap_stats, "mean"),
                            _stat(pred_ap_stats, "p95"),
                            _stat(pred_ap_stats, "max"),
                            _stat(pred_pa_stats, "min"),
                            _stat(pred_pa_stats, "mean"),
                            _stat(pred_pa_stats, "p95"),
                            _stat(pred_pa_stats, "max"),
                            _stat(lambda_stats, "min"),
                            _stat(lambda_stats, "mean"),
                            _stat(lambda_stats, "p95"),
                            _stat(lambda_stats, "max"),
                            _stat(mu_stats, "min"),
                            _stat(mu_stats, "mean"),
                            _stat(mu_stats, "p95"),
                            _stat(mu_stats, "max"),
                            _stat(atten_stats, "min"),
                            _stat(atten_stats, "mean"),
                            _stat(atten_stats, "p95"),
                            _stat(atten_stats, "max"),
                            float(atten_frac_gt20) if atten_frac_gt20 is not None else float("nan"),
                            float(atten_frac_clamp) if atten_frac_clamp is not None else float("nan"),
                            gain_log,
                            nonfinite_pred,
                            nonfinite_lambda,
                            nonfinite_atten,
                            grad_norm_global,
                            float(grad_norm_gen),
                            int(clip_event),
                            z_train_l2,
                            z_enc_l2,
                            z_latent_l2,
                        ],
                    )
                val_stats = None
                if val_interval > 0 and (step % val_interval) == 0 and (not args.no_val):
                    rays_eval = None if ray_split_enabled else rays_per_proj
                    # Testmetriken:
                    # test_all  → gesamter Test-Split (dominiert von BG, kann “zu gut” aussehen)
                    # test_fg   → nur Vordergrund-Rays, misst eigentliche Rekonstruktionsqualität
                    # test_top10→ oberste 10% Test-Intensitäten, fokussiert auf stärkste Aktivität
                    subsets = {
                        "test_all": ray_indices["pixel"]["test_idx_all"],
                    }
                    if ray_split_enabled:
                        subsets["test_fg"] = ray_indices["pixel"]["test_idx_fg"]
                        subsets["test_top10"] = ray_indices["pixel"]["test_idx_top10"]
                        subsets["test_bg"] = ray_indices["pixel"]["test_idx_bg"]
                    val_stats = evaluate_pixel_subsets(
                        generator,
                        z_latent.detach(),
                        rays_cache,
                        subsets=subsets,
                        ap_flat_proc=ap_flat_proc,
                        pa_flat_proc=pa_flat_proc,
                        rays_per_eval=rays_eval,
                        bg_weight=args.bg_weight,
                        weight_threshold=args.weight_threshold,
                        pa_xflip=pa_xflip,
                        ct_context=ct_context,
                        W=W,
                        scale_ap=scale_ap_used if log_proj_metrics_physical else None,
                        scale_pa=scale_pa_used if log_proj_metrics_physical else None,
                        loss_fn=loss_fn,
                        pred_scale=pred_to_counts_scale if (hybrid_enabled and args.proj_target_source == "counts") else 1.0,
                        gain=gain_val if (hybrid_enabled and args.proj_target_source == "counts") else None,
                    )
            val_all = val_stats.get("test_all") if isinstance(val_stats, dict) else None
            val_fg = val_stats.get("test_fg") if isinstance(val_stats, dict) else None
            val_top10 = val_stats.get("test_top10") if isinstance(val_stats, dict) else None
            val_bg = val_stats.get("test_bg") if isinstance(val_stats, dict) else None
    
            val_loss = val_all["loss"] if val_all is not None else None
            val_psnr = val_all["psnr"] if val_all is not None else None
            val_mae = val_all["mae"] if val_all is not None else None
    
            val_loss_fg = val_fg["loss"] if val_fg is not None else None
            val_psnr_fg = val_fg["psnr"] if val_fg is not None else None
            val_mae_fg = val_fg["mae"] if val_fg is not None else None
            val_loss_bg = val_bg["loss"] if val_bg is not None else None
            val_psnr_bg = val_bg["psnr"] if val_bg is not None else None
            val_mae_bg = val_bg["mae"] if val_bg is not None else None
            val_pred_mean_bg = val_bg.get("pred_mean") if val_bg is not None else None
            val_target_mean_bg = val_bg.get("target_mean") if val_bg is not None else None
            val_view_all = val_all.get("view") if val_all is not None else None
            val_phys_all = val_all.get("phys") if val_all is not None else None
            val_phys_fg = val_fg.get("phys") if val_fg is not None else None
            val_phys_bg = val_bg.get("phys") if val_bg is not None else None
            val_phys_top10 = val_top10.get("phys") if val_top10 is not None else None
            val_loss_ap = val_view_all["ap"]["loss"] if val_view_all is not None else None
            val_loss_pa = val_view_all["pa"]["loss"] if val_view_all is not None else None
            val_psnr_ap_val = val_view_all["ap"]["psnr"] if val_view_all is not None else None
            val_psnr_pa_val = val_view_all["pa"]["psnr"] if val_view_all is not None else None
            val_mae_ap_val = val_view_all["ap"]["mae"] if val_view_all is not None else None
            val_mae_pa_val = val_view_all["pa"]["mae"] if val_view_all is not None else None
    
            val_loss_top10 = val_top10["loss"] if val_top10 is not None else None
            val_psnr_top10 = val_top10["psnr"] if val_top10 is not None else None
            val_mae_top10 = val_top10["mae"] if val_top10 is not None else None
    
            msg = (
                f"[step {step:05d}] loss={loss.item():.6f} | act={loss_act.item():.6f} "
                f"| gain_reg={loss_gain.item():.6f} | ct={loss_ct.item():.6f} "
                f"| ray_tv={loss_ray_tv.item():.6f} | ray_tv_w={loss_ray_tv_w.item():.6f} "
                f"| bg_depth_mass={bg_depth_mass.item():.6f} | bg_depth_mass_w={bg_depth_mass_w.item():.6f} | bg_depth_frac={bg_depth_frac:.4f} "
                f"| tv={loss_tv.item():.6f} | zreg={loss_reg.item():.6f} "
            )
            if proj_metrics_enabled:
                msg += (
                    f"| ap={loss_ap.item():.6f} | pa={loss_pa.item():.6f} "
                    f"| mae_ap={mae_ap:.6f} | mae_pa={mae_pa:.6f} "
                    f"| psnr_ap={psnr_ap:.2f} | psnr_pa={psnr_pa:.2f} "
                    f"| predμ_raw=({pred_mean_raw[0]:.3e},{pred_mean_raw[1]:.3e}) predσ_raw=({pred_std_raw[0]:.3e},{pred_std_raw[1]:.3e}) "
                    f"| predμ=({pred_mean[0]:.3e},{pred_mean[1]:.3e}) predσ=({pred_std[0]:.3e},{pred_std[1]:.3e})"
                )
            if proj_metrics_enabled and log_proj_metrics_physical and psnr_ap_phys is not None:
                msg += (
                    f" | mae_ap_phys={mae_ap_phys:.6f} | mae_pa_phys={mae_pa_phys:.6f} "
                    f"| psnr_ap_phys={psnr_ap_phys:.2f} | psnr_pa_phys={psnr_pa_phys:.2f}"
                )
            if ray_tv_weight != 0.0:
                msg += f" | ray_tv_mode={ray_tv_mode}"
                if ray_tv_w_mean is not None:
                    msg += f" | ray_tv_w_mean={ray_tv_w_mean:.6f}"
                if ray_tv_w_min is not None and ray_tv_w_max is not None:
                    msg += f" | ray_tv_w_min={float(ray_tv_w_min):.6f} | ray_tv_w_max={float(ray_tv_w_max):.6f}"
                if ct_boundary_depth_mean is not None and ct_boundary_depth_median is not None:
                    msg += (
                        f" | ct_bnd_mean={float(ct_boundary_depth_mean):.3f}"
                        f" | ct_bnd_med={float(ct_boundary_depth_median):.3f}"
                    )
            if val_all is not None:
                msg += (
                    f" | test_all_loss={val_loss:.6f} | test_all_psnr={val_psnr:.2f} | test_all_mae={val_mae:.6f}"
                )
            if val_fg is not None:
                msg += (
                    f" | test_fg_loss={val_loss_fg:.6f} | test_fg_psnr={val_psnr_fg:.2f} | test_fg_mae={val_mae_fg:.6f}"
                )
            if val_top10 is not None:
                msg += (
                    f" | test_top10_loss={val_loss_top10:.6f} | test_top10_psnr={val_psnr_top10:.2f} "
                    f"| test_top10_mae={val_mae_top10:.6f}"
                )
            if val_bg is not None:
                msg += (
                    f" | test_bg_loss={val_loss_bg:.6f} | test_bg_psnr={val_psnr_bg:.2f} | test_bg_mae={val_mae_bg:.6f}"
                )
            if log_proj_metrics_physical and val_phys_all is not None:
                msg += (
                    f" | test_all_psnr_phys={val_phys_all['psnr']:.2f} | test_all_mae_phys={val_phys_all['mae']:.6f}"
                )
            if log_proj_metrics_physical and val_phys_fg is not None:
                msg += (
                    f" | test_fg_psnr_phys={val_phys_fg['psnr']:.2f} | test_fg_mae_phys={val_phys_fg['mae']:.6f}"
                )
            if log_proj_metrics_physical and val_phys_top10 is not None:
                msg += (
                    f" | test_top10_psnr_phys={val_phys_top10['psnr']:.2f} | test_top10_mae_phys={val_phys_top10['mae']:.6f}"
                )
            if log_proj_metrics_physical and val_phys_bg is not None:
                msg += (
                    f" | test_bg_psnr_phys={val_phys_bg['psnr']:.2f} | test_bg_mae_phys={val_phys_bg['mae']:.6f}"
                )
            if val_pred_mean_bg is not None and val_target_mean_bg is not None:
                print(
                    f"[ray-split-bg-check] mean target={val_target_mean_bg[0]:.3e}/{val_target_mean_bg[1]:.3e} "
                    f"mean pred={val_pred_mean_bg[0]:.3e}/{val_pred_mean_bg[1]:.3e}",
                    flush=True,
                )
            if val_view_all is not None:
                msg += (
                    f" | test_ap_loss={val_loss_ap:.6f} | test_ap_psnr={val_psnr_ap_val:.2f} | test_ap_mae={val_mae_ap_val:.6f}"
                    f" | test_pa_loss={val_loss_pa:.6f} | test_pa_psnr={val_psnr_pa_val:.2f} | test_pa_mae={val_mae_pa_val:.6f}"
                )
            print(msg, flush=True)
            append_log(
                log_path,
                [
                    step,
                    loss.item(),
                    loss_ap.item(),
                    loss_pa.item(),
                    loss_act.item(),
                    loss_ct.item(),
                    loss_ray_tv.item(),
                    loss_ray_tv_w.item(),
                    bg_depth_mass.item(),
                    bg_depth_mass_w.item(),
                    bg_depth_frac,
                    loss_tv.item(),
                    loss_reg.item(),
                    mae_ap,
                    mae_pa,
                    psnr_ap,
                    psnr_pa,
                    pred_mean[0],
                    pred_mean[1],
                    pred_std[0],
                    pred_std[1],
                    val_loss,
                    val_loss_ap,
                    val_loss_pa,
                    val_psnr,
                    val_psnr_ap_val,
                    val_psnr_pa_val,
                    val_mae,
                    val_mae_ap_val,
                    val_mae_pa_val,
                    val_loss_fg,
                    val_psnr_fg,
                    val_mae_fg,
                    val_loss_top10,
                    val_psnr_top10,
                    val_mae_top10,
                    iter_ms,
                    optimizer.param_groups[0]["lr"],
                    ray_tv_mode,
                    ray_tv_w_mean,
                ],
            )
            if args.save_every > 0 and (step % args.save_every == 0):
                save_checkpoint(
                    step,
                    generator,
                    z_train,
                    optimizer,
                    scaler,
                    ckpt_dir,
                    encoder=encoder,
                    z_fuser=z_fuser,
                    gain_head=gain_head,
                    gain_param=gain_param,
                )
            maybe_render_preview(
                step,
                args,
                generator,
                z_latent.detach(),
                outdir,
                ct_vol,
                act_vol,
                ct_context,
                target_ap=ap,
                target_pa=pa,
                target_ap_counts=ap_counts,
                target_pa_counts=pa_counts,
            )
            if args.export_vol_every > 0 and (step % args.export_vol_every == 0):
                export_path = outdir / f"activity_pred_step_{step:05d}.npy"
                export_activity_volume(generator, z_latent.detach(), export_path, args.export_vol_res, device)
    
        proj_metrics_enabled_final = not (args.act_only or (args.hybrid and args.proj_loss_weight <= 0.0))
        if proj_metrics_enabled_final:
            prev_flag = generator.use_test_kwargs
            generator.eval()
            generator.use_test_kwargs = True
            with torch.no_grad():
                proj_ap, _, _, _ = generator.render_from_pose(last_z_latent.detach(), generator.pose_ap, ct_context=ct_context)
                proj_pa, _, _, _ = generator.render_from_pose(last_z_latent.detach(), generator.pose_pa, ct_context=ct_context)
            generator.train()
            generator.use_test_kwargs = prev_flag or False
    
            use_counts_final = (
                (ap_counts is not None)
                and (pa_counts is not None)
                and (ap_counts.numel() > 0)
                and (pa_counts.numel() > 0)
            )
            if proj_loss_type == "poisson":
                lambda_ap_used = F.softplus(proj_ap) - math.log(2.0)
                lambda_pa_used = F.softplus(proj_pa) - math.log(2.0)
                lambda_ap_used = lambda_ap_used.clamp_min(1e-6)
                lambda_pa_used = lambda_pa_used.clamp_min(1e-6)
            else:
                lambda_ap_used = proj_ap
                lambda_pa_used = proj_pa
            if use_counts_final:
                lambda_ap_used = lambda_ap_used * float(pred_to_counts_scale)
                lambda_pa_used = lambda_pa_used * float(pred_to_counts_scale)
                gain_val_final = None
                if gain_param is not None:
                    gain_val_final = F.softplus(gain_param)
                elif gain_head is not None:
                    # compute z_enc for final batch (encoder sees normalized AP/PA)
                    proj_scale_enc = compute_proj_scale(ap, pa, args.proj_scale_source, meta)
                    proj_scale_enc = torch.clamp(proj_scale_enc, min=1e-6)
                    enc_input = build_encoder_input(
                        ap,
                        pa,
                        ct_vol,
                        proj_scale_enc,
                        args.encoder_proj_transform,
                        args.encoder_use_ct,
                    )
                    z_enc_final = encoder(enc_input)
                    gain_val_final = F.softplus(gain_head(z_enc_final))
                if gain_val_final is not None:
                    g_min = float(args.gain_clamp_min) if args.gain_clamp_min is not None else None
                    g_max = args.gain_clamp_max
                    if g_min is not None or g_max is not None:
                        gmin = g_min if g_min is not None else -float("inf")
                        gmax = g_max if g_max is not None else float("inf")
                        gain_val_final = torch.clamp(gain_val_final, min=gmin, max=gmax)
                    lambda_ap_used = lambda_ap_used * gain_val_final
                    lambda_pa_used = lambda_pa_used * gain_val_final
    
            H, W = generator.H, generator.W
            ap_np = lambda_ap_used[0].reshape(H, W).detach().cpu().numpy()
            pa_np = lambda_pa_used[0].reshape(H, W).detach().cpu().numpy()
            fp = outdir / "preview"
            fp.mkdir(parents=True, exist_ok=True)
            if args.log_quantiles_final_only:
                if use_counts_final:
                    ap_t_np = ap_counts.detach().cpu().numpy()[0]
                    pa_t_np = pa_counts.detach().cpu().numpy()[0]
                else:
                    ap_t_np = ap.detach().cpu().numpy()[0] if ap is not None else None
                    pa_t_np = pa.detach().cpu().numpy()[0] if pa is not None else None
                log_projection_quantiles(lambda_ap_used, lambda_pa_used, ap_target=ap_t_np, pa_target=pa_t_np, tag="final")
                if log_proj_metrics_physical:
                    log_projection_quantiles_scaled(
                        lambda_ap_used,
                        lambda_pa_used,
                        ap_target=ap_t_np,
                        pa_target=pa_t_np,
                        tag="final_phys",
                        ap_scale=scale_ap_used,
                        pa_scale=scale_pa_used,
                    )
                if use_counts_final:
                    raw_ap_q = np.quantile(proj_ap.detach().cpu().numpy().ravel(), [0.5, 0.8, 0.95, 0.99])
                    raw_pa_q = np.quantile(proj_pa.detach().cpu().numpy().ravel(), [0.5, 0.8, 0.95, 0.99])
                    soft_ap_q = np.quantile(F.softplus(proj_ap).detach().cpu().numpy().ravel(), [0.5, 0.8, 0.95, 0.99])
                    soft_pa_q = np.quantile(F.softplus(proj_pa).detach().cpu().numpy().ravel(), [0.5, 0.8, 0.95, 0.99])
                    lam_ap_q = np.quantile(lambda_ap_used.detach().cpu().numpy().ravel(), [0.5, 0.8, 0.95, 0.99])
                    lam_pa_q = np.quantile(lambda_pa_used.detach().cpu().numpy().ravel(), [0.5, 0.8, 0.95, 0.99])
                    tgt_ap_q = np.quantile(ap_t_np.ravel(), [0.5, 0.8, 0.95, 0.99]) if ap_t_np is not None else None
                    tgt_pa_q = np.quantile(pa_t_np.ravel(), [0.5, 0.8, 0.95, 0.99]) if pa_t_np is not None else None
                    print(
                        "[quantiles][final][diag] "
                        f"pred_raw_ap p50={raw_ap_q[0]:.3e} p80={raw_ap_q[1]:.3e} p95={raw_ap_q[2]:.3e} p99={raw_ap_q[3]:.3e} | "
                        f"pred_raw_pa p50={raw_pa_q[0]:.3e} p80={raw_pa_q[1]:.3e} p95={raw_pa_q[2]:.3e} p99={raw_pa_q[3]:.3e}",
                        flush=True,
                    )
                    print(
                        "[quantiles][final][diag] "
                        f"softplus_ap p50={soft_ap_q[0]:.3e} p80={soft_ap_q[1]:.3e} p95={soft_ap_q[2]:.3e} p99={soft_ap_q[3]:.3e} | "
                        f"softplus_pa p50={soft_pa_q[0]:.3e} p80={soft_pa_q[1]:.3e} p95={soft_pa_q[2]:.3e} p99={soft_pa_q[3]:.3e}",
                        flush=True,
                    )
                    print(
                        "[quantiles][final][diag] "
                        f"lambda_ap p50={lam_ap_q[0]:.3e} p80={lam_ap_q[1]:.3e} p95={lam_ap_q[2]:.3e} p99={lam_ap_q[3]:.3e} | "
                        f"lambda_pa p50={lam_pa_q[0]:.3e} p80={lam_pa_q[1]:.3e} p95={lam_pa_q[2]:.3e} p99={lam_pa_q[3]:.3e}",
                        flush=True,
                    )
                    if tgt_ap_q is not None and tgt_pa_q is not None:
                        print(
                            "[quantiles][final][diag] "
                            f"target_ap p50={tgt_ap_q[0]:.3e} p80={tgt_ap_q[1]:.3e} p95={tgt_ap_q[2]:.3e} p99={tgt_ap_q[3]:.3e} | "
                            f"target_pa p50={tgt_pa_q[0]:.3e} p80={tgt_pa_q[1]:.3e} p95={tgt_pa_q[2]:.3e} p99={tgt_pa_q[3]:.3e}",
                            flush=True,
                        )
                    if lam_ap_q[2] < 5 or lam_pa_q[2] < 5:
                        print("[WARN] Final pred quantiles appear non-count-scaled; logging wrong tensor.", flush=True)
            save_img(ap_np, fp / "final_AP.png", "AP final")
            save_img(pa_np, fp / "final_PA.png", "PA final")
            print("🖼️ Finale Previews gespeichert.", flush=True)
            print("   ", (fp / "final_AP.png").resolve(), flush=True)
            print("   ", (fp / "final_PA.png").resolve(), flush=True)
    
        export_activity_volume(generator, last_z_latent.detach(), outdir / "activity_pred_final.npy", args.export_vol_res, device)
        save_checkpoint(
            max_steps,
            generator,
            z_train,
            optimizer,
            scaler,
            ckpt_dir,
            encoder=encoder,
            z_fuser=z_fuser,
            gain_head=gain_head,
            gain_param=gain_param,
        )
        print("✅ Training run finished.", flush=True)
    
    
        print(f"[end] reached step={last_step} max_steps={max_steps} exit_reason={exit_reason}", flush=True)
    except Exception as exc:
        exit_reason = "exception"
        print(f"[exception] {exc.__class__.__name__}: {exc}", flush=True)
        traceback.print_exc()
        raise
    finally:
        print(f"[end] reached step={last_step} max_steps={max_steps} exit_reason={exit_reason}", flush=True)
if __name__ == "__main__":
    train()
