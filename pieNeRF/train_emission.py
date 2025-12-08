"""Mini-training script for the SPECT emission NeRF."""
import argparse
import csv
import math
import time
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from torch.utils.data import DataLoader

from graf.config import get_data, build_models

__VERSION__ = "emission-train v0.3"
DEBUG_PRINTS = False  # Nur Debug-Ausgaben, keine Änderung am Verhalten


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
        "--projection-normalization",
        type=str,
        default=None,
        choices=["none", "per_dataset", "per_projection"],
        help="Override data.projection_normalization from the config (none|per_dataset|per_projection).",
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
        "--act-loss-weight",
        type=float,
        default=0.0,
        help="Gewicht für einen optionalen Volumen-Loss gegen act.npy (0 = deaktiviert).",
    )
    parser.add_argument(
        "--act-samples",
        type=int,
        default=2048,
        help="Anzahl zufälliger Voxels zur act-Supervision pro Schritt.",
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
        "--debug-zero-var",
        action="store_true",
        help="Aktiviere zusätzliche Diagnostik und speichere Zwischenergebnisse, sobald Vorhersagen konstante Werte liefern.",
    )
    parser.add_argument(
        "--debug-attenuation-ray",
        action="store_true",
        help="Logge λ/μ/T für einen Beispielstrahl (benötigt nerf.attenuation_debug=True).",
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


def adjust_projection(arr: np.ndarray, view: str) -> np.ndarray:
    """
    Bringe die Projektionen in dieselbe Orientierung wie data_check.py:
    erst vertikal flippen, dann 90° im Uhrzeigersinn drehen.
    """
    flipped = np.flipud(arr)
    rotated = np.rot90(flipped, k=-1)
    return rotated


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


def build_ray_split(num_pixels: int, split_ratio: float, device: torch.device):
    """
    Erzeuge einen festen Train/Test-Split über alle Rays einer Ansicht.
    Split ist reproduzierbar, weil der globale Seed (set_seed) bereits gesetzt wurde.
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


def render_minibatch(generator, z_latent, rays_subset, need_raw: bool = False, ct_context=None):
    """Render a mini-batch of rays from a fixed pose while keeping training kwargs."""
    # train/test kwargs werden durch use_test_kwargs umgeschaltet
    render_kwargs = generator.render_kwargs_train if not generator.use_test_kwargs else generator.render_kwargs_test
    render_kwargs = dict(render_kwargs)
    render_kwargs["features"] = z_latent
    if need_raw:
        render_kwargs["retraw"] = True
    if ct_context is not None:
        render_kwargs["ct_context"] = ct_context
    elif render_kwargs.get("use_attenuation"):
        render_kwargs["use_attenuation"] = False
    if DEBUG_PRINTS:
        render_kwargs["debug_prints"] = True
    proj_map, _, _, extras = generator.render(rays=rays_subset, **render_kwargs)
    return proj_map.view(z_latent.shape[0], -1), extras


def maybe_render_preview(
    step,
    args,
    generator,
    z_eval,
    outdir,
    ct_volume=None,
    act_volume=None,
    ct_context=None,
):
    # Volle AP/PA-Renderings sind teuer; nur alle N Schritte ausführen
    if args.preview_every <= 0 or (step % args.preview_every) != 0:
        return
    prev_flag = generator.use_test_kwargs
    generator.eval()
    generator.use_test_kwargs = True
    ctx = ct_context or generator.build_ct_context(ct_volume)
    with torch.no_grad():
        proj_ap, _, _, _ = generator.render_from_pose(z_eval, generator.pose_ap, ct_context=ctx)
        proj_pa, _, _, _ = generator.render_from_pose(z_eval, generator.pose_pa, ct_context=ctx)
    generator.train()
    generator.use_test_kwargs = prev_flag or False
    H, W = generator.H, generator.W
    ap_np = proj_ap[0].reshape(H, W).detach().cpu().numpy()
    pa_np = proj_pa[0].reshape(H, W).detach().cpu().numpy()
    ap_np = adjust_projection(ap_np, "ap")
    pa_np = adjust_projection(pa_np, "pa")
    out_dir = outdir / "preview"
    out_dir.mkdir(parents=True, exist_ok=True)
    save_img(ap_np, out_dir / f"step_{step:05d}_AP.png", title=f"AP @ step {step}")
    save_img(pa_np, out_dir / f"step_{step:05d}_PA.png", title=f"PA @ step {step}")
    save_depth_profile(step, generator, z_eval, ct_volume, act_volume, out_dir, proj_ap=proj_ap, proj_pa=proj_pa)
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
                "tv",
                "mae_ap",
                "mae_pa",
                "psnr_ap",
                "psnr_pa",
                "pred_mean_ap",
                "pred_mean_pa",
                "pred_std_ap",
                "pred_std_pa",
                "loss_test",
                "psnr_ap_test",
                "psnr_pa_test",
                "mae_ap_test",
                "mae_pa_test",
                "iter_ms",
                "lr",
            ]
        )


def append_log(path: Path, row):
    with path.open("a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(row)


def save_checkpoint(step, generator, z_train, optimizer, scaler, ckpt_dir: Path):
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
    ckpt_path = ckpt_dir / f"checkpoint_step{step:05d}.pt"
    torch.save(state, ckpt_path)
    print(f"💾 Checkpoint gespeichert: {ckpt_path}", flush=True)


def dump_debug_tensor(outpath: Path, tensor: torch.Tensor):
    outpath.parent.mkdir(parents=True, exist_ok=True)
    torch.save(tensor.detach().cpu(), outpath)


def compute_psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
    mse = torch.mean((pred - target) ** 2).item()
    if mse <= 0:
        return float("inf")
    return -10.0 * math.log10(mse + 1e-12)


def evaluate_split(
    generator,
    z_latent,
    rays_cache,
    ray_indices,
    split: str,
    ap_flat_proc: torch.Tensor,
    pa_flat_proc: torch.Tensor,
    rays_per_proj_eval: int,
    bg_weight: float,
    weight_threshold: float,
    ct_context=None,
):
    """Evaluiert Poisson-NLL/PSNR auf einem festen Split (train/test) ohne Gradienten."""
    idx_ap_all = ray_indices["ap"].get(split)
    idx_pa_all = ray_indices["pa"].get(split)
    if idx_ap_all is None or idx_pa_all is None or idx_ap_all.numel() == 0 or idx_pa_all.numel() == 0:
        return None

    n_ap = min(idx_ap_all.numel(), rays_per_proj_eval)
    n_pa = min(idx_pa_all.numel(), rays_per_proj_eval)
    idx_ap = sample_split_indices(idx_ap_all, n_ap)
    idx_pa = sample_split_indices(idx_pa_all, n_pa)

    prev_flag = generator.use_test_kwargs
    # Eval-Mode aktivieren (kein Grad, festes Test-Split)
    generator.eval()
    generator.use_test_kwargs = True
    with torch.no_grad():
        ray_batch_ap = slice_rays(rays_cache["ap"], idx_ap)
        ray_batch_pa = slice_rays(rays_cache["pa"], idx_pa)

        pred_ap, _ = render_minibatch(generator, z_latent, ray_batch_ap, need_raw=False, ct_context=ct_context)
        pred_pa, _ = render_minibatch(generator, z_latent, ray_batch_pa, need_raw=False, ct_context=ct_context)

        target_ap = ap_flat_proc[0, idx_ap].unsqueeze(0)
        target_pa = pa_flat_proc[0, idx_pa].unsqueeze(0)

        pred_ap = pred_ap.clamp_min(1e-8)
        pred_pa = pred_pa.clamp_min(1e-8)

        weight_ap = build_loss_weights(target_ap, bg_weight, weight_threshold)
        weight_pa = build_loss_weights(target_pa, bg_weight, weight_threshold)
        loss_ap = poisson_nll(pred_ap, target_ap, weight=weight_ap)
        loss_pa = poisson_nll(pred_pa, target_pa, weight=weight_pa)
        loss_total = loss_ap + loss_pa

        psnr_ap = compute_psnr(pred_ap, target_ap)
        psnr_pa = compute_psnr(pred_pa, target_pa)
        mae_ap = torch.mean(torch.abs(pred_ap - target_ap)).item()
        mae_pa = torch.mean(torch.abs(pred_pa - target_pa)).item()

    # Ursprünglichen Modus wiederherstellen (Train/Eval + use_test_kwargs)
    if prev_flag:
        generator.eval()
    else:
        generator.train()

    return {
        "loss": loss_total.item(),
        "loss_ap": loss_ap.item(),
        "loss_pa": loss_pa.item(),
        "psnr_ap": psnr_ap,
        "psnr_pa": psnr_pa,
        "mae_ap": mae_ap,
        "mae_pa": mae_pa,
    }


def sample_act_points(act: torch.Tensor, nsamples: int, radius: float) -> Tuple[torch.Tensor, torch.Tensor]:
    """Zufällige Voxel (coords, values) aus act.npy ziehen."""
    if act is None:
        raise ValueError("act tensor missing despite act-loss-weight > 0.")
    # Unterscheide (1,D,H,W) vs (D,H,W)
    if act.dim() == 4:
        act = act.squeeze(0)
    D, H, W = act.shape[-3:]
    flat = act.view(-1)
    nsamples = min(nsamples, flat.numel())
    idx = torch.randint(0, flat.numel(), (nsamples,), device=act.device)
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
    return coords, values


def query_emission_at_points(generator, z_latent, coords: torch.Tensor) -> torch.Tensor:
    """Fragt das NeRF an frei gewählten Koordinaten ab (ohne Integration)."""
    if coords.numel() == 0:
        return torch.tensor([], device=coords.device)
    render_kwargs = generator.render_kwargs_train
    network_fn = render_kwargs["network_fn"]
    network_query_fn = render_kwargs["network_query_fn"]
    pts = coords.unsqueeze(0)
    raw = network_query_fn(pts, None, network_fn, features=z_latent)
    raw = raw.view(-1, raw.shape[-1])
    return F.softplus(raw[:, 0])


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


def save_depth_profile(step, generator, z_latent, ct_vol, act_vol, outdir: Path, proj_ap=None, proj_pa=None):
    # Nur aktiv, wenn Ground Truth CT oder act existieren
    if ct_vol is None and act_vol is None:
        return

    def extract_curve(vol, y_idx: int, x_idx: int):
        if vol is None:
            return None, None
        data = vol.squeeze(0).detach().cpu().numpy() if vol.dim() == 4 else vol.detach().cpu().numpy()
        if data.ndim != 3:
            return None, None
        D, H, W = data.shape
        # Falls H/W vertauscht sind (z. B. 651x256 statt 256x651), tauschen.
        if H == generator.W and W == generator.H:
            data = np.transpose(data, (0, 2, 1))
            D, H, W = data.shape
        y_idx = int(np.clip(y_idx, 0, H - 1))
        x_idx = int(np.clip(x_idx, 0, W - 1))
        return data[:, y_idx, x_idx], (D, H, W)

    def to_np_image(tensor):
        if tensor is None:
            return None
        return tensor.detach().cpu().reshape(generator.H, generator.W).numpy()

    ap_img = to_np_image(proj_ap)
    pa_img = to_np_image(proj_pa)

    act_data = None
    act_masks = None
    if act_vol is not None:
        act_data = act_vol.squeeze(0).detach().cpu().numpy() if act_vol.dim() == 4 else act_vol.detach().cpu().numpy()
        if act_data.ndim != 3:
            act_data = None
        else:
            if act_data.shape[1] == generator.W and act_data.shape[2] == generator.H:
                act_data = np.transpose(act_data, (0, 2, 1))
            depth_max = act_data.max(axis=0)
            if depth_max.shape == (generator.W, generator.H):
                depth_max = depth_max.T
            elif depth_max.shape != (generator.H, generator.W):
                depth_max = None
            if depth_max is not None:
                act_masks = (depth_max <= 1e-8, depth_max > 1e-8)

    ct_data = None
    ct_depth_max = None
    if ct_vol is not None:
        ct_data = ct_vol.squeeze(0).detach().cpu().numpy() if ct_vol.dim() == 4 else ct_vol.detach().cpu().numpy()
        if ct_data.ndim == 3:
            if ct_data.shape[1] == generator.W and ct_data.shape[2] == generator.H:
                ct_data = np.transpose(ct_data, (0, 2, 1))
            ct_depth_max = ct_data.max(axis=0)
            if ct_depth_max.shape == (generator.W, generator.H):
                ct_depth_max = ct_depth_max.T
            elif ct_depth_max.shape != (generator.H, generator.W):
                ct_depth_max = None


    def pick_ray_indices(num: int = 2):
        H, W = generator.H, generator.W
        chosen = []

        def add_unique(idx):
            if idx not in chosen:
                chosen.append(idx)

        # feste Strahlen (y,x), auf Bildgröße geclippt
        fixed_coords = [(72, 428), (69, 336)]
        for y_raw, x_raw in fixed_coords:
            if len(chosen) >= num:
                break
            y = int(np.clip(y_raw, 0, H - 1))
            x = int(np.clip(x_raw, 0, W - 1))
            add_unique((y, x))
        if len(chosen) >= num:
            return chosen[:num]

        def pick_from_mask(mask):
            if mask is None:
                return None
            mask = mask.astype(bool)
            if mask.shape != (H, W) or not mask.any():
                return None
            coords = np.argwhere(mask)
            if coords.size == 0:
                return None
            if ap_img is not None and pa_img is not None:
                weight_map = ap_img + pa_img
                weight_map = np.where(mask, weight_map, -np.inf)
                y, x = np.unravel_index(np.nanargmax(weight_map), weight_map.shape)
                return int(y), int(x)
            yx = coords[0]
            return int(yx[0]), int(yx[1])

        if act_data is not None:
            # Falls act verfügbar ist, gezielt Null-/Nicht-Null-Strahlen auswählen
            if act_masks is not None:
                zero_mask, nonzero_mask = act_masks
            else:
                zero_mask = nonzero_mask = None

            ct_pos_mask = None
            if ct_depth_max is not None:
                ct_pos_mask = ct_depth_max > 1e-8

            def combine(m_act, require_ct: bool):
                if m_act is None:
                    return None
                if ct_pos_mask is not None and require_ct:
                    combo = m_act & ct_pos_mask
                    if combo.any():
                        return combo
                return m_act

            zero_idx = pick_from_mask(combine(zero_mask, require_ct=True))
            nonzero_idx = pick_from_mask(combine(nonzero_mask, require_ct=True))
            if zero_idx is None and zero_mask is not None:
                zero_idx = pick_from_mask(zero_mask)
            if nonzero_idx is None and nonzero_mask is not None:
                nonzero_idx = pick_from_mask(nonzero_mask)
            if zero_idx is not None:
                add_unique(zero_idx)
            if nonzero_idx is not None:
                add_unique(nonzero_idx)

        rel_coords = [(0.5, 0.5), (0.25, 0.75), (0.75, 0.25)]
        for ry, rx in rel_coords:
            if len(chosen) >= num:
                break
            y = int(np.clip(round((H - 1) * ry), 0, H - 1))
            x = int(np.clip(round((W - 1) * rx), 0, W - 1))
            add_unique((y, x))
            if len(chosen) >= num:
                break

        if ap_img is not None and pa_img is not None and len(chosen) < num:
            max_y, max_x = np.unravel_index(np.argmax(ap_img + pa_img), ap_img.shape)
            add_unique((int(max_y), int(max_x)))

        return chosen[:num]

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

    depth_idx = torch.arange(D, device=generator.device)
    z_coords = idx_to_coord(depth_idx, D, radius)
    depth_axis = np.linspace(0.0, 1.0, D)
    import matplotlib.pyplot as plt

    ray_indices = pick_ray_indices(num=2)
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
        coords = torch.stack(
            (x_coord.repeat(D), y_coord.repeat(D), z_coords),
            dim=1,
        )
        # Vorhersage entlang der Tiefe an genau diesem Pixel extrahieren
        pred = query_emission_at_points(generator, z_latent, coords).detach().cpu().numpy()
        curves.append(normalize_curve(pred.copy()))
        labels.append("Aktivität (NeRF)")

        for curve, label in zip(curves, labels):
            ax.plot(depth_axis, curve, label=label)

        title_extra = []
        if ap_img is not None:
            title_extra.append(f"I_AP={ap_img[y_idx, x_idx]:.2e}")
        if pa_img is not None:
            title_extra.append(f"I_PA={pa_img[y_idx, x_idx]:.2e}")
        aux = " | ".join(title_extra)
        ax.set_title(f"Strahl y={y_idx}, x={x_idx}" + (f"\n{aux}" if aux else ""))
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


def log_attenuation_profile(step: int, view: str, extras: dict):
    """Druckt λ/μ/T über der Strahltiefe für Debugging-Zwecke."""
    if extras is None:
        return
    lambda_vals = extras.get("debug_lambda")
    if lambda_vals is None:
        return
    ray_idx = 0
    to_np = lambda tensor: tensor[ray_idx].detach().cpu().numpy()
    lam = to_np(lambda_vals)
    mu_vals = extras.get("debug_mu")
    mu = to_np(mu_vals) if mu_vals is not None else None
    trans_vals = extras.get("debug_transmission")
    trans = to_np(trans_vals) if trans_vals is not None else None
    dists = extras.get("debug_dists")
    d = to_np(dists) if dists is not None else None
    weights = extras.get("debug_weights")
    contrib = to_np(weights) if weights is not None else None
    intensity = float(np.sum(contrib)) if contrib is not None else float(np.sum(lam))

    def fmt(arr):
        if arr is None:
            return "n/a"
        return np.array2string(arr, precision=4, separator=", ")

    print(
        f"[attenuation-debug][{view}][step {step:05d}] I={intensity:.4e} | "
        f"λ={fmt(lam)} | μ={fmt(mu)} | Δ={fmt(d)} | T={fmt(trans)} | λ·T·Δ={fmt(contrib)}",
        flush=True,
    )


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

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA device required – please launch on a GPU node.")

    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True
    set_seed(args.seed)

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    data_cfg = config.setdefault("data", {})
    if args.normalize_targets:
        print("⚠️ --normalize-targets ist veraltet – Projektionen werden bereits im Loader auf [0,1] normiert.", flush=True)
    if args.projection_normalization is not None:
        data_cfg["projection_normalization"] = args.projection_normalization
    proj_mode = data_cfg.setdefault("projection_normalization", "none").lower()
    if proj_mode != "none":
        print("⚠️ projection_normalization != 'none' wird ignoriert – Loader normiert jedes Bild einzeln.", flush=True)
        data_cfg["projection_normalization"] = "none"
    data_cfg.setdefault("act_scale", 1.0)
    data_cfg.setdefault("ray_split_ratio", 0.8)
    training_cfg = config.setdefault("training", {})
    training_cfg.setdefault("val_interval", 0)
    training_cfg.setdefault("tv_weight", 0.001)
    training_cfg["tv_weight"] = args.tv_weight

    print(f"📂 CWD: {Path.cwd().resolve()}", flush=True)
    outdir = Path(config.get("training", {}).get("outdir", "./results_spect")).expanduser().resolve()
    (outdir / "preview").mkdir(parents=True, exist_ok=True)
    print(f"🗂️ Output-Ordner: {outdir}", flush=True)
    ckpt_dir = outdir / "checkpoints"
    log_path = outdir / "train_log.csv"
    init_log_file(log_path)
    debug_dir = outdir / "debug_dump"

    dataset, hwfr, _ = get_data(config)
    config["data"]["hwfr"] = hwfr

    batch_size = config["training"]["batch_size"]
    if batch_size != 1:
        raise ValueError("This mini-training script currently assumes batch_size == 1.")

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
    val_interval = int(training_cfg.get("val_interval", 0) or 0)
    tv_weight = float(training_cfg.get("tv_weight", 0.0))

    generator = build_models(config)
    generator.to(device)
    generator.train()
    generator.use_test_kwargs = False  # enforce training kwargs
    if args.debug_attenuation_ray:
        generator.render_kwargs_train["attenuation_debug"] = True
        generator.render_kwargs_test["attenuation_debug"] = True

    # always provide AP/PA fallback poses if not already configured
    generator.set_fixed_ap_pa(radius=hwfr[3])

    z_dim = config["z_dist"]["dim"]
    z_train = torch.nn.Parameter(torch.zeros(1, z_dim, device=device))
    torch.nn.init.normal_(z_train, mean=0.0, std=1.0)

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
    # Fester Train/Test-Split pro View (reproduzierbar via globalem Seed).
    ray_indices = {
        "ap": build_ray_split(num_pixels, ray_split_ratio, device),
        "pa": build_ray_split(num_pixels, ray_split_ratio, device),
    }
    print(
        f"🔀 Ray-Split (AP/PA): train={ray_indices['ap']['train'].numel()} / test={ray_indices['ap']['test'].numel()} "
        f"(ratio={ray_split_ratio})",
        flush=True,
    )
    if DEBUG_PRINTS:
        print(
            f"[DEBUG] AP train rays: {len(ray_indices['ap']['train'])} | AP test rays: {len(ray_indices['ap']['test'])} | "
            f"PA train rays: {len(ray_indices['pa']['train'])} | PA test rays: {len(ray_indices['pa']['test'])}",
            flush=True,
        )

    rays_per_proj = args.rays_per_step or config["training"]["chunk"]
    if rays_per_proj <= 0:
        raise ValueError("rays-per-step must be > 0.")
    rays_per_proj = min(rays_per_proj, num_pixels)

    optimizer = torch.optim.Adam(
        list(generator.parameters()) + [z_train],
        lr=config["training"]["lr_g"],
    )
    # Poisson-basierter Loss
    loss_fn = poisson_nll

    amp_enabled = bool(config["training"].get("use_amp", False))
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    data_iter = iter(dataloader)
    ct_context = None

    print(
        f"🚀 Starting emission-NeRF training | steps={args.max_steps} | rays/proj={rays_per_proj} "
        f"| image={generator.H}x{generator.W} | chunk={generator.chunk}"
    )

    for step in range(1, args.max_steps + 1):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

        ap = batch["ap"].to(device, non_blocking=True).float()
        pa = batch["pa"].to(device, non_blocking=True).float()
        act_vol = batch.get("act")
        if act_vol is not None:
            if act_vol.numel() == 0:
                act_vol = None
            else:
                act_vol = act_vol.to(device, non_blocking=True)
        ct_vol = batch.get("ct")
        if ct_vol is not None:
            ct_vol = ct_vol.to(device, non_blocking=True).float()
        ct_context = generator.build_ct_context(ct_vol) if ct_vol is not None else None

        ap_flat = ap.view(batch_size, -1)
        pa_flat = pa.view(batch_size, -1)
        ap_flat_proc = ap_flat
        pa_flat_proc = pa_flat

        z_latent = z_train

        optimizer.zero_grad(set_to_none=True)
        t0 = time.perf_counter()

        idx_ap = sample_split_indices(ray_indices["ap"]["train"], rays_per_proj)
        idx_pa = sample_split_indices(ray_indices["pa"]["train"], rays_per_proj)

        ray_batch_ap = slice_rays(rays_cache["ap"], idx_ap)
        ray_batch_pa = slice_rays(rays_cache["pa"], idx_pa)

        with torch.cuda.amp.autocast(enabled=amp_enabled):
            pred_ap, extras_ap = render_minibatch(
                generator, z_latent, ray_batch_ap, need_raw=args.debug_zero_var, ct_context=ct_context
            )
            pred_pa, extras_pa = render_minibatch(
                generator, z_latent, ray_batch_pa, need_raw=args.debug_zero_var, ct_context=ct_context
            )

            target_ap = ap_flat_proc[0, idx_ap].unsqueeze(0)
            target_pa = pa_flat_proc[0, idx_pa].unsqueeze(0)

            # Poisson-NLL erwartet pred >= 0
            pred_ap_raw = pred_ap.clamp_min(1e-8)
            pred_pa_raw = pred_pa.clamp_min(1e-8)

            pred_ap = pred_ap_raw
            pred_pa = pred_pa_raw

            weight_ap = build_loss_weights(target_ap, args.bg_weight, args.weight_threshold)
            weight_pa = build_loss_weights(target_pa, args.bg_weight, args.weight_threshold)

            loss_ap = poisson_nll(pred_ap, target_ap, weight=weight_ap)
            loss_pa = poisson_nll(pred_pa, target_pa, weight=weight_pa)
            loss = loss_ap + loss_pa
            if DEBUG_PRINTS and (step % 50 == 0):
                print(
                    f"[DEBUG][step {step}] TARGET AP min/max: {target_ap.min().item():.3e}/{target_ap.max().item():.3e} | "
                    f"PRED AP min/max: {pred_ap.min().item():.3e}/{pred_ap.max().item():.3e} | "
                    f"TARGET PA min/max: {target_pa.min().item():.3e}/{target_pa.max().item():.3e} | "
                    f"PRED PA min/max: {pred_pa.min().item():.3e}/{pred_pa.max().item():.3e}",
                    flush=True,
                )
            if args.debug_attenuation_ray:
                log_attenuation_profile(step, "AP", extras_ap)
                log_attenuation_profile(step, "PA", extras_pa)

            loss_act = torch.tensor(0.0, device=device)
            if args.act_loss_weight > 0.0 and act_vol is not None:
                radius = generator.radius
                if isinstance(radius, tuple):
                    radius = radius[1]
                # Stichprobe aus act.npy und direkte Dichteabfrage im NeRF
                coords, act_samples = sample_act_points(act_vol, args.act_samples, radius=radius)
                pred_act = query_emission_at_points(generator, z_latent, coords)
                loss_act = F.l1_loss(pred_act, act_samples)
                loss = loss + args.act_loss_weight * loss_act

            loss_ct = torch.tensor(0.0, device=device)
            if args.ct_loss_weight > 0.0 and ct_vol is not None:
                radius = generator.radius
                if isinstance(radius, tuple):
                    radius = radius[1]
                ct_pairs = sample_ct_pairs(ct_vol, args.ct_samples, args.ct_threshold, radius=radius)
                if ct_pairs is not None:
                    coords1, coords2, weights = ct_pairs
                    pred1 = query_emission_at_points(generator, z_latent, coords1)
                    pred2 = query_emission_at_points(generator, z_latent, coords2)
                    # Loss zwingt Emission auf flachen CT-Strecken zur Konstanz
                    loss_ct = torch.mean(torch.abs(pred1 - pred2) * weights)
                    loss = loss + args.ct_loss_weight * loss_ct

            loss_reg = torch.tensor(0.0, device=device)
            if args.z_reg_weight > 0.0:
                loss_reg = z_latent.pow(2).mean()
                loss = loss + args.z_reg_weight * loss_reg

            tv_loss = torch.tensor(0.0, device=device)
            loss_tv = torch.tensor(0.0, device=device)
            if tv_weight != 0.0:
                tv_terms = []
                tv_ap = extras_ap.get("tv_loss") if isinstance(extras_ap, dict) else None
                tv_pa = extras_pa.get("tv_loss") if isinstance(extras_pa, dict) else None
                if tv_ap is not None:
                    tv_terms.append(tv_ap)
                if tv_pa is not None:
                    tv_terms.append(tv_pa)
                if tv_terms:
                    tv_loss = torch.stack(tv_terms).mean()
                    loss_tv = tv_weight * tv_loss
                    loss = loss + loss_tv

        scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(generator.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        torch.cuda.synchronize()
        iter_ms = (time.perf_counter() - t0) * 1000.0

        with torch.no_grad():
            mae_ap = torch.mean(torch.abs(pred_ap - target_ap)).item()
            mae_pa = torch.mean(torch.abs(pred_pa - target_pa)).item()
            pred_mean = (pred_ap.mean().item(), pred_pa.mean().item())              # skaliert gemäß Projektnorm
            pred_std = (pred_ap.std().item(), pred_pa.std().item())
            pred_mean_raw = (pred_ap_raw.mean().item(), pred_pa_raw.mean().item())  # physikalischer Maßstab
            pred_std_raw = (pred_ap_raw.std().item(), pred_pa_raw.std().item())
            psnr_ap = compute_psnr(pred_ap, target_ap)
            psnr_pa = compute_psnr(pred_pa, target_pa)
            val_stats = None
            if val_interval > 0 and (step % val_interval) == 0 and (not args.no_val):
                # Test-Split evaluiert strikt auf den fixen Test-Rays, Gradienten bleiben aus.
                rays_eval = rays_per_proj
                val_stats = evaluate_split(
                    generator,
                    z_latent.detach(),
                    rays_cache,
                    ray_indices,
                    split="test",
                    ap_flat_proc=ap_flat_proc,
                    pa_flat_proc=pa_flat_proc,
                    rays_per_proj_eval=rays_eval,
                    bg_weight=args.bg_weight,
                    weight_threshold=args.weight_threshold,
                    ct_context=ct_context,
                )
            if args.debug_zero_var:
                targ_std = (target_ap.std().item(), target_pa.std().item())
                if pred_std[0] < 1e-7 or pred_std[1] < 1e-7:
                    print("⚠️ Zero-Var Vorhersage erkannt – dumppe Debug-Daten ...", flush=True)
                    dump_debug_tensor(debug_dir / f"step_{step:05d}_pred_ap.pt", pred_ap)
                    dump_debug_tensor(debug_dir / f"step_{step:05d}_pred_pa.pt", pred_pa)
                    dump_debug_tensor(debug_dir / f"step_{step:05d}_target_ap.pt", target_ap)
                    dump_debug_tensor(debug_dir / f"step_{step:05d}_target_pa.pt", target_pa)
                    dump_debug_tensor(debug_dir / f"step_{step:05d}_rays_ap.pt", ray_batch_ap)
                    dump_debug_tensor(debug_dir / f"step_{step:05d}_rays_pa.pt", ray_batch_pa)
                    if extras_ap.get("raw") is not None:
                        dump_debug_tensor(debug_dir / f"step_{step:05d}_raw_ap.pt", extras_ap["raw"])
                    if extras_pa.get("raw") is not None:
                        dump_debug_tensor(debug_dir / f"step_{step:05d}_raw_pa.pt", extras_pa["raw"])
                    print(
                        f"   targetσ=({targ_std[0]:.3e},{targ_std[1]:.3e}) "
                        f"| predμ=({pred_mean[0]:.3e},{pred_mean[1]:.3e})",
                        flush=True,
                    )

        val_loss = val_stats["loss"] if val_stats is not None else None
        val_psnr_ap = val_stats["psnr_ap"] if val_stats is not None else None
        val_psnr_pa = val_stats["psnr_pa"] if val_stats is not None else None
        val_mae_ap = val_stats["mae_ap"] if val_stats is not None else None
        val_mae_pa = val_stats["mae_pa"] if val_stats is not None else None

        msg = (
            f"[step {step:05d}] loss={loss.item():.6f} | ap={loss_ap.item():.6f} | pa={loss_pa.item():.6f} "
            f"| act={loss_act.item():.6f} | ct={loss_ct.item():.6f} | tv={loss_tv.item():.6f} | zreg={loss_reg.item():.6f} "
            f"| mae_ap={mae_ap:.6f} | mae_pa={mae_pa:.6f} "
            f"| psnr_ap={psnr_ap:.2f} | psnr_pa={psnr_pa:.2f} "
            f"| predμ_raw=({pred_mean_raw[0]:.3e},{pred_mean_raw[1]:.3e}) predσ_raw=({pred_std_raw[0]:.3e},{pred_std_raw[1]:.3e}) "
            f"| predμ=({pred_mean[0]:.3e},{pred_mean[1]:.3e}) predσ=({pred_std[0]:.3e},{pred_std[1]:.3e})"
        )
        if val_stats is not None:
            msg += (
                f" | test_loss={val_loss:.6f} | test_psnr_ap={val_psnr_ap:.2f} | test_psnr_pa={val_psnr_pa:.2f} "
                f"| test_mae_ap={val_mae_ap:.6f} | test_mae_pa={val_mae_pa:.6f}"
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
                loss_tv.item(),
                mae_ap,
                mae_pa,
                psnr_ap,
                psnr_pa,
                pred_mean[0],
                pred_mean[1],
                pred_std[0],
                pred_std[1],
                val_loss,
                val_psnr_ap,
                val_psnr_pa,
                val_mae_ap,
                val_mae_pa,
                iter_ms,
                optimizer.param_groups[0]["lr"],
            ],
        )
        if args.save_every > 0 and (step % args.save_every == 0):
            save_checkpoint(step, generator, z_train, optimizer, scaler, ckpt_dir)
        maybe_render_preview(
            step,
            args,
            generator,
            z_train.detach(),
            outdir,
            ct_vol,
            act_vol,
            ct_context,
        )

    prev_flag = generator.use_test_kwargs
    generator.eval()
    generator.use_test_kwargs = True
    with torch.no_grad():
        proj_ap, _, _, _ = generator.render_from_pose(z_train.detach(), generator.pose_ap, ct_context=ct_context)
        proj_pa, _, _, _ = generator.render_from_pose(z_train.detach(), generator.pose_pa, ct_context=ct_context)
    generator.train()
    generator.use_test_kwargs = prev_flag or False

    H, W = generator.H, generator.W
    ap_np = proj_ap[0].reshape(H, W).detach().cpu().numpy()
    pa_np = proj_pa[0].reshape(H, W).detach().cpu().numpy()
    ap_np = adjust_projection(ap_np, "ap")
    pa_np = adjust_projection(pa_np, "pa")
    fp = outdir / "preview"
    fp.mkdir(parents=True, exist_ok=True)
    save_img(ap_np, fp / "final_AP.png", "AP final")
    save_img(pa_np, fp / "final_PA.png", "PA final")
    print("🖼️ Finale Previews gespeichert.", flush=True)
    print("   ", (fp / "final_AP.png").resolve(), flush=True)
    print("   ", (fp / "final_PA.png").resolve(), flush=True)

    save_checkpoint(args.max_steps, generator, z_train, optimizer, scaler, ckpt_dir)
    print("✅ Training run finished.", flush=True)


if __name__ == "__main__":
    train()
