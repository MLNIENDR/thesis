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
        default=0,
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
        help="Apply per-projection min/max normalisation to both targets and predictions.",
    )
    parser.add_argument(
        "--bg-weight",
        type=float,
        default=1.0,
        help="Down-weights Hintergrundstrahlen im Loss (<1 reduziert Null-Strahlen, 1 = deaktiviert).",
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
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)


def save_img(arr, path, title=None):
    """Robust PNG visualisation with optional logarithmic stretch."""
    import matplotlib.pyplot as plt

    if not np.isfinite(arr).all():
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

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
    if view.lower() == "ap":
        return np.rot90(arr, k=1)
    if view.lower() == "pa":
        rotated = np.rot90(arr, k=-1)
        return np.fliplr(rotated)
    return arr


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
    """
    pred = pred.clamp_min(eps).clamp_max(clamp_max)
    nll = pred - target * torch.log(pred)
    if weight is not None:
        nll = nll * weight
    return nll.mean()


def normalize_counts(x: torch.Tensor, return_params: bool = False):
    """Normalise per projection to [0, 1] to stabilise the loss."""
    minv = torch.amin(x)
    maxv = torch.amax(x)
    span = maxv - minv
    applied = bool((span > 1e-8).item())
    if applied:
        norm = (x - minv) / (span + 1e-8)
    else:
        norm = torch.zeros_like(x)
    if return_params:
        return norm, minv, maxv, applied
    return norm


def build_loss_weights(target: torch.Tensor, bg_weight: float, threshold: float) -> Optional[torch.Tensor]:
    """Erzeuge optionale Strahl-Gewichte, die Null-Strahlen abschwächen."""
    if bg_weight >= 1.0:
        return None
    weights = torch.ones_like(target)
    weights = weights.masked_fill(target <= threshold, bg_weight)
    return weights


def build_pose_rays(generator, pose):
    """Pre-compute all rays for a fixed pose and keep them on the target device."""
    focal_or_size = (generator.H, generator.W) if generator.orthographic else generator.focal
    rays_full, _, _ = generator.val_ray_sampler(generator.H, generator.W, focal_or_size, pose)
    return rays_full.to(generator.device, non_blocking=True)


def slice_rays(rays_full: torch.Tensor, ray_idx: torch.Tensor) -> torch.Tensor:
    """Select a subset of rays (by linear indices) for a mini-batch."""
    return torch.stack(
        (
            rays_full[0, ray_idx],
            rays_full[1, ray_idx],
        ),
        dim=0,
    )


def render_minibatch(generator, z_latent, rays_subset, need_raw: bool = False, ct_context=None):
    """Render a mini-batch of rays from a fixed pose while keeping training kwargs."""
    render_kwargs = generator.render_kwargs_train if not generator.use_test_kwargs else generator.render_kwargs_test
    render_kwargs = dict(render_kwargs)
    render_kwargs["features"] = z_latent
    if need_raw:
        render_kwargs["retraw"] = True
    if ct_context is not None:
        render_kwargs["ct_context"] = ct_context
    elif render_kwargs.get("use_attenuation"):
        render_kwargs["use_attenuation"] = False
    proj_map, _, _, extras = generator.render(rays=rays_subset, **render_kwargs)
    return proj_map.view(z_latent.shape[0], -1), extras


def maybe_render_preview(step, args, generator, z_eval, outdir, ct_volume=None, act_volume=None, ct_context=None):
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
    ap_np = adjust_projection(ap_np, "ap")
    pa_np = adjust_projection(pa_np, "pa")
    out_dir = outdir / "preview"
    out_dir.mkdir(parents=True, exist_ok=True)
    save_img(ap_np, out_dir / f"step_{step:05d}_AP.png", title=f"AP @ step {step}")
    save_img(pa_np, out_dir / f"step_{step:05d}_PA.png", title=f"PA @ step {step}")
    save_depth_profile(step, generator, z_eval, ct_volume, act_volume, out_dir)
    print("🖼️ Preview gespeichert:", flush=True)
    print("   ", (out_dir / f"step_{step:05d}_AP.png").resolve(), flush=True)
    print("   ", (out_dir / f"step_{step:05d}_PA.png").resolve(), flush=True)


def init_log_file(path: Path):
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
                "mae_ap",
                "mae_pa",
                "psnr_ap",
                "psnr_pa",
                "pred_mean_ap",
                "pred_mean_pa",
                "pred_std_ap",
                "pred_std_pa",
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


def sample_act_points(act: torch.Tensor, nsamples: int, radius: float) -> Tuple[torch.Tensor, torch.Tensor]:
    """Zufällige Voxel (coords, values) aus act.npy ziehen."""
    if act is None:
        raise ValueError("act tensor missing despite act-loss-weight > 0.")
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


def save_depth_profile(step, generator, z_latent, ct_vol, act_vol, outdir: Path):
    if ct_vol is None and act_vol is None:
        return

    def extract_curve(vol):
        if vol is None:
            return None
        data = vol.squeeze(0).detach().cpu().numpy() if vol.dim() == 4 else vol.detach().cpu().numpy()
        if data.ndim != 3:
            return None
        D, H, W = data.shape
        return data[:, H // 2, W // 2], (D, H, W)

    curve_ct, shape_ct = extract_curve(ct_vol) if ct_vol is not None else (None, None)
    curve_act, shape_act = extract_curve(act_vol) if act_vol is not None else (None, None)

    target_shape = shape_ct or shape_act
    if target_shape is None:
        return
    D = target_shape[0]
    radius = generator.radius
    if isinstance(radius, tuple):
        radius = radius[1]

    depth_idx = torch.arange(D, device=generator.device)
    z_coords = idx_to_coord(depth_idx, D, radius)
    zeros = torch.zeros_like(z_coords)
    coords = torch.stack([zeros, zeros, z_coords], dim=1)
    pred = query_emission_at_points(generator, z_latent, coords).detach().cpu().numpy()

    depth_axis = np.linspace(0.0, 1.0, D)
    curves = []
    labels = []
    if curve_ct is not None:
        curves.append(normalize_curve(curve_ct.copy()))
        labels.append("μ (CT)")
    if curve_act is not None:
        curves.append(normalize_curve(curve_act.copy()))
        labels.append("Aktivität (GT)")
    curves.append(normalize_curve(pred.copy()))
    labels.append("Aktivität (NeRF)")

    import matplotlib.pyplot as plt

    plt.figure(figsize=(6, 4))
    for curve, label in zip(curves, labels):
        plt.plot(depth_axis, curve, label=label)
    plt.xlabel("Tiefe (anterior → posterior)")
    plt.ylabel("normierte Intensität")
    plt.ylim(0, 1.05)
    plt.legend()
    plt.tight_layout()
    outdir.mkdir(parents=True, exist_ok=True)
    plt.savefig(outdir / f"depth_profile_step_{step:05d}.png", dpi=150)
    plt.close()


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

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA device required – please launch on a GPU node.")

    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True
    set_seed(args.seed)

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

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
    num_pixels = generator.H * generator.W

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

        if args.normalize_targets:
            ap_flat_proc, ap_min, ap_max, ap_norm_active = normalize_counts(ap_flat, return_params=True)
            pa_flat_proc, pa_min, pa_max, pa_norm_active = normalize_counts(pa_flat, return_params=True)
        else:
            ap_flat_proc = ap_flat
            pa_flat_proc = pa_flat
            ap_min = pa_min = ap_max = pa_max = None
            ap_norm_active = pa_norm_active = False

        z_latent = z_train

        optimizer.zero_grad(set_to_none=True)
        t0 = time.perf_counter()

        idx_ap = torch.randint(0, num_pixels, (rays_per_proj,), device=device)
        idx_pa = torch.randint(0, num_pixels, (rays_per_proj,), device=device)

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
            pred_ap = pred_ap.clamp_min(1e-8)
            pred_pa = pred_pa.clamp_min(1e-8)

            if args.normalize_targets and ap_norm_active:
                scale_ap = (ap_max - ap_min).clamp_min(1e-8)
                pred_ap = (pred_ap - ap_min) / scale_ap
            if args.normalize_targets and pa_norm_active:
                scale_pa = (pa_max - pa_min).clamp_min(1e-8)
                pred_pa = (pred_pa - pa_min) / scale_pa

            weight_ap = build_loss_weights(target_ap, args.bg_weight, args.weight_threshold)
            weight_pa = build_loss_weights(target_pa, args.bg_weight, args.weight_threshold)

            loss_ap = poisson_nll(pred_ap, target_ap, weight=weight_ap)
            loss_pa = poisson_nll(pred_pa, target_pa, weight=weight_pa)
            loss = loss_ap + loss_pa
            if args.debug_attenuation_ray:
                log_attenuation_profile(step, "AP", extras_ap)
                log_attenuation_profile(step, "PA", extras_pa)

            loss_act = torch.tensor(0.0, device=device)
            if args.act_loss_weight > 0.0 and act_vol is not None:
                radius = generator.radius
                if isinstance(radius, tuple):
                    radius = radius[1]
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
                    loss_ct = torch.mean(torch.abs(pred1 - pred2) * weights)
                    loss = loss + args.ct_loss_weight * loss_ct

            loss_reg = torch.tensor(0.0, device=device)
            if args.z_reg_weight > 0.0:
                loss_reg = z_latent.pow(2).mean()
                loss = loss + args.z_reg_weight * loss_reg

        scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(generator.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        torch.cuda.synchronize()
        iter_ms = (time.perf_counter() - t0) * 1000.0

        with torch.no_grad():
            mae_ap = torch.mean(torch.abs(pred_ap - target_ap)).item()
            mae_pa = torch.mean(torch.abs(pred_pa - target_pa)).item()
            pred_mean = (pred_ap.mean().item(), pred_pa.mean().item())
            pred_std = (pred_ap.std().item(), pred_pa.std().item())
            psnr_ap = compute_psnr(pred_ap, target_ap)
            psnr_pa = compute_psnr(pred_pa, target_pa)
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

        print(
            f"[step {step:05d}] loss={loss.item():.6f} | ap={loss_ap.item():.6f} | pa={loss_pa.item():.6f} "
            f"| act={loss_act.item():.6f} | ct={loss_ct.item():.6f} | zreg={loss_reg.item():.6f} "
            f"| mae_ap={mae_ap:.6f} | mae_pa={mae_pa:.6f} "
            f"| psnr_ap={psnr_ap:.2f} | psnr_pa={psnr_pa:.2f} "
            f"| predμ=({pred_mean[0]:.3e},{pred_mean[1]:.3e}) predσ=({pred_std[0]:.3e},{pred_std[1]:.3e})",
            flush=True
        )
        append_log(
            log_path,
            [
                step,
                loss.item(),
                loss_ap.item(),
                loss_pa.item(),
                loss_act.item(),
                loss_ct.item(),
                mae_ap,
                mae_pa,
                psnr_ap,
                psnr_pa,
                pred_mean[0],
                pred_mean[1],
                pred_std[0],
                pred_std[1],
                iter_ms,
                optimizer.param_groups[0]["lr"],
            ],
        )
        if args.save_every > 0 and (step % args.save_every == 0):
            save_checkpoint(step, generator, z_train, optimizer, scaler, ckpt_dir)
        maybe_render_preview(step, args, generator, z_train.detach(), outdir, ct_vol, act_vol, ct_context)

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
