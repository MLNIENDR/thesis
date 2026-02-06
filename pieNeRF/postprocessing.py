#!/usr/bin/env python3
"""Postprocess run results: total-activity bias, voxel metrics, optional organ/projection metrics.

Usage examples:
  python postprocessing.py --run-dir results_spect
  python postprocessing.py --run-dir results_spect --checkpoint results_spect/checkpoints/checkpoint_step00400.pt
  python postprocessing.py --run-dir results_spect --pred-ap-path results_spect/pred_ap.npy --pred-pa-path results_spect/pred_pa.npy
"""
import argparse
import csv
import json
import logging
import math
import subprocess
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import yaml

_LOG = logging.getLogger(__name__)

METRICS_CSV_HEADER = [
    "phantom_id",
    "vol_mae_roi",
    "vol_rmse_roi",
    "activity_bias_roi",
    "activity_rel_abs_error_roi",
    "proj_mae_counts",
    "proj_poisson_dev_counts",
    "organ_rel_error_total_activity",
    "A_gt_roi",
    "A_pred_roi",
    "A_pred_native",
    "timestamp",
    "git_hash",
    "config_path",
    "proj_status",
    "proj_domain",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Postprocess pieNeRF evaluation run")
    parser.add_argument("--run-dir", default="results_spect", help="path to inference run directory")
    parser.add_argument("--split-json", default="results_spect/split.json", help="run split json path")
    parser.add_argument("--manifest", default="data/manifest_abs.csv", help="dataset manifest csv")
    parser.add_argument("--config", default="configs/spect.yaml", help="config yaml (radius/voxel_mm)")
    parser.add_argument("--out-dir", default=None, help="postprocessing output directory (defaults to run-dir/postproc)")
    parser.add_argument("--mask-path-pattern", default="data/{phantom}/out/mask.npy", help="mask path template")
    parser.add_argument(
        "--pred-act-pattern",
        default="activity_pred_final.npy,activity_pred_step_*.npy",
        help="comma separated glob patterns (relative to run dir) to search for pred act npy",
    )
    parser.add_argument("--pred-act-path", help="explicit pred activity file (overrides pattern)")
    parser.add_argument(
        "--pred-act-per-phantom",
        action="store_true",
        help="expect run_dir/{phantom}/activity_pred_*.npy",
    )
    parser.add_argument("--checkpoint", help="checkpoint path (optional) to render AP/PA projections")
    parser.add_argument("--pred-ap-path", help="explicit pred AP projection (.npy)")
    parser.add_argument("--pred-pa-path", help="explicit pred PA projection (.npy)")
    parser.add_argument(
        "--pred-proj-search",
        action="store_true",
        default=True,
        help="automatically search run_dir[/{phantom}] for pred_ap.npy/pred_pa.npy",
    )
    parser.add_argument("--save-proj-npy", action="store_true", help="store rendered pred AP/PA as .npy")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu", help="torch device")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def git_short_hash():
    try:
        proc = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True)
        return proc.stdout.strip()
    except Exception:  # pragma: no cover - best effort
        return "unknown"


def load_yaml(path: Path) -> dict:
    with path.open() as f:
        return yaml.safe_load(f)


def read_manifest(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"manifest missing: {path}")
    result = {}
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pid = row.get("patient_id")
            if not pid:
                continue
            result[pid] = row
    return result


def resolve_manifest_path(entry: dict, candidates: list[str]) -> Path | None:
    for key in candidates:
        val = entry.get(key)
        if not val:
            continue
        path = Path(val)
        if path.exists():
            return path
    return None


def read_split_ids(path: Path) -> list:
    if not path.exists():
        raise FileNotFoundError(f"split json missing: {path}")
    data = json.loads(path.read_text())
    ids = data.get("test_ids")
    if ids is None:
        raise KeyError("split json missing 'test_ids'")
    return list(ids)


def find_pred_activity_paths(
    run_dir: Path,
    test_ids: list[str],
    args,
) -> dict[str, Path]:
    if args.pred_act_path:
        explicit = Path(args.pred_act_path)
        if not explicit.is_absolute():
            explicit = run_dir / explicit
        if not explicit.exists():
            raise FileNotFoundError(explicit)
        return {pid: explicit for pid in test_ids}

    if args.pred_act_per_phantom:
        mapping = {}
        for pid in test_ids:
            candidate_dir = run_dir / pid
            if not candidate_dir.exists():
                raise FileNotFoundError(f"expected folder for phantom {pid} at {candidate_dir}")
            files = sorted(candidate_dir.glob("activity_pred*.npy"))
            if not files:
                raise FileNotFoundError(f"no activity_pred*.npy in {candidate_dir} for {pid}")
            mapping[pid] = sorted_by_step(files)[-1]
        return mapping

    patterns = [p.strip() for p in args.pred_act_pattern.split(",") if p.strip()]
    candidates = []
    for pattern in patterns:
        candidates.extend(sorted(run_dir.glob(pattern)))
    if not candidates:
        raise FileNotFoundError(f"no pred act files found in {run_dir} with {patterns}")
    if len(candidates) == 1:
        return {pid: candidates[0] for pid in test_ids}
    # multiple files: pick final if exists, else highest step
    final = [p for p in candidates if p.name == "activity_pred_final.npy"]
    if final:
        return {pid: final[0] for pid in test_ids}
    highest = sorted_by_step(candidates)[-1]
    if len(test_ids) == 1:
        return {test_ids[0]: highest}
    raise RuntimeError(
        "multiple pred files found but cannot attribute to phantom_id; use --pred-act-path or --pred-act-per-phantom"
    )


def find_projection_candidates(args, run_dir: Path, out_dir: Path, phantom_id: str) -> dict:
    if args.pred_ap_path and args.pred_pa_path:
        return {"ap": Path(args.pred_ap_path), "pa": Path(args.pred_pa_path), "status": "explicit"}
    if not args.pred_proj_search:
        return {}
    candidates = []
    base_candidates = [
        (out_dir / phantom_id / "pred_ap.npy", out_dir / phantom_id / "pred_pa.npy"),
        (run_dir / phantom_id / "pred_ap.npy", run_dir / phantom_id / "pred_pa.npy"),
        (run_dir / "pred_ap.npy", run_dir / "pred_pa.npy"),
        (run_dir / "preview" / "pred_ap.npy", run_dir / "preview" / "pred_pa.npy"),
    ]
    for ap_path, pa_path in base_candidates:
        if ap_path.exists() and pa_path.exists():
            return {"ap": ap_path, "pa": pa_path, "status": "loaded_npy"}
    return {}


def load_projection_arrays(paths: dict[str, Path], shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    ap = np.load(paths["ap"]).astype(np.float32, copy=False)
    pa = np.load(paths["pa"]).astype(np.float32, copy=False)
    if ap.shape != shape or pa.shape != shape:
        raise ValueError("projection shape mismatch with GT")
    return ap, pa


def sorted_by_step(paths: list[Path]) -> list[Path]:
    def step_value(path: Path) -> int:
        parts = path.stem.split("_")
        for part in parts:
            if part.isdigit():
                return int(part)
        return 0

    return sorted(paths, key=step_value)


def load_array(path: Path) -> np.ndarray:
    return np.load(path).astype(np.float32, copy=False)


def spacing_from_meta(act_path: Path, fallback_mm: float) -> tuple[float, float, float]:
    meta_path = act_path.parent / "meta_simple.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        sd_mm = meta.get("sd_mm")
        if sd_mm is not None:
            sd = float(sd_mm)
            sd_cm = sd / 10.0
            return sd_cm, sd_cm, sd_cm
    spacing_cm = fallback_mm / 10.0
    return spacing_cm, spacing_cm, spacing_cm


def compute_rois(shape: tuple[int, int, int], spacing: tuple[float, float, float], radius_xyz_cm: tuple[float, float, float]) -> tuple[tuple[slice, slice, slice], dict, dict, tuple[int, int, int]]:
    slices = []
    idx_ranges = {}
    phys_extent = {}
    shape_out = []
    for axis, size in enumerate(shape):
        center = (size - 1) / 2
        radius_cm = radius_xyz_cm[axis]
        spacing_cm = spacing[axis]
        min_idx = math.ceil(max(0.0, center - radius_cm / spacing_cm))
        max_idx = math.floor(min(size - 1, center + radius_cm / spacing_cm))
        slices.append(slice(min_idx, max_idx + 1))
        idx_ranges[f"axis{axis}"] = {"min": min_idx, "max": max_idx}
        size_roi = max_idx - min_idx + 1
        shape_out.append(size_roi)
        phys_extent[f"axis{axis}"] = [
            index_to_world(min_idx, center, spacing_cm),
            index_to_world(max_idx, center, spacing_cm),
        ]
    return tuple(slices), idx_ranges, phys_extent, tuple(shape_out)


def index_to_world(idx: int, center: float, spacing_cm: float) -> float:
    return (idx - center) * spacing_cm


def resample_pred_to_gt(pred: np.ndarray, target_shape: tuple[int, int, int], device: str) -> np.ndarray:
    tensor = torch.from_numpy(pred.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        resampled = F.interpolate(tensor, size=target_shape, mode="trilinear", align_corners=False)
    return resampled.squeeze(0).squeeze(0).cpu().numpy().astype(np.float32, copy=False)


def compute_voxel_metrics(gt_roi: np.ndarray, pred_roi: np.ndarray) -> tuple[float, float]:
    diff = gt_roi - pred_roi
    mae = float(np.mean(np.abs(diff)))
    rmse = float(np.sqrt(np.mean(diff * diff)))
    return mae, rmse


def load_mask(mask_path: Path, gt_shape: tuple[int, int, int]) -> np.ndarray | None:
    if not mask_path.exists():
        return None
    mask = np.load(mask_path)
    if mask.shape != gt_shape:
        tensor = torch.from_numpy(mask.astype(np.float32)).unsqueeze(0).unsqueeze(0)
        with torch.no_grad():
            resized = F.interpolate(tensor, size=gt_shape, mode="nearest")
        mask = resized.squeeze(0).squeeze(0).cpu().numpy()
    mask_int = np.rint(mask).astype(np.int32)
    return mask_int


def read_organ_names(path: Path) -> dict[int, str]:
    if not path.exists():
        return {}
    mapping = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" in line:
            name, val = line.split("=", 1)
            name = name.strip()
            val = val.strip()
            try:
                mapping[int(val)] = name
            except ValueError:
                continue
    return mapping


def compute_projection_metrics_from_arrays(
    pred_ap: np.ndarray,
    pred_pa: np.ndarray,
    gt_ap: np.ndarray,
    gt_pa: np.ndarray,
    domain: str,
) -> dict[str, float]:
    def mae(target, pred):
        return float(np.mean(np.abs(pred - target)))

    def poisson_dev(target, pred):
        eps = 1e-6
        y = np.clip(target, 0.0, None)
        mu = np.clip(pred, eps, None)
        ratio_term = np.where(y == 0, 0.0, y * np.log((y + eps) / mu))
        term = 2.0 * (ratio_term - (y - mu))
        return float(np.mean(term))

    mae_ap = mae(gt_ap, pred_ap)
    mae_pa = mae(gt_pa, pred_pa)
    results = {"proj_mae_counts": 0.5 * (mae_ap + mae_pa), "proj_domain": domain, "proj_status": ""}
    if domain == "counts":
        dev_ap = poisson_dev(gt_ap, pred_ap)
        dev_pa = poisson_dev(gt_pa, pred_pa)
        results["proj_poisson_dev_counts"] = 0.5 * (dev_ap + dev_pa)
        results["proj_status"] = "loaded_npy"
    else:
        results["proj_poisson_dev_counts"] = float("nan")
        results["proj_status"] = "normalized_inputs"
    return results


def compute_projection_metrics_with_fallback(
    args,
    run_dir: Path,
    out_dir: Path,
    phantom_id: str,
    gt_ap: np.ndarray,
    gt_pa: np.ndarray,
    domain: str,
) -> dict[str, float]:
    if args.checkpoint:
        _LOG.warning("checkpoint render not implemented; projections require latent export")
        return {
            "proj_mae_counts": float("nan"),
            "proj_poisson_dev_counts": float("nan"),
            "proj_status": "render_missing",
            "proj_domain": domain,
        }
    candidates = find_projection_candidates(args, run_dir, out_dir, phantom_id)
    if candidates and "ap" in candidates and "pa" in candidates:
        ap, pa = load_projection_arrays(candidates, gt_ap.shape)
        metrics = compute_projection_metrics_from_arrays(ap, pa, gt_ap, gt_pa, domain)
        metrics["proj_status"] = candidates.get("status", metrics["proj_status"])
        metrics["proj_domain"] = domain
        _LOG.info("loaded pred projections from %s/%s for %s", candidates["ap"], candidates["pa"], phantom_id)
        return metrics
    return {
        "proj_mae_counts": float("nan"),
        "proj_poisson_dev_counts": float("nan"),
        "proj_status": "missing",
        "proj_domain": domain,
    }


def compute_organ_statistics(
    mask: np.ndarray,
    gt_act: np.ndarray,
    pred_act: np.ndarray,
    roi_slice: tuple[slice, slice, slice],
    spacing_gt: tuple[float, float, float],
    organ_map: dict[int, str],
) -> tuple[dict[int, dict], float]:
    roi_mask = np.zeros_like(mask, dtype=bool)
    roi_mask[roi_slice] = True
    organ_ids = sorted(set(np.unique(mask)) - {0})
    if not organ_ids:
        return {}, float("nan")
    vol_voxel = spacing_gt[0] * spacing_gt[1] * spacing_gt[2]
    stats = {}
    rel_errors = []
    for organ_id in organ_ids:
        organ_mask = (mask == organ_id) & roi_mask
        if not organ_mask.any():
            continue
        gt_vals = gt_act[organ_mask]
        pred_vals = pred_act[organ_mask]
        gt_sum = float(gt_vals.sum()) * vol_voxel
        pred_sum = float(pred_vals.sum()) * vol_voxel
        gt_mean = float(np.mean(gt_vals))
        pred_mean = float(np.mean(pred_vals))
        denom = gt_sum if gt_sum != 0 else 1e-6
        rel_error = abs(pred_sum - gt_sum) / denom
        rel_errors.append(rel_error)
        stats[organ_id] = {
            "organ_name": organ_map.get(organ_id, f"organ_{organ_id}"),
            "gt_sum_activity": gt_sum,
            "pred_sum_activity": pred_sum,
            "gt_mean": gt_mean,
            "pred_mean": pred_mean,
            "rel_error_total_activity": rel_error,
        }
    aggregate = float(np.nanmean(rel_errors)) if rel_errors else float("nan")
    return stats, aggregate


def render_projections_stub():
    # Projections currently not rendered (no checkpoint or latent).
    return {
        "proj_mae_counts": float("nan"),
        "proj_poisson_dev_counts": float("nan"),
        "proj_status": "missing",
    }


def write_metrics_json(path: Path, data: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2))


def append_metrics_csv(path: Path, row: dict):
    write_header = not path.exists()
    with path.open("a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(METRICS_CSV_HEADER)
        writer.writerow([row.get(col, "") for col in METRICS_CSV_HEADER])


def plot_organs(organ_stats: dict[int, dict], out_dir: Path, phantom_id: str):
    if not organ_stats:
        return
    organ_ids = list(organ_stats.keys())
    names = [organ_stats[oid]["organ_name"] for oid in organ_ids]
    gt_sums = [organ_stats[oid]["gt_sum_activity"] for oid in organ_ids]
    pred_sums = [organ_stats[oid]["pred_sum_activity"] for oid in organ_ids]
    bar_out = out_dir / f"{phantom_id}_organ_bar.png"
    fig, ax = plt.subplots(figsize=(8, max(4, len(names) * 0.3)))
    width = 0.35
    indices = np.arange(len(names))
    ax.bar(indices - width / 2, gt_sums, width, label="GT")
    ax.bar(indices + width / 2, pred_sums, width, label="Pred")
    ax.set_xticks(indices)
    ax.set_xticklabels(names, rotation=90, fontsize=8)
    ax.set_ylabel("sum activity (kBq/mL * cm^3)")
    ax.set_title(f"Organ sums {phantom_id}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(bar_out, dpi=150)
    plt.close(fig)
    scatter_out = out_dir / f"{phantom_id}_organ_scatter.png"
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(gt_sums, pred_sums, alpha=0.7)
    max_val = max(max(gt_sums), max(pred_sums))
    ax.plot([0, max_val], [0, max_val], "k--", linewidth=1)
    ax.set_xlabel("GT sum activity")
    ax.set_ylabel("Pred sum activity")
    ax.set_title(f"Organ scatter {phantom_id}")
    fig.tight_layout()
    fig.savefig(scatter_out, dpi=150)
    plt.close(fig)


def main():
    args = parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)
    run_dir = Path(args.run_dir)
    out_dir = Path(args.out_dir) if args.out_dir else run_dir / "postproc"
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = read_manifest(Path(args.manifest))
    test_ids = read_split_ids(Path(args.split_json))
    for pid in test_ids:
        if pid not in manifest:
            raise KeyError(f"patient_id {pid} missing in manifest")
    pred_paths = find_pred_activity_paths(run_dir, test_ids, args)
    config = load_yaml(Path(args.config))
    data_cfg = config.get("data", {})
    radius_xyz_cm = config.get("data", {}).get("radius_xyz_cm")
    if radius_xyz_cm is None:
        raise KeyError("config missing data.radius_xyz_cm")
    radius_xyz = tuple(float(r) for r in radius_xyz_cm)
    voxel_mm = float(data_cfg.get("voxel_mm", 1.5))
    timestamp = datetime.utcnow().isoformat()
    git_hash = git_short_hash()
    config_path = str(Path(args.config).resolve())
    aggregated_path = out_dir / "metrics.csv"
    organ_name_map = read_organ_names(Path("data/organ_ids.txt"))

    counts_keys_ap = ["ap_counts_path", "ap_counts", "ap_counts_abs", "ap_counts_path_abs"]
    counts_keys_pa = ["pa_counts_path", "pa_counts", "pa_counts_abs", "pa_counts_path_abs"]
    norm_keys_ap = ["ap_path", "ap", "ap_abs", "ap_path_abs"]
    norm_keys_pa = ["pa_path", "pa", "pa_abs", "pa_path_abs"]
    for pid in test_ids:
        entry = manifest[pid]
        patient_dir = out_dir / pid
        patient_dir.mkdir(parents=True, exist_ok=True)
        act_path = Path(entry["act_path"]) if entry.get("act_path") else None
        if not act_path or not act_path.exists():
            raise FileNotFoundError(f"missing act for {pid}")
        gt_act = load_array(act_path)
        ap_counts_path = resolve_manifest_path(entry, counts_keys_ap)
        pa_counts_path = resolve_manifest_path(entry, counts_keys_pa)
        if ap_counts_path and pa_counts_path:
            proj_domain = "counts"
            gt_ap = load_array(ap_counts_path)
            gt_pa = load_array(pa_counts_path)
        else:
            proj_domain = "normalized"
            ap_path = resolve_manifest_path(entry, norm_keys_ap)
            pa_path = resolve_manifest_path(entry, norm_keys_pa)
            if not ap_path or not pa_path:
                raise FileNotFoundError(f"missing projection paths for {pid}")
            _LOG.warning("counts projections missing for %s, using normalized targets", pid)
            gt_ap = load_array(ap_path)
            gt_pa = load_array(pa_path)
        gt_shape = gt_act.shape
        spacing_gt = spacing_from_meta(act_path, voxel_mm)
        slices, idx_ranges, phys_extent, roi_shape = compute_rois(gt_shape, spacing_gt, radius_xyz)
        pred_path = pred_paths.get(pid)
        if pred_path is None:
            raise RuntimeError(f"no pred path for {pid}")
        pred_act = load_array(pred_path)
        pred_resampled = resample_pred_to_gt(pred_act, gt_shape, args.device)
        roi_slices = slices
        gt_roi = gt_act[roi_slices]
        pred_roi = pred_resampled[roi_slices]
        V_gt = spacing_gt[0] * spacing_gt[1] * spacing_gt[2]
        A_gt_roi = float(gt_roi.sum()) * V_gt
        A_pred_roi = float(pred_roi.sum()) * V_gt
        bias_roi = (A_pred_roi - A_gt_roi) / A_gt_roi if A_gt_roi != 0 else float("nan")
        rel_abs = abs(A_pred_roi - A_gt_roi) / (A_gt_roi if A_gt_roi != 0 else 1e-6)
        vol_mae, vol_rmse = compute_voxel_metrics(gt_roi, pred_roi)
        pred_res = pred_act.shape
        spacing_pred = tuple((2.0 * float(radius_xyz[i])) / (pred_res[i] - 1) for i in range(3))
        V_pred = spacing_pred[0] * spacing_pred[1] * spacing_pred[2]
        A_pred_native = float(pred_act.sum()) * V_pred
        mask_path = Path(args.mask_path_pattern.format(phantom=pid))
        mask = load_mask(mask_path, gt_shape)
        organ_stats = {}
        organ_rel_error = float("nan")
        if mask is not None:
            organ_stats, organ_rel_error = compute_organ_statistics(
                mask,
                gt_act,
                pred_resampled,
                roi_slices,
                spacing_gt,
                organ_name_map,
            )
            table_path = patient_dir / "organ_table.csv"
            with table_path.open("w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "organ_id",
                    "organ_name",
                    "gt_sum_activity",
                    "pred_sum_activity",
                    "gt_mean",
                    "pred_mean",
                    "rel_error_total_activity",
                ])
                for oid, row in organ_stats.items():
                    writer.writerow([
                        oid,
                        row["organ_name"],
                        row["gt_sum_activity"],
                        row["pred_sum_activity"],
                        row["gt_mean"],
                        row["pred_mean"],
                        row["rel_error_total_activity"],
                    ])
            plot_dir = patient_dir / "plots"
            plot_dir.mkdir(exist_ok=True)
            plot_organs(organ_stats, plot_dir, pid)
        proj_metrics = compute_projection_metrics_with_fallback(
            args, run_dir, out_dir, pid, gt_ap, gt_pa, proj_domain
        )
        metrics_data = {
            "phantom_id": pid,
            "run": {
                "git_hash": git_hash,
                "config_path": config_path,
                "args": vars(args),
                "timestamp": timestamp,
            },
            "grid": {
                "gt_shape": gt_shape,
                "pred_shape": pred_act.shape,
                "spacing_gt_cm": spacing_gt,
                "spacing_pred_cm": spacing_pred,
                "roi_slices": idx_ranges,
                "roi_shape": roi_shape,
                "physical_extent_cm": phys_extent,
                "resample": {
                    "direction": "pred_to_gt",
                    "mode": "trilinear",
                    "align_corners": False,
                    "device": args.device,
                },
            },
            "metrics": {
                "vol_mae_roi": vol_mae,
                "vol_rmse_roi": vol_rmse,
                "activity_bias_roi": bias_roi,
                "activity_rel_abs_error_roi": rel_abs,
                "proj_mae_counts": proj_metrics["proj_mae_counts"],
                "proj_poisson_dev_counts": proj_metrics["proj_poisson_dev_counts"],
                "organ_rel_error_total_activity": organ_rel_error,
                "A_gt_roi": A_gt_roi,
                "A_pred_roi": A_pred_roi,
                "A_pred_native": A_pred_native,
                "proj_status": proj_metrics["proj_status"],
                "proj_domain": proj_metrics.get("proj_domain", proj_domain),
            },
            "assumptions": {"roi_assumption": "world box centered on GT grid center"},
        }
        write_metrics_json(patient_dir / "metrics.json", metrics_data)
        append_metrics_csv(
            aggregated_path,
            {
                "phantom_id": pid,
                "vol_mae_roi": vol_mae,
                "vol_rmse_roi": vol_rmse,
                "activity_bias_roi": bias_roi,
                "activity_rel_abs_error_roi": rel_abs,
                "proj_mae_counts": proj_metrics["proj_mae_counts"],
                "proj_poisson_dev_counts": proj_metrics["proj_poisson_dev_counts"],
                "organ_rel_error_total_activity": organ_rel_error,
                "A_gt_roi": A_gt_roi,
                "A_pred_roi": A_pred_roi,
                "A_pred_native": A_pred_native,
                "timestamp": timestamp,
                "git_hash": git_hash,
                "config_path": config_path,
                "proj_status": proj_metrics["proj_status"],
                "proj_domain": proj_metrics.get("proj_domain", proj_domain),
            },
        )
        _LOG.info(
            "processed phantom %s (metrics -> %s; projections status=%s domain=%s)",
            pid,
            patient_dir / "metrics.json",
            proj_metrics["proj_status"],
            proj_metrics["proj_domain"],
        )

if __name__ == "__main__":
    main()
