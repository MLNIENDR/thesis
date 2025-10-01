#!/usr/bin/env python3
import argparse, json, os
import numpy as np
from scipy.ndimage import gaussian_filter
import nibabel as nib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------- helpers ----------
def fwhm_to_sigma(fwhm_mm: float) -> float:
    return float(fwhm_mm) / 2.355

def psf_blur_mu(mu_xyz: np.ndarray, voxel_xyz_mm, psf_fwhm_mm: float) -> np.ndarray:
    if psf_fwhm_mm <= 0:
        return mu_xyz.astype(np.float32, copy=True)
    sx = fwhm_to_sigma(psf_fwhm_mm) / float(voxel_xyz_mm[0])
    sy = fwhm_to_sigma(psf_fwhm_mm) / float(voxel_xyz_mm[1])
    sz = fwhm_to_sigma(psf_fwhm_mm) / float(voxel_xyz_mm[2])
    return gaussian_filter(mu_xyz.astype(np.float32), sigma=(sx,sy,sz), mode="nearest")

def add_bias_and_noise_hu(hu_xyz: np.ndarray, voxel_xyz_mm,
                          bias_amp_hu=0.0, bias_fwhm_mm=150.0,
                          noise_hu=20.0, seed=1234) -> np.ndarray:
    out = hu_xyz.astype(np.float32, copy=True)
    rng = np.random.default_rng(seed)
    if bias_amp_hu > 0:
        low = rng.normal(0,1,size=out.shape).astype(np.float32)
        sx = fwhm_to_sigma(bias_fwhm_mm) / float(voxel_xyz_mm[0])
        sy = fwhm_to_sigma(bias_fwhm_mm) / float(voxel_xyz_mm[1])
        sz = fwhm_to_sigma(bias_fwhm_mm) / float(voxel_xyz_mm[2])
        low = gaussian_filter(low, sigma=(sx,sy,sz), mode="nearest")
        p1,p99 = np.percentile(low,[1,99]); low = np.clip((low-p1)/(p99-p1+1e-6),0,1)
        out += (low*2-1)*float(bias_amp_hu)
    if noise_hu > 0:
        out += rng.normal(0.0, float(noise_hu), size=out.shape).astype(np.float32)
    return out

def add_lung_texture(hu_xyz: np.ndarray, lung_mask: np.ndarray, voxel_xyz_mm,
                     target_mean_hu=-850.0, bp_lo_mm=1.0, bp_hi_mm=6.0,
                     texture_sd_hu=60.0, seed=42) -> np.ndarray:
    """Bandpass-Textur nur im Lungenparenchym."""
    out = hu_xyz.astype(np.float32, copy=True)
    if not np.any(lung_mask):
        return out
    rng = np.random.default_rng(seed)
    noise = rng.normal(0,1,size=out.shape).astype(np.float32)

    sig_lo = [bp_lo_mm / float(v) for v in voxel_xyz_mm]
    sig_hi = [bp_hi_mm / float(v) for v in voxel_xyz_mm]
    fine = gaussian_filter(noise, sigma=sig_lo, mode="nearest")
    coarse = gaussian_filter(noise, sigma=sig_hi, mode="nearest")
    band = fine - coarse

    std = np.std(band[lung_mask]);  std = 1.0 if std < 1e-6 else std
    band = band / std * float(texture_sd_hu)

    current_mean = float(np.mean(out[lung_mask]))
    shift = float(target_mean_hu) - current_mean
    out[lung_mask] = out[lung_mask] + band[lung_mask] + shift
    return out

def save_nifti_int16_xyz(hu_xyz: np.ndarray, voxel_xyz_mm, out_path: str):
    affine = np.diag([float(voxel_xyz_mm[0]), float(voxel_xyz_mm[1]),
                      float(voxel_xyz_mm[2]), 1.0]).astype(np.float32)
    img = nib.Nifti1Image(np.round(hu_xyz).astype(np.int16), affine)
    img.set_qform(affine); img.set_sform(affine)
    nib.save(img, out_path)

def qc_like_visualize(vol_xyz: np.ndarray, axis: int, idx: int, window, title: str, out_png: str):
    """Anzeige exakt wie dein visualize_totalseg.py: rot90 + aspect=1."""
    if idx < 0:
        idx = (vol_xyz.shape[axis]//2)
    if axis == 0:   sl = vol_xyz[idx, :, :]
    elif axis == 1: sl = vol_xyz[:, idx, :]
    else:           sl = vol_xyz[:, :, idx]
    sl = np.rot90(sl)
    vmin, vmax = window
    plt.figure(figsize=(6,10))
    plt.imshow(sl, cmap="gray", vmin=vmin, vmax=vmax, aspect=1, interpolation="bilinear")
    plt.axis("off"); plt.title(title)
    plt.savefig(out_png, dpi=300, bbox_inches="tight"); plt.close()

def hist_png(arr, fname, rng=(-1200,2000), title="Histogram"):
    plt.figure(figsize=(8,4))
    plt.hist(arr.ravel(), bins=500, range=rng, log=True)
    plt.title(title); plt.xlabel("HU"); plt.ylabel("Voxels (log)")
    plt.tight_layout(); plt.savefig(fname, dpi=250); plt.close()

def print_stats(name, a):
    a = a.astype(np.float32)
    p = np.percentile(a, [1,5,25,50,75,95,99])
    print(f"[{name}] min={a.min():.1f} max={a.max():.1f} mean={a.mean():.1f} std={a.std():.1f} "
          f"p1={p[0]:.1f} p5={p[1]:.1f} p50={p[3]:.1f} p95={p[5]:.1f} p99={p[6]:.1f}")

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser(description="ATN(.bin, μ in 1/mm) → CT-ähnliches HU-NIfTI (int16) + QC + Lungen-Textur.")
    ap.add_argument("-i","--input", required=True, help=".bin/.atn (float32)")
    ap.add_argument("-o","--output", required=True, help="Ausgabe NIfTI (.nii.gz)")
    ap.add_argument("--shape", nargs=3, type=int, required=True, metavar=("X","Y","Z"),
                    help="Shape (X Y Z), Fortran-Order.")
    ap.add_argument("--voxel", nargs=3, type=float, default=[1.5,1.5,1.5], metavar=("dX","dY","dZ"),
                    help="Voxel (mm) X Y Z.")
    ap.add_argument("--mu_water_mm", type=float, default=0.0195, help="μ(Wasser) [1/mm].")
    ap.add_argument("--psf_fwhm_mm", type=float, default=1.2, help="PSF in μ (FWHM mm).")
    ap.add_argument("--bias_amp_hu", type=float, default=0.0, help="Bias-Amplitude (HU).")
    ap.add_argument("--bias_fwhm_mm", type=float, default=150.0, help="Bias-FWHM (mm).")
    ap.add_argument("--noise_hu", type=float, default=15.0, help="σ Rauschen (HU).")
    ap.add_argument("--clip", nargs=2, type=int, default=[-1000, 1500], metavar=("HU_MIN","HU_MAX"))
    ap.add_argument("--body_threshold_mu", type=float, default=0.003, help="Körpermaske in μ; 0=aus.")
    # Lunge
    ap.add_argument("--lung_texture", action="store_true", help="Bandpass-Textur in der Lunge hinzufügen.")
    ap.add_argument("--lung_mean", type=float, default=-850.0, help="Zielmittelwert HU in Lunge.")
    ap.add_argument("--lung_sd", type=float, default=60.0, help="Textur-SD in Lunge (HU).")
    ap.add_argument("--lung_bp_lo", type=float, default=1.0, help="Bandpass low (mm).")
    ap.add_argument("--lung_bp_hi", type=float, default=6.0, help="Bandpass high (mm).")
    # QC
    ap.add_argument("--qc", action="store_true", help="QC-PNGs (vor/nach) erzeugen.")
    ap.add_argument("--qc_axis", type=int, default=1, help="0=axial, 1=koronal, 2=sagittal (wie dein Script).")
    ap.add_argument("--qc_slice", type=int, default=-1, help="Sliceindex; -1=Mitte.")
    ap.add_argument("--qc_window", nargs=2, type=int, default=None, help="Fenster für QC (min max).")
    ap.add_argument("--qc_autowindow", action="store_true",
                    help="QC: Fenster automatisch aus p5–p95 (auf HU_after) bestimmen.")
    args = ap.parse_args()

    X,Y,Z = args.shape
    voxel_xyz_mm = [args.voxel[0], args.voxel[1], args.voxel[2]]

    # 1) Rohdaten laden (Fortran-Order) → (X,Y,Z)
    raw = np.fromfile(args.input, dtype=np.float32)
    expected = X*Y*Z
    if raw.size != expected:
        raise ValueError(f"Elementanzahl passt nicht: {raw.size} vs {expected} für shape {args.shape}")
    mu_xyz = raw.reshape((X,Y,Z), order="F").astype(np.float32)
    print(f"μ: shape={mu_xyz.shape}, min={mu_xyz.min():.6f}, max={mu_xyz.max():.6f}, mean={mu_xyz.mean():.6f}")

    # 2) Körpermaske
    body = (mu_xyz > float(args.body_threshold_mu)).astype(np.float32) if args.body_threshold_mu>0 \
           else np.ones_like(mu_xyz, dtype=np.float32)

    # 3) μ→HU (before)
    muw = float(args.mu_water_mm)
    hu_before = 1000.0 * (mu_xyz - muw) / (muw + 1e-8)
    hu_before = np.where(body>0.5, hu_before, -1000.0)
    hu_before = np.clip(hu_before, args.clip[0], args.clip[1]).astype(np.float32)

    # 4) μ-PSF + HU + globales Noise/Bias (after)
    mu_blur = psf_blur_mu(mu_xyz, voxel_xyz_mm, psf_fwhm_mm=float(args.psf_fwhm_mm))
    hu_after = 1000.0 * (mu_blur - muw) / (muw + 1e-8)
    hu_after = np.where(body>0.5, hu_after, -1000.0)
    hu_after = add_bias_and_noise_hu(
        hu_after, voxel_xyz_mm,
        bias_amp_hu=float(args.bias_amp_hu),
        bias_fwhm_mm=float(args.bias_fwhm_mm),
        noise_hu=float(args.noise_hu),
        seed=1234
    )

    # 5) Lungen-Textur optional
    if args.lung_texture:
        lung_mask = hu_before < -650.0   # grobe Maske vor der Anpassung
        hu_after = add_lung_texture(
            hu_after, lung_mask, voxel_xyz_mm,
            target_mean_hu=float(args.lung_mean),
            bp_lo_mm=float(args.lung_bp_lo),
            bp_hi_mm=float(args.lung_bp_hi),
            texture_sd_hu=float(args.lung_sd),
            seed=4242
        )

    # 6) Clipping (final)
    hu_after = np.clip(hu_after, args.clip[0], args.clip[1]).astype(np.float32)

    # 7) Diagnose-Stats + Histogramme
    def hist_png(arr, fname, rng=(-1200,2000), title="Histogram"):
        plt.figure(figsize=(8,4))
        plt.hist(arr.ravel(), bins=500, range=rng, log=True)
        plt.title(title); plt.xlabel("HU"); plt.ylabel("Voxels (log)")
        plt.tight_layout(); plt.savefig(fname, dpi=250); plt.close()

    def print_stats(name, a):
        a = a.astype(np.float32)
        p = np.percentile(a, [1,5,25,50,75,95,99])
        print(f"[{name}] min={a.min():.1f} max={a.max():.1f} mean={a.mean():.1f} std={a.std():.1f} "
              f"p1={p[0]:.1f} p5={p[1]:.1f} p50={p[3]:.1f} p95={p[5]:.1f} p99={p[6]:.1f}")

    print_stats("HU_before", hu_before)
    print_stats("HU_after ", hu_after)
    base = os.path.splitext(os.path.basename(args.output))[0]
    hist_png(hu_before, f"{base}_hist_before.png", title="HU Histogram (before)")
    hist_png(hu_after,  f"{base}_hist_after.png",  title="HU Histogram (after)")

    # 8) QC-Fenster bestimmen
    if args.qc_autowindow:
        lo, hi = np.percentile(hu_after, [5,95])
        qc_win = (float(lo), float(hi))
    else:
        qc_win = tuple(args.qc_window) if args.qc_window is not None else (-200.0, 300.0)

    # 9) QC-Bilder
    if args.qc:
        qc_like_visualize(hu_before, args.qc_axis, args.qc_slice, qc_win,
                          f"QC vor Anpassung (axis={args.qc_axis})", f"{base}_qc_before.png")
        qc_like_visualize(hu_after,  args.qc_axis, args.qc_slice, qc_win,
                          f"QC nach Anpassung (axis={args.qc_axis})", f"{base}_qc_after.png")

    # 10) NIfTI speichern (int16)
    save_nifti_int16_xyz(hu_after, voxel_xyz_mm, args.output)

    # 11) Log
    stats = dict(
        shape_in=[int(X),int(Y),int(Z)],
        voxel_in_mm=[float(v) for v in voxel_xyz_mm],
        mu_water_mm=float(muw),
        psf_fwhm_mm=float(args.psf_fwhm_mm),
        noise_hu=float(args.noise_hu),
        bias_amp_hu=float(args.bias_amp_hu),
        clip_hu=[int(args.clip[0]), int(args.clip[1])],
        outputs=dict(
            nifti=args.output,
            hist_before=f"{base}_hist_before.png",
            hist_after=f"{base}_hist_after.png",
            qc_before=(f"{base}_qc_before.png" if args.qc else None),
            qc_after=(f"{base}_qc_after.png" if args.qc else None),
        )
    )
    print(json.dumps(stats, indent=2))
    print("Gespeichert:", args.output)

if __name__ == "__main__":
    main()