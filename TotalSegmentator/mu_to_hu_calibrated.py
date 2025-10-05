#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
µ → HU mit Zwei-Punkt-Kalibrierung (Luft→-1000 HU, Wasser/Weichteil→0 HU)
+ Plausibilitätscheck im Körper
+ korrekte coronale AP-Preview (hochkant 800×500 mm) für Volumen in (X,Z,Y).

WICHTIG:
- Spacings werden aus header.get_zooms() gelesen:
    dx, dz, dy = zooms[0], zooms[1], zooms[2]
- HU werden am Ende hart auf [-1050, 3000] geclippt (dein Rahmen).
"""

import os, argparse, numpy as np, nibabel as nib
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import binary_closing, binary_fill_holes, gaussian_filter

def save_coronal_png_xzy(nii_path, out_png, base_h_in=8.0, title="Coronal (AP, 800×500 mm)"):
    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
    img = nib.load(nii_path)
    vol = np.asarray(img.dataobj, dtype=np.float32)  # (X,Z,Y)
    zooms = img.header.get_zooms()[:3]
    dx, dz, dy = float(zooms[0]), float(zooms[1]), float(zooms[2])  # << fix
    X, Z, Y = vol.shape

    sl = vol[:, :, Y//2]          # (X,Z)
    img2 = np.flipud(sl.T)        # AP, Kopf nach oben

    finite = np.isfinite(img2)
    p2, p98 = np.percentile(img2[finite], [2, 98]) if np.any(finite) else (-1000, 3000)
    shown = np.clip((img2 - p2) / (p98 - p2 + 1e-6), 0, 1)

    width_mm, height_mm = dx*X, dz*Z   # 500 × 800 mm
    aspect = height_mm / max(width_mm, 1e-9)
    extent = [0, width_mm, 0, height_mm]

    h_in = float(base_h_in)
    w_in = h_in / aspect

    fig, ax = plt.subplots(figsize=(w_in, h_in), dpi=200)
    ax.imshow(shown, cmap="gray", origin="lower", extent=extent, aspect="equal")
    ax.set_xlabel("X [mm]"); ax.set_ylabel("Z [mm]")
    ax.set_title(title)
    fig.savefig(out_png, dpi=200, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print(f"[OK] Coronal (AP) gespeichert: {out_png} | extent={width_mm:.1f}×{height_mm:.1f} mm | zooms(dx,dz,dy)=({dx:.4f},{dz:.4f},{dy:.4f})")

def body_mask_from_mu(mu, smooth_sigma=1.0):
    v = mu[np.isfinite(mu)]
    if v.size < 100:
        return np.zeros_like(mu, dtype=bool)
    mu_s = gaussian_filter(mu, smooth_sigma)
    thr = np.percentile(v, 60)  # Luft deutlich darunter
    m = (mu_s > thr)
    m = binary_closing(m, iterations=2)
    m = binary_fill_holes(m)
    return m.astype(bool)

def two_point_calibration(mu, mask=None, air_p=1.0, water_p=(40.0, 60.0)):
    v_all = mu[np.isfinite(mu)]
    if v_all.size < 100:
        raise ValueError("Zu wenige valide Voxels.")
    mu_air = float(np.percentile(v_all, air_p))  # sehr niedrige Perzentile ~ Luft

    if mask is None:
        mask = body_mask_from_mu(mu)
    v_body = mu[mask & np.isfinite(mu)]
    if v_body.size >= 100:
        lo, hi = np.percentile(v_body, [water_p[0], water_p[1]])
        sel = v_body[(v_body >= lo) & (v_body <= hi)]
        mu_soft = float(np.median(sel)) if sel.size else float(np.median(v_body))
    else:
        lo, hi = np.percentile(v_all, [40, 60])
        sel = v_all[(v_all >= lo) & (v_all <= hi)]
        mu_soft = float(np.median(sel)) if sel.size else float(np.median(v_all))
    if not np.isfinite(mu_soft) or mu_soft <= mu_air:
        lo, hi = np.percentile(v_all, [40, 60])
        sel = v_all[(v_all >= lo) & (v_all <= hi)]
        mu_soft = float(np.median(sel)) if sel.size else (mu_air + 1e-3)
    return mu_air, mu_soft

def hu_from_mu(mu, mu_air, mu_water):
    s = 1000.0 / max((mu_water - mu_air), 1e-12)
    b = -1000.0 - s * mu_air
    return s*mu + b, s, b

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i","--in-nii", required=True, help="Eingang µ-Volumen (X,Z,Y), z.B. ct_recon_rtk_RAS.nii")
    ap.add_argument("-o","--out-hu", required=True, help="Ausgang HU-Volumen (float32)")
    ap.add_argument("--coronal-png", default="ts_preview/coronal_AP_HU_calib.png")
    ap.add_argument("--base-h-in", type=float, default=8.0)
    ap.add_argument("--force-mu-water", type=float, default=None, help="Optionaler Fixwert für µ_wasser (überschreibt Auto)")
    ap.add_argument("--force-mu-air",   type=float, default=None, help="Optionaler Fixwert für µ_luft (überschreibt Auto)")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out_hu) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(args.coronal_png) or ".", exist_ok=True)

    img = nib.load(args.in_nii)
    mu  = np.asarray(img.dataobj, dtype=np.float32)

    mask = body_mask_from_mu(mu)
    mu_air_auto, mu_w_auto = two_point_calibration(mu, mask=mask)
    mu_air   = args.force_mu_air   if args.force_mu_air   is not None else mu_air_auto
    mu_water = args.force_mu_water if args.force_mu_water is not None else mu_w_auto

    print(f"[CAL] µ_air   ≈ {mu_air:.6f}")
    print(f"[CAL] µ_water ≈ {mu_water:.6f}")

    hu, s, b = hu_from_mu(mu, mu_air, mu_water)
    # HARTES CLIPPING auf deinen Zielbereich
    hu = np.clip(hu, -1050.0, 3000.0)

    print(f"[CAL] HU = {s:.3e} * µ + ({b:.2f})")
    hb = hu[mask & np.isfinite(hu)]
    if hb.size:
        p = np.percentile(hb, [1,50,99])
        print(f"[INFO] HU (im Körper): p1={p[0]:.1f}, p50={p[1]:.1f}, p99={p[2]:.1f}")
    else:
        print("[WARN] Körpermaske leer – HU-Plausi nicht aussagekräftig.")

    hu_img = nib.Nifti1Image(hu.astype(np.float32), img.header.get_best_affine(), img.header)
    nib.save(hu_img, args.out_hu)
    print(f"[OK] HU-CT gespeichert: {args.out_hu}")

    save_coronal_png_xzy(args.out_hu, args.coronal_png, base_h_in=args.base_h_in)

if __name__ == "__main__":
    main()