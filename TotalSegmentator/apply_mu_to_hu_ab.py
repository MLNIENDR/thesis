#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Apply linear HU mapping to a mu-volume and save a coronal preview PNG.

Usage example:
python apply_mu_to_hu_ab.py \
  --in runs_360/ct_recon_rtk_mu.nii.gz \
  --out runs_360/ct_recon_rtk_HU.nii.gz \
  --a 1.310979 --b -103.88 \
  --png runs_360/ts_preview/HU_coronal.png
"""

import os, argparse
import numpy as np
import nibabel as nib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def render_coronal_png(nifti_path, out_png, hu_window=(-1000,1000), base_h_in=7.0, coronal_orient="AP"):
    img = nib.load(nifti_path)
    img_ras = nib.as_closest_canonical(img)                 # sichert RAS
    vol = np.asarray(img_ras.dataobj, dtype=np.float32)
    aff = img_ras.affine

    X, Y, Z = vol.shape
    # positive Spacings aus Affine
    dx, dy, dz = map(float, np.abs(np.diag(aff)[:3]))

    # Coronal: X/2-Ebene (mittlerer Sagittalschnitt)
    sl = vol[X//2, :, :]
    # Darstellung: wie zuvor (AP-Option)
    if coronal_orient.upper() == "AP":      # anterior->posterior
        im = np.flipud(np.fliplr(sl))
    else:                                   # PA
        im = np.flipud(sl)

    extent = [0, dz*Z, 0, dy*Y]             # [Z(mm) min,max, Y(mm) min,max]
    xlabel, ylabel, ttl = "Z [mm]", "Y [mm]", f"Coronal ({coronal_orient})"

    # Seitenverhältnis -> saubere Figuregröße
    width_mm, height_mm = extent[1]-extent[0], extent[3]-extent[2]
    aspect = height_mm / max(width_mm, 1e-6)
    h_in = base_h_in
    w_in = h_in / max(aspect, 1e-6)

    vmin, vmax = hu_window
    fig, ax = plt.subplots(figsize=(w_in, h_in), dpi=150)
    imh = ax.imshow(im, origin="lower", extent=extent, cmap="turbo",
                    vmin=vmin, vmax=vmax, aspect="equal")
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.set_title(ttl)
    cb = fig.colorbar(imh, ax=ax, shrink=0.8); cb.set_label("Hounsfield Units [HU]")
    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
    plt.tight_layout(); plt.savefig(out_png, dpi=150); plt.close(fig)
    print(f"[OK] Preview saved: {out_png} | window=[{vmin:.0f},{vmax:.0f}]")

def main():
    ap = argparse.ArgumentParser(description="Apply linear HU mapping (HU = a*mu + b) + coronal preview")
    ap.add_argument("--in",  dest="infile",  required=True, help="Input NIfTI (mu)")
    ap.add_argument("--out", dest="outfile", required=True, help="Output NIfTI (HU)")
    ap.add_argument("--a", type=float, required=True, help="Slope (a) for HU = a*mu + b")
    ap.add_argument("--b", type=float, required=True, help="Offset (b) for HU = a*mu + b")
    ap.add_argument("--clip", nargs=2, type=float, default=[-1024, 2000], help="Clip HU min max")
    ap.add_argument("--png", required=True, help="Output PNG (coronal preview)")
    ap.add_argument("--hu_window", nargs=2, type=float, default=[-1000, 1000], help="Display window for preview")
    ap.add_argument("--base_h_in", type=float, default=7.0, help="Figure height in inches")
    ap.add_argument("--coronal_orient", choices=["AP","PA"], default="AP", help="Coronal orientation label")
    args = ap.parse_args()

    # --- Load mu volume ---
    img = nib.load(args.infile)
    img_ras = nib.as_closest_canonical(img)                      # RAS-Kanon
    mu = np.asarray(img_ras.dataobj, dtype=np.float32)

    # --- Apply linear mapping ---
    a, b = float(args.a), float(args.b)
    print(f"[MAP] HU = {a:.6f} * mu + {b:.2f}")
    hu = a * mu + b

    # --- Clip & save ---
    hu_min, hu_max = args.clip
    hu = np.clip(hu, hu_min, hu_max).astype(np.float32)

    out_img = nib.Nifti1Image(hu, img_ras.affine, img_ras.header)
    # Header sauber halten
    out_img.header.set_qform(img_ras.affine, code=1)
    out_img.header.set_sform(img_ras.affine, code=1)
    nib.save(out_img, args.outfile)
    print(f"[DONE] saved HU NIfTI: {args.outfile}")

    # --- Coronal preview ---
    render_coronal_png(
        nifti_path=args.outfile,
        out_png=args.png,
        hu_window=tuple(args.hu_window),
        base_h_in=args.base_h_in,
        coronal_orient=args.coronal_orient
    )

if __name__ == "__main__":
    main()