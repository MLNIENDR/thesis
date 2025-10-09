#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
µ→HU-Kalibrierung + HU-Preview (coronal/axial/sagittal) in einem Skript.

Default-Verzeichnis-Setup (wie bei dir):
  Base:  ~/projects/thesis/TotalSegmentator
  In:    runs_360/ct_recon_rtk360_RAS.nii
  Out:   runs_360/ct_recon_rtk360_HU.nii
  PNG:   runs_360/ct_recon_rtk360_HU_coronal.png  (oder *_preview3.png bei --tri)

Features
- Ankerbasierte lineare Kalibrierung: HU = a*µ + b (≥2 Anker; robust via Median in Kugel-ROI)
- Body-Maske (Schwelle konfigurierbar), Morphologie, größter Komponenten-Filter
- HU-Clip & int16-Speicherung (qform/sform gesetzt)
- Direktes Rendering: echte HU-Farbleiste, mm-Extents, wahlweise 1 Ansicht oder 3 Ansichten

Beispiele:
  python mu_to_hu_and_preview.py
  python mu_to_hu_and_preview.py --body_thresh -300 --preview --view coronal --percentile 2 98 --cmap turbo
  python mu_to_hu_and_preview.py --tri --cmap gray
"""

import os, re, sys, argparse
import numpy as np
import nibabel as nib
from scipy.ndimage import binary_opening, binary_dilation, label, generate_binary_structure

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ------------------------- Helpers (geometry & masks) -------------------------

def mm_to_idx(mm, spacing):
    return int(round(mm / float(spacing)))

def parse_mm(token, total_mm):
    """'x/2', 'y/2', 'z/2', '500', '1/3' -> mm (float)."""
    token = token.strip().lower()
    if token in ("x/2", "y/2", "z/2"):
        axis = token[0]
        return dict(x=total_mm[0]/2, y=total_mm[1]/2, z=total_mm[2]/2)[axis]
    m = re.fullmatch(r'(\d+(?:\.\d+)?)/(\d+(?:\.\d+)?)', token)
    return float(token) if not m else total_mm[0] * (float(m.group(1))/float(m.group(2)))

def sphere_indices(center, radius_vox, shape):
    cx, cy, cz = center
    r = int(max(1, radius_vox))
    x = np.arange(max(0, cx-r), min(shape[0], cx+r+1))
    y = np.arange(max(0, cy-r), min(shape[1], cy+r+1))
    z = np.arange(max(0, cz-r), min(shape[2], cz+r+1))
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
    mask = (X-cx)**2 + (Y-cy)**2 + (Z-cz)**2 <= r**2
    return X[mask], Y[mask], Z[mask]

def largest_component(mask):
    if mask.sum() == 0:
        return mask
    struc = generate_binary_structure(3, 2)
    lbl, n = label(mask, structure=struc)
    if n <= 1:
        return mask
    sizes = np.bincount(lbl.ravel()); sizes[0] = 0
    return lbl == sizes.argmax()


# ------------------------- Calibration (µ -> HU) -------------------------

def calibrate_to_hu(in_path, out_path, anchors, radius_mm=8.0, body_thresh=-400.0,
                    clip=(-1024, 2000)):
    print(f"[INFO] Lade: {in_path}")
    img = nib.load(in_path)
    vol = np.asarray(img.dataobj).astype(np.float32)  # (X,Y,Z), RAS
    aff = img.affine
    dx, dy, dz = float(aff[0,0]), float(aff[1,1]), float(aff[2,2])
    X, Y, Z = vol.shape
    size_mm = (dx*X, dy*Y, dz*Z)

    # --- Anker auswerten ---
    mus, hus = [], []
    for a in anchors:
        coord, hu_t = a.split(":")
        xs, ys, zs = [t.strip() for t in coord.split(",")]
        x_mm = parse_mm(xs, size_mm)
        y_mm = parse_mm(ys, (size_mm[1],)*3)
        z_mm = parse_mm(zs, (size_mm[2],)*3)
        ix, iy, iz = mm_to_idx(x_mm, dx), mm_to_idx(y_mm, dy), mm_to_idx(z_mm, dz)
        r_vox = max(1, int(round(radius_mm / min(dx, dy, dz))))
        x_idx, y_idx, z_idx = sphere_indices((ix, iy, iz), r_vox, vol.shape)
        vals = vol[x_idx, y_idx, z_idx]
        mu_med = float(np.median(vals))
        mus.append(mu_med); hus.append(float(hu_t))
        print(f"[ANCHOR] {a} → idx=({ix},{iy},{iz}) r={r_vox} → µ_median={mu_med:.6f}")

    if len(mus) < 2:
        print("[ERR] Mindestens 2 Anker nötig!", file=sys.stderr)
        sys.exit(1)

    # HU = a*µ + b (Least Squares)
    A = np.stack([mus, np.ones(len(mus), dtype=np.float32)], axis=1)
    x_fit, *_ = np.linalg.lstsq(A, np.array(hus, np.float32), rcond=None)
    a, b = float(x_fit[0]), float(x_fit[1])
    pred = a*np.array(mus) + b
    rmse = float(np.sqrt(np.mean((pred - np.array(hus))**2)))
    print(f"[FIT] HU = {a:.6f} * µ + {b:.2f} | RMSE={rmse:.2f} HU")

    # anwenden
    hu = a*vol + b

    # Body-Maske (schärfer einstellbar)
    mask = hu > float(body_thresh)
    mask = binary_opening(mask, structure=np.ones((3,3,3), bool), iterations=1)
    mask = largest_component(mask)
    mask = binary_dilation(mask, iterations=2)
    hu[~mask] = -1024.0

    # Clip & Speichern (int16)
    hu = np.clip(hu, clip[0], clip[1]).astype(np.int16)
    out_img = nib.Nifti1Image(hu, aff)
    out_img.header.set_qform(aff, code=1)
    out_img.header.set_sform(aff, code=1)
    nib.save(out_img, out_path)
    print(f"[DONE] gespeichert: {out_path}")

    return out_path  # Rückgabe für direkten Preview


# ------------------------- Preview (HU farbig + mm-Extents) -------------------------

def render_slice_png(nifti_path, out_png, view="coronal", base_h_in=7.0,
                     hu_window=(-1000,1000), percentile=None,
                     cmap="turbo", coronal_orient="AP", three_views=False):
    img = nib.load(nifti_path)
    img_ras = nib.as_closest_canonical(img)
    vol = np.asarray(img_ras.dataobj, dtype=np.float32)
    aff = img_ras.affine
    X, Y, Z = vol.shape
    dx, dy, dz = float(aff[0,0]), float(aff[1,1]), float(aff[2,2])

    def get_view(v):
        v = v.lower()
        if v == "coronal":
            sl = vol[X//2,:,:]
            im = np.flipud(sl)
            if coronal_orient.upper() == "AP":
                im = np.fliplr(im)
            extent = [0, dz*Z, 0, dy*Y]
            xlabel, ylabel, ttl = "Z [mm]", "Y [mm]", f"Coronal ({coronal_orient})"
        elif v == "axial":
            sl = vol[:,Y//2,:]
            im = np.flipud(sl.T)
            extent = [0, dx*X, 0, dz*Z]
            xlabel, ylabel, ttl = "X [mm]", "Z [mm]", "Axial"
        elif v == "sagittal":
            sl = vol[:,:,Z//2]
            im = np.flipud(sl.T)
            extent = [0, dx*X, 0, dy*Y]
            xlabel, ylabel, ttl = "X [mm]", "Y [mm]", "Sagittal"
        else:
            raise ValueError("view must be coronal|axial|sagittal")
        return im, extent, xlabel, ylabel, ttl

    def window_limits(arr):
        if percentile:
            p1, p99 = np.percentile(arr, percentile)
            return float(p1), float(p99)
        return hu_window

    if not three_views:
        img2, ex, xlabel, ylabel, ttl = get_view(view)
        width_mm, height_mm = ex[1]-ex[0], ex[3]-ex[2]
        aspect = height_mm / max(width_mm,1e-6)
        h_in = base_h_in; w_in = h_in / max(aspect,1e-6)
        vmin, vmax = window_limits(img2)

        fig, ax = plt.subplots(figsize=(w_in, h_in), dpi=150)
        im = ax.imshow(img2, origin="lower", extent=ex, cmap=cmap,
                       vmin=vmin, vmax=vmax, aspect="equal")
        ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.set_title(ttl)
        cb = fig.colorbar(im, ax=ax, shrink=0.8); cb.set_label("Hounsfield Units [HU]")
        plt.tight_layout()
        os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
        plt.savefig(out_png, dpi=150); plt.close(fig)
        print(f"[OK] Preview saved: {out_png} | window=[{vmin:.0f},{vmax:.0f}]")
        return

    # three-views layout
    views = ["coronal","axial","sagittal"]
    imgs_meta = [get_view(v) for v in views]
    # gleiche Farbrange für alle
    all_vals = np.concatenate([m[0].ravel() for m in imgs_meta])
    if percentile:
        vmin, vmax = np.percentile(all_vals, percentile)
    else:
        vmin, vmax = hu_window
    # Figure-Größe auf größte Aspect anpassen
    h_in = base_h_in
    widths = []
    for _, ex, *_ in imgs_meta:
        width_mm, height_mm = ex[1]-ex[0], ex[3]-ex[2]
        widths.append(h_in / max(height_mm/ max(width_mm,1e-6), 1e-6))
    w_in = sum(widths) + 2.0  # etwas Platz für Colorbar

    fig, axes = plt.subplots(1, 3, figsize=(w_in, h_in), dpi=150)
    for ax, (img2, ex, xlabel, ylabel, ttl) in zip(axes, imgs_meta):
        im = ax.imshow(img2, origin="lower", extent=ex, cmap=cmap,
                       vmin=vmin, vmax=vmax, aspect="equal")
        ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.set_title(ttl)
    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.8)
    cbar.set_label("Hounsfield Units [HU]")
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
    plt.savefig(out_png, dpi=150); plt.close(fig)
    print(f"[OK] Preview (3) saved: {out_png} | window=[{vmin:.0f},{vmax:.0f}]")


# ------------------------- CLI -------------------------

def main():
    base_dir = os.path.expanduser("~/projects/thesis/TotalSegmentator")
    default_in  = os.path.join(base_dir, "runs_360", "ct_recon_rtk360_RAS.nii")
    default_out = os.path.join(base_dir, "runs_360", "ct_recon_rtk360_HU.nii")
    default_png = os.path.join(base_dir, "runs_360", "ct_recon_rtk360_HU_coronal.png")

    ap = argparse.ArgumentParser(description="µ→HU-Kalibrierung + HU-Preview")
    # IO
    ap.add_argument("--in",  dest="infile",  default=default_in,  help="Input NIfTI (µ)")
    ap.add_argument("--out", dest="outfile", default=default_out, help="Output NIfTI (HU,int16)")
    ap.add_argument("--png", dest="pngfile", default=default_png, help="Output PNG (Preview)")
    # anchors
    ap.add_argument("--anchor", action="append",
                    default=["x/2,500,200:+60", "x/2,300,100:-1000"],
                    help='Format "x_mm,y_mm,z_mm:HU" (mehrfach, default: Leber+Luft)')
    ap.add_argument("--radius_mm", type=float, default=8.0, help="ROI-Radius um Anker (mm)")
    ap.add_argument("--clip", nargs=2, type=float, default=[-1024, 2000], help="HU-Clip")
    ap.add_argument("--body_thresh", type=float, default=-300.0,
                    help="Schwellwert für Body-Maske (HU), z.B. -300..-450")
    # preview
    ap.add_argument("--preview", action="store_true", help="nach der Kalibrierung ein PNG rendern")
    ap.add_argument("--view", choices=["coronal","axial","sagittal"], default="coronal")
    ap.add_argument("--percentile", nargs=2, type=float, help="z.B. 2 98 (überschreibt hu_window)")
    ap.add_argument("--hu_window", nargs=2, type=float, default=[-1000,1000])
    ap.add_argument("--cmap", default="turbo")
    ap.add_argument("--base_h_in", type=float, default=7.0)
    ap.add_argument("--tri", action="store_true", help="3-Ansichten-Preview (coronal, axial, sagittal)")
    ap.add_argument("--coronal_orient", choices=["AP","PA"], default="AP")

    args = ap.parse_args()

    # 1) Kalibrieren & speichern
    hu_path = calibrate_to_hu(
        in_path=args.infile,
        out_path=args.outfile,
        anchors=args.anchor,
        radius_mm=args.radius_mm,
        body_thresh=args.body_thresh,
        clip=tuple(args.clip),
    )

    # 2) Optional Preview rendern
    if args.preview or args.tri:
        png = args.pngfile if not args.tri else args.pngfile.replace(".png", "_preview3.png")
        render_slice_png(
            nifti_path=hu_path,
            out_png=png,
            view=args.view,
            base_h_in=args.base_h_in,
            hu_window=tuple(args.hu_window),
            percentile=tuple(args.percentile) if args.percentile else None,
            cmap=args.cmap,
            coronal_orient=args.coronal_orient,
            three_views=bool(args.tri),
        )

if __name__ == "__main__":
    main()