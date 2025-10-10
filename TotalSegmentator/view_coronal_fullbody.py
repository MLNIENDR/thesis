#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Coronale Ansicht (CT) mit optionalen Segment-Konturen.
- Funktioniert auch, wenn KEIN segmentations/-Ordner vorhanden ist (dann nur CT).
- CT in mm (korrektes Seitenverhältnis), Windowing, Auto-Crop & manuelle x/z-Limits.
"""

import os, glob, argparse
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

def hu_window(img, wl_min, wl_max):
    img = np.clip(img, wl_min, wl_max)
    img = (img - wl_min) / float(wl_max - wl_min + 1e-6)
    return img

def find_seg_dir(root):
    for name in ("segmentations","_segmentations","._segmentations",".__segmentations"):
        p = os.path.join(root, name)
        if os.path.isdir(p):
            return p
    return None

def auto_body_crop(ct_slice, dx, dz, margin_mm):
    """Einfache Körper-BBox (Schwelle) und in mm zurückgeben."""
    thr = -500
    mask = ct_slice > thr
    if not np.any(mask):
        return None
    rows = np.where(mask.any(axis=1))[0]  # Z-Richtung (Zeilen)
    cols = np.where(mask.any(axis=0))[0]  # X-Richtung (Spalten)
    z0, z1 = rows.min(), rows.max()
    x0, x1 = cols.min(), cols.max()
    x_min_mm = x0 * dx
    x_max_mm = (x1 + 1) * dx
    z_min_mm = z0 * dz
    z_max_mm = (z1 + 1) * dz
    x_min_mm = max(0.0, x_min_mm - margin_mm)
    z_min_mm = max(0.0, z_min_mm - margin_mm)
    return x_min_mm, x_max_mm + margin_mm, z_min_mm, z_max_mm + margin_mm

def main(args):
    # --- CT-Pfad bestimmen ---
    if args.ct is not None:
        ct_path = args.ct
    else:
        ct_path = os.path.join(args.root, "ct.nii.gz")

    if not os.path.isfile(ct_path):
        raise FileNotFoundError(f"CT nicht gefunden: {ct_path}")

    # Segmentations-Ordner (optional!)
    seg_dir = find_seg_dir(args.root)
    have_segs = seg_dir is not None and len(glob.glob(os.path.join(seg_dir, "*.nii*"))) > 0
    if not have_segs:
        print("[INFO] Keine Segmentierungen gefunden – zeige nur CT.")

    # --- CT laden ---
    ct_img = nib.load(ct_path)
    ct = ct_img.get_fdata()
    dx, dy, dz = ct_img.header.get_zooms()[:3]
    nx, ny, nz = ct.shape

    # Y-Slice wählen (coronal)
    y_idx = ny // 2 if args.slices == "mid" else max(0, min(ny-1, int(args.slices)))

    # Coronaler Slice (Z,X) und Fensterung
    ct_slice = ct[:, y_idx, :].T
    if args.window:
        wl_min, wl_max = args.window
        ct_disp = hu_window(ct_slice, wl_min, wl_max)
    else:
        p1, p99 = np.percentile(ct_slice, [1, 99])
        ct_disp = np.clip((ct_slice - p1)/(p99 - p1 + 1e-6), 0, 1)

    # Physik → mm
    width_mm  = nx * dx
    height_mm = nz * dz
    extent = [0, width_mm, 0, height_mm]

    # Figure-Größe passend zum Seitenverhältnis
    fig_w = args.figwidth
    fig_h = fig_w * (height_mm / max(1e-6, width_mm))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.imshow(ct_disp, cmap="gray", origin="lower", extent=extent)
    ax.set_aspect("equal")
    ax.set_xlabel("X [mm]")
    ax.set_ylabel("Z [mm]")
    ax.set_title(f"Coronal view\nY={y_idx}, voxel={dx:.1f}×{dy:.1f}×{dz:.1f} mm")

    # --- Segmentierungen (nur wenn vorhanden & nicht per --no_segs unterdrückt) ---
    if have_segs and not args.no_segs:
        seg_paths = sorted(glob.glob(os.path.join(seg_dir, "*.nii*")))
        if args.include:
            wanted = set(args.include)
            seg_paths = [p for p in seg_paths if os.path.splitext(os.path.basename(p))[0] in wanted]
        cmap = plt.cm.tab20
        ncol = 20
        for i, sp in enumerate(seg_paths):
            seg = nib.load(sp).get_fdata()
            if seg.shape != ct.shape:
                print(f"⚠️ Shape-Mismatch: {os.path.basename(sp)} → skip")
                continue
            seg_slice = seg[:, y_idx, :].T
            mask = (seg_slice > 0.5).astype(float)
            if not np.any(mask):
                continue
            color = cmap((i % ncol) / max(1, ncol-1))
            ax.contour(mask, levels=[0.5], colors=[color], linewidths=args.lw,
                       alpha=args.alpha, origin="lower", extent=extent, antialiased=True)

    # --- Zoom/Crop ---
    if args.auto_crop:
        bb = auto_body_crop(ct_slice, dx, dz, args.margin_mm)
        if bb is not None:
            x_min_mm, x_max_mm, z_min_mm, z_max_mm = bb
            ax.set_xlim(max(0, x_min_mm), min(width_mm, x_max_mm))
            ax.set_ylim(max(0, z_min_mm), min(height_mm, z_max_mm))
    if args.xlim:
        xmin, xmax = map(float, args.xlim)
        ax.set_xlim(max(0, xmin), min(width_mm, xmax))
    if args.zlim:
        zmin, zmax = map(float, args.zlim)
        ax.set_ylim(max(0, zmin), min(height_mm, zmax))

    plt.tight_layout()
    if args.save:
        plt.savefig(args.save, dpi=args.dpi)
        print(f"[OK] gespeichert: {args.save}")
        plt.close(fig)
    else:
        plt.show()

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".", help="Fall-Ordner (für segmentations/ und default-CT)")
    ap.add_argument("--ct", help="Pfad zur CT-Datei (.nii oder .nii.gz). Wenn gesetzt, überschreibt --root/ct.nii.gz")
    ap.add_argument("--slices", default="mid", help="'mid' oder Y-Index (coronal)")
    ap.add_argument("--window", nargs=2, type=float, default=[-1000, 400],
                    help="HU-Window, z.B. -1000 400; ohne -> Auto (1/99%)")
    ap.add_argument("--include", nargs="*", help="Nur diese Masken zeigen (Basename ohne .nii.gz)")
    ap.add_argument("--no_segs", action="store_true", help="Segment-Overlays komplett deaktivieren")
    ap.add_argument("--alpha", type=float, default=0.8, help="Transparenz der Konturen")
    ap.add_argument("--lw", type=float, default=2.0, help="Linienstärke der Konturen")
    ap.add_argument("--figwidth", type=float, default=8.0, help="Figurbreite in Zoll")
    ap.add_argument("--dpi", type=int, default=250, help="PNG-DPI")
    ap.add_argument("--auto_crop", action="store_true", help="Auto-Crop auf Körper")
    ap.add_argument("--margin_mm", type=float, default=20.0, help="Rand bei Auto-Crop [mm]")
    ap.add_argument("--xlim", nargs=2, help="X-Ausschnitt [mm]: xmin xmax")
    ap.add_argument("--zlim", nargs=2, help="Z-Ausschnitt [mm]: zmin zmax")
    ap.add_argument("--save", default="coronal_ct.png", help="Ausgabedatei (PNG)")
    args = ap.parse_args()
    main(args)