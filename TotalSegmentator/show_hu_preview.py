#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Zeigt einen HU-Slice (coronal/axial/sagittal) aus einer NIfTI-Datei
mit echter Farbskala und mm-Achsen an, ähnlich dem Preview-System des RTK-Skripts.
"""

import os, argparse, nibabel as nib, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def show_hu_slice(nifti_path, out_png, view="coronal", base_h_in=7.0,
                  hu_window=(-1000,1000), percentile=None,
                  cmap="turbo", coronal_orient="AP"):
    img = nib.load(nifti_path)
    img_ras = nib.as_closest_canonical(img)
    vol = np.asarray(img_ras.dataobj, dtype=np.float32)
    aff = img_ras.affine
    X, Y, Z = vol.shape
    dx, dy, dz = float(aff[0,0]), float(aff[1,1]), float(aff[2,2])

    v = view.lower()
    if v == "coronal":
        sl = vol[X//2,:,:]              # (Y,Z)
        img2 = np.flipud(sl)
        if coronal_orient.upper() == "AP":
            img2 = np.fliplr(img2)
        ex = [0, dz*Z, 0, dy*Y]
        xlabel, ylabel = "Z [mm]", "Y [mm]"
        ttl = f"Coronal ({coronal_orient})"

    elif v == "axial":
        sl = vol[:,Y//2,:]
        img2 = np.flipud(sl.T)
        ex = [0, dx*X, 0, dz*Z]
        xlabel, ylabel = "X [mm]", "Z [mm]"
        ttl = "Axial"

    elif v == "sagittal":
        sl = vol[:,:,Z//2]
        img2 = np.flipud(sl.T)
        ex = [0, dx*X, 0, dy*Y]
        xlabel, ylabel = "X [mm]", "Y [mm]"
        ttl = "Sagittal"
    else:
        raise ValueError("view must be coronal|axial|sagittal")

    # Fensterung / Farbskala bestimmen
    if percentile:
        p1, p99 = np.percentile(img2, percentile)
        vmin, vmax = p1, p99
    else:
        vmin, vmax = hu_window

    width_mm, height_mm = ex[1]-ex[0], ex[3]-ex[2]
    aspect = height_mm / max(width_mm,1e-6)
    h_in = base_h_in
    w_in = h_in / max(aspect,1e-6)

    fig, ax = plt.subplots(figsize=(w_in, h_in), dpi=150)
    im = ax.imshow(img2, origin="lower", extent=ex, cmap=cmap,
                   vmin=vmin, vmax=vmax, aspect="equal")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(ttl)
    cb = fig.colorbar(im, ax=ax, shrink=0.8)
    cb.set_label("Hounsfield Units [HU]")
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
    plt.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"[OK] Preview saved: {out_png}  | window=[{vmin:.0f},{vmax:.0f}]  | size≈{w_in:.2f}×{h_in:.2f} in")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", required=True, help="HU-NIfTI")
    ap.add_argument("--out", required=True, help="PNG-Output")
    ap.add_argument("--view", choices=["coronal","axial","sagittal"], default="coronal")
    ap.add_argument("--hu_window", nargs=2, type=float, default=[-1000,1000], help="fixe HU-Fensterung")
    ap.add_argument("--percentile", nargs=2, type=float, help="alternative automatische Fensterung (z. B. 2 98)")
    ap.add_argument("--cmap", default="turbo", help="Colormap (z. B. gray, turbo, bone)")
    ap.add_argument("--base_h_in", type=float, default=7.0)
    args = ap.parse_args()
    show_hu_slice(getattr(args, "in"), args.out, view=args.view,
              hu_window=tuple(args.hu_window),
              percentile=args.percentile,
              cmap=args.cmap,
              base_h_in=args.base_h_in)