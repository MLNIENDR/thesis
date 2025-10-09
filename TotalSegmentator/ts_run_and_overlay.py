#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HU-Range-Check → (optional) TotalSegmentator --ml → robustes Coronal(AP)-Overlay (mm-Extents, Colorbar optional)
- Erwartet HU-CT in RAS (X,Y,Z), int16 HU.
- Findet/erzeugt Multi-Label-Segmentierung.
- Resampled die Segmentation bei Bedarf auf das HU-Gitter (nearest).

Beispiele:
  # nur Overlay einer existierenden Segmentation
  python ts_run_and_overlay.py --in-hu runs_360/ct_recon_rtk360_HU.nii \
      --seg-dir runs_360/ts_output --overlay runs_360/ts_preview/coronal_AP_overlay.png

  # TotalSegmentator ausführen UND overlayn
  python ts_run_and_overlay.py --in-hu runs_360/ct_recon_rtk360_HU.nii \
      --seg-dir runs_360/ts_output --run-ts --overlay runs_360/ts_preview/coronal.png

  # TS mit Zusatzargumenten (z.B. --fast)
  python ts_run_and_overlay.py --in-hu ... --run-ts --ts-args "--fast"
"""

import os, sys, glob, argparse, subprocess
import numpy as np
import nibabel as nib
from nibabel.processing import resample_from_to
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------------- Utils ----------------

def hu_stats(path):
    img = nib.load(path)
    arr = np.asarray(img.dataobj).astype(np.float32)
    f = arr[np.isfinite(arr)]
    qs = np.percentile(f, [1, 50, 99])
    return dict(min=float(f.min()), p1=float(qs[0]), p50=float(qs[1]),
                p99=float(qs[2]), max=float(f.max()), mean=float(f.mean()),
                shape=arr.shape, spacing=(float(img.affine[0,0]), float(img.affine[1,1]), float(img.affine[2,2])))

def hu_range_plausible(stats):
    # recht großzügig, aber sinnvoll: Luft ~ -1000, 99% deutlich >100
    ok_min  = -1200 < stats["min"] < -400
    ok_p50  = -350 < stats["p50"] < 300
    ok_p99  =  100 < stats["p99"] < 3500
    return ok_min and ok_p50 and ok_p99

def find_multilabel(seg_dir):
    seg_dir = os.path.abspath(seg_dir)
    cwd = os.getcwd()
    cand = [
        os.path.join(seg_dir, "segmentation.nii.gz"),
        os.path.join(seg_dir, "segmentation.nii"),
        os.path.join(cwd, "segmentation.nii.gz"),
        os.path.join(cwd, "ts_output.nii"),
        os.path.join(cwd, f"{os.path.basename(seg_dir)}.nii"),
        os.path.join(cwd, f"{os.path.basename(seg_dir)}.nii.gz"),
    ]
    for p in cand:
        if os.path.exists(p):
            print(f"[INFO] Multi-Label gefunden: {p}")
            return p
    # heuristisch im Ordner
    globbed = glob.glob(os.path.join(seg_dir, "*.nii")) + glob.glob(os.path.join(seg_dir, "*.nii.gz"))
    if globbed:
        prefer = [p for p in globbed if any(k in os.path.basename(p).lower() for k in ("seg", "output", "multi"))]
        picks = prefer if prefer else globbed
        picks.sort(key=lambda p: os.path.getsize(p), reverse=True)
        print(f"[INFO] Kandidaten im Ordner gefunden, nehme: {picks[0]}")
        return picks[0]
    raise FileNotFoundError(f"Keine Multi-Label-Datei gefunden in {seg_dir} und CWD.")

def ensure_same_grid(seg_path, ref_img):
    """Resample seg (labels) auf das Grid des ref_img (HU)."""
    seg_img = nib.load(seg_path)
    if seg_img.shape == ref_img.shape and np.allclose(seg_img.affine, ref_img.affine, atol=1e-4):
        return seg_img  # passt
    print("[WARN] Segmentation shape/affine ≠ HU → resample (nearest)")
    rs = resample_from_to(seg_img, ref_img, order=0)  # nearest for labels
    return rs

def save_coronal_overlay_ras(hu_img, seg_img, out_png, base_h_in=8.0, coronal_orient="AP",
                             cmap_name="turbo", alpha=0.35):
    """Coronal(AP) Overlay für RAS (X,Y,Z)."""
    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)

    vol = np.asarray(hu_img.dataobj, dtype=np.float32)   # (X,Y,Z)
    lab = np.asarray(seg_img.dataobj, dtype=np.int16)

    aff = hu_img.affine
    dx, dy, dz = float(aff[0,0]), float(aff[1,1]), float(aff[2,2])
    X, Y, Z = vol.shape

    # Coronal = X fixieren → (Y,Z)
    sl_hu  = vol[X//2, :, :]
    sl_seg = lab[X//2, :, :]

    # Darstellung: Kopf oben, Z horizontal, Y vertikal
    img2 = np.flipud(sl_hu)   # (Y,Z) -> oben=großes Y
    seg2 = np.flipud(sl_seg)

    # AP-Spiegelung optional (horizontal)
    if str(coronal_orient).upper() == "AP":
        img2 = np.fliplr(img2)
        seg2 = np.fliplr(seg2)

    # Grau-Hintergrund fenstern (2..98%)
    p2, p98 = np.percentile(img2[np.isfinite(img2)], [2, 98])
    shown = np.clip((img2 - p2)/(p98 - p2 + 1e-6), 0, 1)

    # Farblayer
    vmax = int(seg2.max()) if np.isfinite(seg2).any() else 0
    cmap = plt.cm.get_cmap(cmap_name, max(vmax,1)+1)
    seg_norm = seg2 / (max(vmax,1)+1.0)
    seg_rgba = cmap(seg_norm)
    seg_rgba[...,3] = (seg2>0) * alpha

    # Extents in mm (Z horizontal, Y vertikal)
    width_mm, height_mm = dz*Z, dy*Y
    extent = [0, width_mm, 0, height_mm]
    aspect = height_mm / max(width_mm, 1e-6)
    h_in = float(base_h_in); w_in = h_in / max(aspect, 1e-6)

    fig, ax = plt.subplots(figsize=(w_in, h_in), dpi=200)
    ax.imshow(shown, cmap="gray", origin="lower", extent=extent, aspect="equal")
    ax.imshow(seg_rgba, origin="lower", extent=extent, aspect="equal")
    ax.set_xlabel("Z [mm]"); ax.set_ylabel("Y [mm]")
    ax.set_title("Coronal (AP) Overlay")
    fig.savefig(out_png, dpi=200, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print(f"[OK] Overlay gespeichert: {out_png}  | extent={width_mm:.1f}×{height_mm:.1f} mm")


# ---------------- Main ----------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-hu", required=True, help="Input HU-CT (RAS, X,Y,Z)")
    ap.add_argument("--seg-dir", default="ts_output", help="TS Output-Ordner")
    ap.add_argument("--overlay", default="ts_preview/coronal_AP_overlay.png")
    ap.add_argument("--base-h-in", type=float, default=8.0)
    ap.add_argument("--force", action="store_true", help="TS/Overlay trotz HU-Warnung")
    ap.add_argument("--run-ts", action="store_true", help="TotalSegmentator ausführen")
    ap.add_argument("--ts-args", type=str, default="", help="zusätzliche TS-Argumente als String, z.B. \"--fast\"")
    ap.add_argument("--cmap", default="turbo")
    ap.add_argument("--alpha", type=float, default=0.35)
    ap.add_argument("--coronal_orient", choices=["AP","PA"], default="AP")
    args = ap.parse_args()

    # 1) HU-Check
    stats = hu_stats(args.in_hu)
    print(f"[INFO] HU stats: shape={stats['shape']}, spacing={stats['spacing']} mm")
    print(f"       min={stats['min']:.1f}, p1={stats['p1']:.1f}, p50={stats['p50']:.1f}, "
          f"p99={stats['p99']:.1f}, max={stats['max']:.1f}, mean={stats['mean']:.1f}")
    if not hu_range_plausible(stats) and not args.force:
        print("[WARN] HU-Range unplausibel. Abbruch (verwende --force zum Überschreiben).")
        sys.exit(2)

    # 2) ggf. TS laufen lassen
    os.makedirs(args.seg_dir, exist_ok=True)
    if args.run_ts:
        cmd = ["TotalSegmentator", "-i", args.in_hu, "-o", args.seg_dir, "--ml"]
        if args.ts_args:
            cmd += args.ts_args.split()
        print("[INFO] Running TotalSegmentator:\n  " + " ".join(cmd))
        subprocess.run(cmd, check=True)
        print(f"[OK] Segmentation finished → {args.seg_dir}")

    # 3) Multi-Label suchen & ggf. resamplen
    seg_path = find_multilabel(args.seg_dir)
    hu_img = nib.load(args.in_hu)
    seg_img = ensure_same_grid(seg_path, hu_img)

    # 4) Overlay speichern
    os.makedirs(os.path.dirname(args.overlay) or ".", exist_ok=True)
    save_coronal_overlay_ras(
        hu_img, seg_img, args.overlay,
        base_h_in=args.base_h_in,
        coronal_orient=args.coronal_orient,
        cmap_name=args.cmap,
        alpha=args.alpha,
    )

if __name__ == "__main__":
    main()