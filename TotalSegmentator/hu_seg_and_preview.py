#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HU-Range-Check → TotalSegmentator --ml → korrekte coronale Overlay-Ansicht (AP, 800×500 mm, X/Z/Y)
Sucht robust nach der Multi-Label-Datei:
  1) <seg-dir>/segmentation.nii.gz
  2) ./segmentation.nii.gz
  3) ./ts_output.nii
  4) ./<seg-dir>.nii
  5) ./<seg-dir>.nii.gz
  6) <seg-dir>/*.nii / *.nii.gz (heuristisch: enthält 'seg' oder 'output' oder ist größte Datei)
"""

import os, argparse, subprocess, numpy as np, nibabel as nib, glob
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------- Overlay ----------
def save_coronal_overlay_xzy(hu_path, seg_path, out_png, base_h_in=8.0):
    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
    img = nib.load(hu_path)
    seg = nib.load(seg_path)
    vol = np.asarray(img.dataobj, dtype=np.float32)   # (X,Z,Y)
    lab = np.asarray(seg.dataobj, dtype=np.int16)
    aff = img.affine
    dx, dz, dy = float(aff[0,0]), float(aff[1,1]), float(aff[2,2])
    X, Z, Y = vol.shape

    sl_hu  = vol[:, :, Y//2]      # (X,Z)
    sl_seg = lab[:, :, Y//2]
    img2 = np.flipud(sl_hu.T)     # AP, Kopf oben
    seg2 = np.flipud(sl_seg.T)

    p2, p98 = np.percentile(img2[np.isfinite(img2)], [2, 98])
    shown = np.clip((img2 - p2)/(p98 - p2 + 1e-6), 0, 1)

    vmax = int(np.nanmax(seg2)) if np.isfinite(np.nanmax(seg2)) else 0
    cmap = plt.cm.get_cmap("turbo", max(vmax,1)+1)
    seg_norm = seg2 / (max(vmax,1)+1.0)
    seg_rgba = cmap(seg_norm)
    seg_rgba[...,3] = (seg2>0)*0.35

    width_mm, height_mm = dx*X, dz*Z   # 500 × 800
    aspect = height_mm / width_mm
    extent = [0, width_mm, 0, height_mm]

    h_in = float(base_h_in); w_in = h_in / max(aspect,1e-6)
    fig, ax = plt.subplots(figsize=(w_in, h_in), dpi=200)
    ax.imshow(shown, cmap="gray", origin="lower", extent=extent, aspect="equal")
    ax.imshow(seg_rgba, origin="lower", extent=extent, aspect="equal")
    ax.set_xlabel("X [mm]"); ax.set_ylabel("Z [mm]")
    ax.set_title("Coronal (AP) Overlay")
    fig.savefig(out_png, dpi=200, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print(f"[OK] Overlay gespeichert: {out_png}  | extent={width_mm:.1f}×{height_mm:.1f} mm")

# ---------- Suche Multi-Label ----------
def find_multilabel(seg_dir: str) -> str:
    seg_dir = os.path.abspath(seg_dir)
    cwd = os.getcwd()
    cand = [
        os.path.join(seg_dir, "segmentation.nii.gz"),
        os.path.join(cwd, "segmentation.nii.gz"),
        os.path.join(cwd, "ts_output.nii"),
        os.path.join(cwd, f"{os.path.basename(seg_dir)}.nii"),
        os.path.join(cwd, f"{os.path.basename(seg_dir)}.nii.gz"),
    ]
    for p in cand:
        if os.path.exists(p):
            print(f"[INFO] Multi-Label gefunden: {p}")
            return p

    # Heuristik: suche im seg_dir sinnvolle .nii(.gz)
    globbed = glob.glob(os.path.join(seg_dir, "*.nii")) + glob.glob(os.path.join(seg_dir, "*.nii.gz"))
    if globbed:
        # Bevorzuge Dateien mit 'seg' oder 'output' im Namen, sonst größte Datei
        prefer = [p for p in globbed if any(k in os.path.basename(p).lower() for k in ("seg", "output", "multi"))]
        picks = prefer if prefer else globbed
        picks.sort(key=lambda p: os.path.getsize(p), reverse=True)
        print(f"[INFO] Kandidaten im Ordner gefunden, nehme: {picks[0]}")
        return picks[0]

    raise FileNotFoundError("Keine Multi-Label-Datei gefunden.\nGeprüft:\n  " + "\n  ".join(cand) + f"\n  + scan in {seg_dir}/*.nii*")

# ---------- Main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-hu", required=True, help="Input HU-CT (X,Z,Y, z. B. ct_recon_rtk_RAS_HU_calib.nii.gz)")
    ap.add_argument("--seg-dir", default="ts_output", help="Output-Ordner für TotalSegmentator")
    ap.add_argument("--coronal-out", default="ts_preview/coronal_AP_overlay.png")
    ap.add_argument("--base-h-in", type=float, default=8.0)
    ap.add_argument("--force", action="store_true", help="TS trotzdem laufen lassen, auch wenn HU-Stats warnen")
    args = ap.parse_args()

    # HU-Stats
    img = nib.load(args.in_hu)
    vol = np.asarray(img.dataobj, dtype=np.float32)
    finite = vol[np.isfinite(vol)]
    p1, p50, p99 = np.percentile(finite, [1, 50, 99])
    vmin, vmax, vmean = float(np.min(finite)), float(np.max(finite)), float(np.mean(finite))
    print(f"[INFO] HU stats: min={vmin:.1f}, p1={p1:.1f}, p50={p50:.1f}, p99={p99:.1f}, max={vmax:.1f}, mean={vmean:.1f}")

    ok = (-1100 < vmin < -500) and (100 < p99 < 3500) and (-300 < p50 < 300)
    if not ok:
        print("[WARN] HU-Range wirkt unplausibel. Wenn gewollt, mit --force fortfahren.")
        if not args.force:
            raise ValueError("HU-Range unplausibel! Abbruch ohne --force.")

    # TotalSegmentator ausführen
    os.makedirs(args.seg_dir, exist_ok=True)
    cmd = ["TotalSegmentator", "-i", args.in_hu, "-o", args.seg_dir, "--ml"]
    print("[INFO] Running TotalSegmentator:\n  " + " ".join(cmd))
    subprocess.run(cmd, check=True)
    print(f"[OK] Segmentation finished → {args.seg_dir}")

    # Multi-Label finden (robuste Suche)
    seg_path = find_multilabel(args.seg_dir)

    # Overlay
    save_coronal_overlay_xzy(args.in_hu, seg_path, args.coronal_out, base_h_in=args.base_h_in)

if __name__ == "__main__":
    main()