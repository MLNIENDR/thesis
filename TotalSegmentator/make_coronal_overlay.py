#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, glob, argparse, numpy as np, nibabel as nib
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

def pct_window(im, p=(1,99)):
    s = im[np.isfinite(im)]
    if s.size == 0: return -1024.0, 1500.0
    lo, hi = np.percentile(s, p)
    if lo == hi: hi = lo + np.finfo(np.float32).eps
    return float(lo), float(hi)

def reorder_to_XYZ(vol, spc, order="xyz"):
    idx = {'x':0,'y':1,'z':2}
    perm = [idx[c] for c in order.lower()]
    return np.transpose(vol, perm), tuple(spc[i] for i in perm)

def extent_from_wh(w,h):  # mm
    return [-w/2, w/2, -h/2, h/2]

def load_multilabel_from_dir(seg_dir):
    # 1) Multi-Label-Datei bevorzugt
    for cand in ["segmentations.nii.gz","segmentations.nii","labels.nii.gz","labels.nii"]:
        p = os.path.join(seg_dir, cand)
        if os.path.isfile(p):
            print(f"[INFO] Multi-Label gefunden: {p}")
            return nib.load(p)
    # 2) Einzelmasken
    paths = sorted([p for p in glob.glob(os.path.join(seg_dir, "*.nii*"))
                    if os.path.basename(p).lower() not in
                    ("segmentations.nii.gz","segmentations.nii","labels.nii.gz","labels.nii")])
    if not paths:
        raise RuntimeError(f"Keine Label-NIfTIs in {seg_dir} gefunden.")
    return paths

def build_multilabel_from_masks(mask_paths, ref_img):
    lab = np.zeros(ref_img.shape, dtype=np.int32)
    cur = 1
    for p in mask_paths:
        m = nib.load(p).get_fdata().astype(np.uint8)
        if m.shape != lab.shape:
            raise RuntimeError(f"Shape mismatch: {p} hat {m.shape}, erwartet {lab.shape}")
        upd = (m>0) & (lab==0)
        lab[upd] = cur
        cur += 1
    return nib.Nifti1Image(lab, ref_img.affine)

def color_map(nlab):
    cmap = plt.cm.get_cmap('tab20', max(nlab,20))
    cols = [cmap(i)[:3] for i in range(nlab+1)]
    cols[0] = (0,0,0)
    return cols

def main():
    ap = argparse.ArgumentParser(description="Coronal-AP Overlay (1 PNG)")
    ap.add_argument("--ct", required=True, help="Eingabe-CT (HU, NIfTI)")
    ap.add_argument("--seg", required=True, help="TotalSegmentator-Output-Ordner ODER Multi-Label-Datei")
    ap.add_argument("--out", default="coronal_AP_overlay.png")
    ap.add_argument("--alpha", type=float, default=0.35)
    ap.add_argument("--base_h_in", type=float, default=6.0)
    ap.add_argument("--order", default="xyz")
    # >>> diese Flags brauchst du <<<
    ap.add_argument("--use_swap", action="store_true",
                    help="Axial als Quelle benutzen (entspricht deinem 'guten' Look)")
    ap.add_argument("--flip180", action="store_true",
                    help="2D-Bild um 180° drehen (wie bei dir benötigt)")
    ap.add_argument("--view", choices=["ap","pa"], default="ap",
                    help="AP: Links/Rechts-Flip anwenden")
    ap.add_argument("--win_low", type=float, default=None)
    ap.add_argument("--win_high", type=float, default=None)
    args = ap.parse_args()

    # --- CT laden & nach (X,Y,Z) ---
    ct_img = nib.load(args.ct)
    ct_can = nib.as_closest_canonical(ct_img)
    vol0   = ct_can.get_fdata().astype(np.float32)
    spc0   = ct_can.header.get_zooms()[:3]
    vol, (dx,dy,dz) = reorder_to_XYZ(vol0, spc0, args.order)
    nx,ny,nz = vol.shape
    X, Y, Z  = dx*nx, dy*ny, dz*nz
    print(f"[INFO] CT shape(X,Y,Z)={vol.shape}  vox=({dx:.4f},{dy:.4f},{dz:.4f})  FOV(mm)=({X:.1f},{Y:.1f},{Z:.1f})")

    # --- Labels laden ---
    if os.path.isdir(args.seg):
        ml = load_multilabel_from_dir(args.seg)
        lbl_img = build_multilabel_from_masks(ml, ct_can) if isinstance(ml, list) else ml
    else:
        print(f"[INFO] Multi-Label Datei: {args.seg}")
        lbl_img = nib.load(args.seg)

    lbl_can = nib.as_closest_canonical(lbl_img)
    lbl     = lbl_can.get_fdata().astype(np.int32)
    if lbl.shape != vol.shape:
        raise RuntimeError(f"Shape mismatch: labels {lbl.shape} vs CT {vol.shape}")

    # --- Slice wählen ---
    zc, yc = nz//2, ny//2
    src_ax  = vol[:, :, zc].T     # XY -> (Y,X)
    src_cor = vol[:, yc, :].T     # XZ -> (Z,X)
    lab_ax  = lbl[:, :, zc].T
    lab_cor = lbl[:, yc, :].T

    # Dein bewährter Modus: axial als Quelle + 180° + AP
    img2d = src_ax if args.use_swap else src_cor
    lab2d = lab_ax if args.use_swap else lab_cor

    if args.flip180:
        img2d = np.flipud(np.fliplr(img2d))
        lab2d = np.flipud(np.fliplr(lab2d))
    if args.view == "ap":
        img2d = np.fliplr(img2d)
        lab2d = np.fliplr(lab2d)

    # Extent (Coronal: Breite=X, Höhe=Z)
    w_mm, h_mm = (X, Z)
    ext = extent_from_wh(w_mm, h_mm)

    # Plot
    H = args.base_h_in
    W = H * (w_mm / h_mm)
    fig = plt.figure(figsize=(W, H), constrained_layout=True)
    ax  = fig.add_subplot(111)

    lo, hi = pct_window(img2d, p=(1,99)) if (args.win_low is None or args.win_high is None) else (args.win_low, args.win_high)
    ax.imshow(img2d, cmap="gray", origin="lower", vmin=lo, vmax=hi, extent=ext, interpolation="nearest")

    nlab = int(lab2d.max())
    if nlab > 0:
        cols = color_map(nlab)
        rgb = np.zeros(lab2d.shape+(3,), dtype=np.float32)
        for i in range(1, nlab+1):
            m = (lab2d==i)
            if m.any():
                rgb[m] = cols[i] if i < len(cols) else cols[i%len(cols)]
        ax.imshow(rgb, origin="lower", extent=ext, alpha=args.alpha, interpolation="nearest")

    ax.set_xlabel("mm"); ax.set_ylabel("mm"); ax.set_title(f"Coronal ({args.view.upper()})")
    fig.savefig(args.out, dpi=150)
    print(f"[OK] saved {args.out}")

if __name__ == "__main__":
    main()