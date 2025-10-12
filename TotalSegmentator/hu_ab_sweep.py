#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, argparse, csv, json
import numpy as np
import nibabel as nib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from subprocess import run

def coronal_color_png(arr, aff, out_png, hu_window=(-1000,1000), coronal_orient="AP", base_h_in=7.0):
    # arr: RAS (X,Y,Z), float32
    dx,dy,dz = float(aff[0,0]), float(aff[1,1]), float(aff[2,2])
    X,Y,Z = arr.shape
    sl = arr[X//2,:,:]         # (Y,Z)
    im = np.flipud(sl)
    if coronal_orient.upper()=="AP":
        im = np.fliplr(im)
    vmin, vmax = hu_window
    w_mm, h_mm = dz*Z, dy*Y
    aspect = h_mm / max(w_mm, 1e-6)
    h_in = float(base_h_in); w_in = h_in / max(aspect, 1e-6)
    fig,ax = plt.subplots(figsize=(w_in,h_in), dpi=150)
    hm = ax.imshow(im, origin="lower", extent=[0, w_mm, 0, h_mm],
                   cmap="turbo", vmin=vmin, vmax=vmax, aspect="equal")
    ax.set_title("Coronal (AP)")
    ax.set_xlabel("Z [mm]"); ax.set_ylabel("Y [mm]")
    cb = fig.colorbar(hm, ax=ax, shrink=0.8); cb.set_label("Hounsfield Units [HU]")
    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
    fig.savefig(out_png, dpi=150, bbox_inches="tight"); plt.close(fig)

def stats_for(arr):
    # Recompute masks after transform (realistischer als lineare Prozentil-Umrechnung)
    body = arr > -500
    if not np.any(body):
        return dict(valid=False)
    soft_mask = body & (arr > -200) & (arr < 300)
    lung_mask = body & (arr < -400)
    bone_mask = body & (arr > 300)

    def safe_med(m): 
        v = arr[m]; 
        return float(np.median(v)) if v.size>100 else float("nan")
    def safe_p95(m):
        v = arr[m]; 
        return float(np.percentile(v,95)) if v.size>100 else float("nan")

    return dict(
        valid=True,
        lung_med=safe_med(lung_mask),
        soft_med=safe_med(soft_mask),
        bone_p95=safe_p95(bone_mask),
        min=float(np.nanmin(arr)),
        max=float(np.nanmax(arr)),
        mean=float(np.nanmean(arr)),
        med=float(np.nanmedian(arr)),
    )

def score_stats(st, targets, weights):
    if not st.get("valid", False): 
        return 1e30
    e = 0.0
    # quadratische Fehler
    e += weights["soft_med"] * (st["soft_med"] - targets["soft_med"])**2
    lm = st["lung_med"]
    if np.isfinite(lm):
        e += weights["lung_med"] * (lm - targets["lung_med"])**2
    bp = st["bone_p95"]
    if np.isfinite(bp):
        # Knochen p95: nur bestrafen, wenn zu niedrig
        e += weights["bone_p95"] * max(0.0, targets["bone_p95"] - bp)**2
    # Regularisierung: nicht zu extreme a/b
    return e

def main():
    ap = argparse.ArgumentParser(description="Grid-Suche für HU_out = a*HU_in + b mit Previews/CSV/optional TS")
    ap.add_argument("--in", dest="in_path", required=True, help="Eingabe NIfTI (float32 HU), RAS, bereits Body-masked/prepped")
    ap.add_argument("--outdir", default="runs_360/ab_sweep", help="Ausgabeordner")
    ap.add_argument("--alist", default="0.90,0.95,1.00,1.05,1.10,1.15,1.20", help="Kommagetrennte a-Werte")
    ap.add_argument("--blist", default="-60,-40,-20,0,20,40,60", help="Kommagetrennte b-Werte (HU)")
    ap.add_argument("--lock_liver", action="store_true", help="b automatisch so wählen, dass Leber-Median ~ target_soft_med bleibt (überschreibt --blist)")
    ap.add_argument("--liver_yz_mm", default="480,180", help="Leber-ROI (y_mm,z_mm), x wird mittig genommen")
    ap.add_argument("--target_soft_med", type=float, default=60.0)
    ap.add_argument("--target_lung_med", type=float, default=-750.0)
    ap.add_argument("--target_bone_p95", type=float, default=800.0)
    ap.add_argument("--w_soft_med", type=float, default=3.0)
    ap.add_argument("--w_lung_med", type=float, default=1.5)
    ap.add_argument("--w_bone_p95", type=float, default=1.5)
    ap.add_argument("--save_top", type=int, default=3, help="Nur die Top-Kandidaten als NIfTI speichern (alle Previews & CSV werden immer geschrieben)")
    ap.add_argument("--run_ts", action="store_true", help="TotalSegmentator auf besten Kandidaten ausführen")
    ap.add_argument("--ts_args", default="", help="Zusatzargumente für TotalSegmentator (z.B. \"--preview -ot nifti\")")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    base_img = nib.load(args.in_path)
    base = np.asarray(base_img.dataobj, np.float32)
    dx,dy,dz = base_img.header.get_zooms()[:3]
    X,Y,Z = base.shape

    # Liver-ROI (für lock_liver)
    ly, lz = [float(t) for t in args.liver_yz_mm.split(",")]
    ix, iy, iz = int(X/2), int(ly/dy), int(lz/dz)
    roi = base[ix-8:ix+8, max(0,iy-12):min(Y,iy+12), max(0,iz-12):min(Z,iz+12)]
    liver_med_base = float(np.median(roi[np.isfinite(roi)])) if roi.size else 0.0

    alist = [float(x) for x in args.alist.split(",") if x.strip()]
    if args.lock_liver:
        blist = [None]  # wird für jedes a berechnet
    else:
        blist = [float(x) for x in args.blist.split(",") if x.strip()]

    targets = dict(soft_med=args.target_soft_med, lung_med=args.target_lung_med, bone_p95=args.target_bone_p95)
    weights = dict(soft_med=args.w_soft_med, lung_med=args.w_lung_med, bone_p95=args.w_bone_p95)

    rows = []
    previews = []
    candidates = []

    for a in alist:
        bvals = blist
        if args.lock_liver:
            # b so wählen, dass Leber-Median ~ target_soft_med
            b_auto = args.target_soft_med - a*liver_med_base
            bvals = [float(np.clip(b_auto, -200, 200))]
        for b in bvals:
            arr = (base*a + b).astype(np.float32)
            arr = np.clip(arr, -1024, 2500)
            st = stats_for(arr)
            sc = score_stats(st, targets, weights)
            name = f"a{a:.3f}_b{b:+.0f}"
            # Preview
            png = os.path.join(args.outdir, f"{name}_corHU.png")
            coronal_color_png(arr, base_img.affine, png, hu_window=(-1000,1000))
            previews.append(png)
            # Tabellenzeile
            row = dict(name=name, a=a, b=b, score=sc,
                       lung_med=st.get("lung_med", np.nan),
                       soft_med=st.get("soft_med", np.nan),
                       bone_p95=st.get("bone_p95", np.nan),
                       glob_min=st.get("min", np.nan),
                       glob_max=st.get("max", np.nan),
                       glob_mean=st.get("mean", np.nan),
                       glob_med=st.get("med", np.nan))
            rows.append(row)
            candidates.append((sc, a, b, arr))

    # CSV schreiben
    csv_path = os.path.join(args.outdir, "ab_sweep_summary.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(sorted(rows, key=lambda r: r["score"]))
    print(f"[OK] Summary: {csv_path}")

    # Top-K speichern
    candidates.sort(key=lambda t: t[0])
    top = candidates[:max(1, args.save_top)]
    saved = []
    for rank,(sc,a,b,arr) in enumerate(top, start=1):
        out = os.path.join(args.outdir, f"BEST{rank}_a{a:.3f}_b{b:+.0f}.nii")
        nib.save(nib.Nifti1Image(arr, base_img.affine), out)
        saved.append(out)
        print(f"[OK] Saved #{rank}: {out} (score={sc:.1f})")

    # Optional TS auf Best1
    if args.run_ts and saved:
        best = saved[0]
        seg_dir = os.path.join(args.outdir, "ts_out")
        os.makedirs(seg_dir, exist_ok=True)
        cmd = ["TotalSegmentator", "-i", best, "-o", seg_dir, "--ml"]
        if args.ts_args:
            cmd += args.ts_args.split()
        print("[INFO] Run TS:\n  "+" ".join(cmd))
        run(cmd, check=True)
        print(f"[OK] TS done → {seg_dir}")

if __name__ == "__main__":
    main()