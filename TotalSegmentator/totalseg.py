#!/usr/bin/env python3
import numpy as np
import nibabel as nib
import subprocess
import os
import argparse
import matplotlib
matplotlib.use("Agg")  # headless: PNG statt Fenster
import matplotlib.pyplot as plt

def main():
    parser = argparse.ArgumentParser(
        description="XCAT attenuation (.bin, µ in mm^-1 @~140 keV) → HU → NIfTI → TotalSegmentator"
    )
    parser.add_argument('-i', '--input', required=True, help='Pfad zur .bin-Datei (XCAT attenuation map)')
    parser.add_argument('-o', '--output', required=True, help='Ordner für TotalSegmentator-Ausgabe')

    # Volumen-Infos (XCAT ist Fortran-ordered)
    parser.add_argument('--shape', nargs=3, type=int, default=[256, 256, 651],
                        help='Volumen-Dimensionen x y z')
    parser.add_argument('--voxel', nargs=3, type=float, default=[1.5, 1.5, 1.5],
                        help='Voxelgröße in mm (dx dy dz)')

    # µ_Wasser: automatisch oder manuell
    parser.add_argument('--mu_water_mm', type=float, default=None,
                        help='µ_Wasser in mm^-1 (manuell setzen)')
    parser.add_argument('--auto_mu', action='store_true',
                        help='µ_Wasser automatisch aus Histogramm (größter Nicht-Luft-Peak)')

    # HU-Output / Diagnose
    parser.add_argument('--clip', nargs=2, type=int, default=[-1000, 2000],
                        help='HU-Clipping [min max]')
    parser.add_argument('--int16_out', action='store_true',
                        help='NIfTI als int16 speichern (typisch für CT)')
    parser.add_argument('--plot_hist', action='store_true',
                        help='Histogramm der HU-Werte als PNG speichern')

    # TS-Settings
    parser.add_argument('--nr', type=int, default=1, help='Threads für Resampling (TS)')
    parser.add_argument('--ns', type=int, default=1, help='Threads für Saving (TS)')
    parser.add_argument('--device', choices=['gpu', 'cpu', 'mps'], default='gpu',
                        help='TotalSegmentator Gerät: gpu, gpu:X, cpu, mps')
    parser.add_argument('--skip_seg', action='store_true',
                        help='Nur NIfTI erzeugen, Segmentierung überspringen')

    # Optional direkt TS-Flags durchreichen
    parser.add_argument('--ts_body_crop', action='store_true',
                        help='TotalSegmentator Body-Crop aktivieren (-bs)')
    parser.add_argument('--roi_subset', nargs='*', default=[],
                        help='Liste von Labels für -rs (z.B. liver spleen kidney_left kidney_right …)')

    args = parser.parse_args()

    # ----------------- .bin laden -----------------
    print(f"Lade {args.input} ...")
    raw = np.fromfile(args.input, dtype=np.float32)
    expected = int(np.prod(args.shape))
    if raw.size != expected:
        raise ValueError(f"Elementanzahl passt nicht zu shape {args.shape}: {raw.size} vs {expected}")
    # XCAT → Fortran-Order
    mu_mm = raw.reshape(args.shape, order='F').astype(np.float32)
    print(f"Volumen geladen: shape={mu_mm.shape}, dtype={mu_mm.dtype}")
    print(f"µ(mm^-1): min={mu_mm.min():.6f}, max={mu_mm.max():.6f}, mean={mu_mm.mean():.6f}, unique≈{len(np.unique(mu_mm))}")

    # ----------------- µ_Wasser bestimmen -----------------
    if args.auto_mu:
        vals = mu_mm[mu_mm > 0.001].ravel()
        if vals.size < 10:
            raise ValueError("Zu wenige Nicht-Luft-Voxel für auto µ_Wasser.")
        hist, bin_edges = np.histogram(vals, bins=500)
        idx_max = int(np.argmax(hist))
        mu_water = float((bin_edges[idx_max] + bin_edges[idx_max+1]) / 2.0)
        print(f"Automatisch bestimmtes µ_Wasser ≈ {mu_water:.6f} mm^-1")
    else:
        mu_water = float(args.mu_water_mm) if args.mu_water_mm is not None else 0.0153
        print(("Manuell gesetztes " if args.mu_water_mm is not None else "Fallback ") +
              f"µ_Wasser = {mu_water:.6f} mm^-1")

    # ----------------- HU-Umrechnung -----------------
    volume_hu = 1000.0 * (mu_mm - mu_water) / mu_water
    # Luft explizit: µ==0 → -1000 HU
    volume_hu[mu_mm == 0.0] = -1000.0
    # CT-typisches Clipping
    hu_min, hu_max = args.clip
    volume_hu = np.clip(volume_hu, hu_min, hu_max).astype(np.float32)
    print(f"HU: min={volume_hu.min():.1f}, max={volume_hu.max():.1f}, mean={float(volume_hu.mean()):.1f}")

    # ----------------- Histogramm (optional) -----------------
    if args.plot_hist:
        plt.figure(figsize=(8, 4))
        plt.hist(volume_hu.ravel(), bins=200)
        plt.xlabel("HU"); plt.ylabel("Voxelanzahl")
        plt.title(f"HU-Histogramm (µ_Wasser={mu_water:.5f} mm^-1)")
        out_hist = os.path.splitext(os.path.basename(args.input))[0] + "_HU_histogram.png"
        plt.savefig(out_hist, dpi=150, bbox_inches='tight')
        print(f"Histogramm gespeichert: {out_hist}")

    # ----------------- NIfTI speichern -----------------
    affine = np.diag(list(args.voxel) + [1.0]).astype(np.float32)
    nii_path = os.path.splitext(os.path.basename(args.input))[0] + "_hu_scaled.nii.gz"
    data_to_save = np.round(volume_hu).astype(np.int16) if args.int16_out else volume_hu
    img = nib.Nifti1Image(data_to_save, affine)
    img.set_qform(affine); img.set_sform(affine)
    nib.save(img, nii_path)
    print(f"NIfTI gespeichert: {nii_path}")
    print(f"Affine/Spacing (mm): diag = {args.voxel}")

    # ----------------- Segmentierung -----------------
    if args.skip_seg:
        print("Segmentierung übersprungen (--skip_seg).")
        return

    # Synonyme/Abkürzungen abfangen (kleine Helfer gegen Tippfehler)
    synonyms = {
        "kidneys": ["kidney_left", "kidney_right"],
        "kidney": ["kidney_left", "kidney_right"],
        "lungs": ["lung_upper_lobe_left","lung_upper_lobe_right",
                  "lung_middle_lobe_right","lung_lower_lobe_left","lung_lower_lobe_right"],
        "lung": ["lung_upper_lobe_left","lung_upper_lobe_right",
                 "lung_middle_lobe_right","lung_lower_lobe_left","lung_lower_lobe_right"],
    }
    roi_expanded = []
    for r in args.roi_subset:
        roi_expanded.extend(synonyms.get(r, [r]))

    cmd = ["TotalSegmentator","-i",nii_path,"-o",args.output,
           "-nr",str(args.nr),"-ns",str(args.ns),"-d",args.device]
    if args.ts_body_crop:
        cmd.append("-bs")
    if roi_expanded:
        cmd.extend(["-rs"] + roi_expanded)

    print("Starte TotalSegmentator:", " ".join(cmd))
    subprocess.run(cmd, check=True)
    print(f"Segmentierung fertig in {args.output}")

if __name__ == "__main__":
    main()