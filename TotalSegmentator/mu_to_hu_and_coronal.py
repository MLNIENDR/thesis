#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mu_to_hu_and_precheck.py
------------------------
- Liest eine NIfTI mit linearem Dämpfungskoeffizienten µ [1/mm]
- Rechnet in Hounsfield Units (HU) um:
      HU = 1000 * (µ / µ_water - 1)
- µ_water kann manuell (--mu_water_mm) oder automatisch (--auto_mu) geschätzt werden
  (robuste Peak-Suche ~Weichgewebe in 0.02..0.04 1/mm; Fallback 0.030).
- Optionaler Clip, optional int16-Ausgabe.
- Speichert Histogramm als PNG (Agg, headless).
- Erstes Plausibilitätsbild: Coronal-AP (zentraler Y-Slice), korrekt skaliert in mm,
  mit bewährten Defaults: swap=True, flip180=True, view=AP. Flags zum Modifizieren vorhanden.

Abhängigkeiten: nibabel, numpy, scipy, matplotlib
"""

import os, sys, argparse, logging
import numpy as np
import nibabel as nib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ----------------------- Logging -----------------------
def setup_logging():
    logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

# ----------------------- IO utils ----------------------
def ensure_dir(path: str):
    d = os.path.dirname(path)
    if d and not os.path.isdir(d):
        os.makedirs(d, exist_ok=True)

# ----------------- Auto µ_water Schätzung --------------
def estimate_mu_water_auto(mu: np.ndarray) -> float:
    """
    Schätzt µ_water robust aus Histogramm im Bereich 0.02..0.04 1/mm.
    Fallback: 0.030.
    """
    mu = mu[np.isfinite(mu)]
    if mu.size == 0:
        return 0.030
    # Grob auf plausible Range beschränken, aber nicht alles wegwerfen:
    mu_clip = mu[(mu >= 0.005) & (mu <= 0.08)]
    if mu_clip.size < 100:
        mu_clip = mu  # notfalls alles

    # Feiner Fokus auf 0.02..0.04
    soft = mu_clip[(mu_clip >= 0.02) & (mu_clip <= 0.04)]
    if soft.size >= 100:
        # Peak via Histogramm-Maximum
        hist, edges = np.histogram(soft, bins=256, range=(0.02, 0.04))
        idx = int(np.argmax(hist))
        muw = 0.5 * (edges[idx] + edges[idx+1])
        return float(muw)
    else:
        # Fallback Median im groben Clip
        return float(np.median(mu_clip)) if mu_clip.size > 0 else 0.030

# ------------------ HU Konvertierung -------------------
def mu_to_hu(mu_img: nib.Nifti1Image, mu_water: float) -> nib.Nifti1Image:
    mu = mu_img.get_fdata(dtype=np.float32)
    with np.errstate(divide='ignore', invalid='ignore'):
        hu = 1000.0 * (mu / float(mu_water) - 1.0)
    # Affine & Header übernehmen
    return nib.Nifti1Image(hu.astype(np.float32), mu_img.affine, mu_img.header)

# ------------------ Histogramm Plot --------------------
def plot_histogram(mu: np.ndarray, out_png: str, mu_water: float):
    ensure_dir(out_png)
    plt.figure(figsize=(6,4), dpi=150)
    vals = mu[np.isfinite(mu)]
    vals = vals[(vals >= 0.0) & (vals <= 0.08)]
    if vals.size == 0:
        vals = mu[np.isfinite(mu)]
    plt.hist(vals.flatten(), bins=256, range=(max(0.0, np.min(vals)), np.max(vals)), alpha=0.8)
    plt.axvline(mu_water, linestyle='--', linewidth=1.0)
    plt.title(f"µ-Histogramm (µ_water ≈ {mu_water:.4f} 1/mm)")
    plt.xlabel("µ [1/mm]")
    plt.ylabel("Voxels")
    plt.tight_layout()
    plt.savefig(out_png, bbox_inches='tight')
    plt.close()

# --------------- Coronal-AP Quicklook ------------------
def make_quick_coronal_ap_png(hu_img: nib.Nifti1Image,
                              out_png: str,
                              win_low: float = None, win_high: float = None,
                              base_h_in: float = 6.0,
                              do_swap: bool = True,
                              do_flip180: bool = True,
                              view: str = "ap"):
    """
    Schnelles Plausibilitätsbild:
    - NIfTI → RAS-nahe (as_closest_canonical)
    - Explizit als (X,Y,Z) interpretieren (wir nehmen shape → (nx,ny,nz), affine-Diagonale → Voxelgrößen)
    - Coronal: zentraler Y-Slice → Bild hat (Z, X) Pixel (wird auf Extents X(mm) x Z(mm) geplottet)
    - bewährte Defaults: swap(True), flip180(True), view='ap'
      * swap: (Z,X) -> (X,Z) Darstellung (wir transponieren, damit horizontale Achse X ist)
      * flip180: 180°-Flip (np.flipud + np.fliplr)
      * AP vs PA: bei AP zusätzlich L/R-Flip (np.fliplr), damit Blickrichtung passt
    """
    ensure_dir(out_png)

    # In RAS-nahe Anordnung bringen
    img = nib.as_closest_canonical(hu_img)
    data = img.get_fdata(dtype=np.float32)
    affine = img.affine.copy()

    # Voxelgrößen (mm) aus Affine-Diagonale (Annahme: diag)
    dx, dy, dz = float(abs(affine[0,0])), float(abs(affine[1,1])), float(abs(affine[2,2]))
    nx, ny, nz = data.shape  # (X,Y,Z) in RAS

    # zentraler Coronal-Slice (Y-Mitte)
    ymid = ny // 2
    cor = data[:, ymid, :]  # (X,Z)

    # optionales Windowing
    if win_low is not None and win_high is not None and win_high > win_low:
        wl, wh = float(win_low), float(win_high)
        cor = np.clip(cor, wl, wh)
        if wh - wl > 1e-6:
            cor = (cor - wl) / (wh - wl)  # 0..1
    else:
        # robustes Auto-Window (5..95 Perzentil)
        lo, hi = np.percentile(cor[np.isfinite(cor)], [5, 95])
        if hi > lo:
            cor = np.clip(cor, lo, hi)
            cor = (cor - lo) / (hi - lo)

    # bewährtes Rezept
    img2d = cor.T  # (Z,X) → für Darstellung leichter zu drehen/flippen
    if do_flip180:
        img2d = np.flipud(np.fliplr(img2d))  # 180° Flip
    if view.lower() == "ap":
        img2d = np.fliplr(img2d)             # L/R Flip für AP
    # swap: sicherstellen, dass Breite=X(mm), Höhe=Z(mm)
    if do_swap:
        # img2d ist aktuell (Z,X); für plot mit extent(X,Z) ist (Z,X) bereits korrekt
        # Wir belassen es und steuern die Extents entsprechend.
        pass

    # Physikalische Extents
    width_mm = dx * nx   # X-Richtung
    height_mm = dz * nz  # Z-Richtung

    # Plot
    aspect = height_mm / width_mm
    fig_w = max(3.0, base_h_in / aspect)
    fig_h = base_h_in

    plt.figure(figsize=(fig_w, fig_h), dpi=150)
    extent = (0, width_mm, 0, height_mm)  # X(mm) horizontal, Z(mm) vertikal
    plt.imshow(img2d, cmap="gray", extent=extent, origin="lower", interpolation="nearest")
    plt.xlabel("X [mm]")
    plt.ylabel("Z [mm]")
    plt.title("Quick Coronal (AP)" if view.lower()=="ap" else "Quick Coronal (PA)")
    plt.tight_layout()
    plt.savefig(out_png, bbox_inches='tight')
    plt.close()

# --------------------------- MAIN ----------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--in_nifti", required=True, help="Input NIfTI (µ in 1/mm)")
    ap.add_argument("-o", "--out_hu", required=True, help="Output NIfTI (HU)")
    ap.add_argument("--mu_water_mm", type=float, default=None, help="µ_water in 1/mm")
    ap.add_argument("--auto_mu", action="store_true", help="µ_water automatisch schätzen (0.02..0.04)")
    ap.add_argument("--clip", nargs=2, type=float, default=None, metavar=("LOW","HIGH"),
                    help="HU Clip [LOW, HIGH], z.B. -1024 3071")
    ap.add_argument("--int16_out", action="store_true", help="HU als int16 speichern")
    ap.add_argument("--hist_png", default="ts_preview/mu_hist.png", help="Histogramm-PNG")
    # Quicklook Coronal
    ap.add_argument("--quick_ap_png", default="ts_preview/quick_coronal_AP.png",
                    help="Quick Coronal-AP PNG-Ausgabe")
    ap.add_argument("--base_h_in", type=float, default=6.0, help="Höhe der Figur in Zoll")
    ap.add_argument("--win_low", type=float, default=None)
    ap.add_argument("--win_high", type=float, default=None)
    ap.add_argument("--no-swap", dest="swap", action="store_false", help="Swap deaktivieren")
    ap.add_argument("--no-flip180", dest="flip180", action="store_false", help="180° Flip deaktivieren")
    ap.add_argument("--view", choices=["ap","pa"], default="ap")
    args = ap.parse_args()

    setup_logging()

    # --- Laden ---
    mu_img = nib.load(args.in_nifti)
    mu = mu_img.get_fdata(dtype=np.float32)
    logging.info(f"Input µ shape={mu.shape} | affine diag≈({mu_img.affine[0,0]:.4f},{mu_img.affine[1,1]:.4f},{mu_img.affine[2,2]:.4f})")

    # --- µ_water ---
    muw = args.mu_water_mm
    if args.auto_mu or muw is None:
        muw = estimate_mu_water_auto(mu)
        logging.info(f"µ_water (auto) ≈ {muw:.5f} 1/mm")
    else:
        logging.info(f"µ_water (manuell) = {muw:.5f} 1/mm")

    # --- Histogramm ---
    if args.hist_png:
        plot_histogram(mu, args.hist_png, muw)
        logging.info(f"Histogramm gespeichert: {args.hist_png}")

    # --- HU rechnen ---
    hu_img = mu_to_hu(mu_img, muw)
    hu = hu_img.get_fdata(dtype=np.float32)

    # optional Clip + int16
    if args.clip is not None and len(args.clip) == 2:
        lo, hi = float(args.clip[0]), float(args.clip[1])
        hu = np.clip(hu, lo, hi)
        if args.int16_out:
            hu_out = hu.astype(np.int16)
        else:
            hu_out = hu.astype(np.float32)
        hu_img = nib.Nifti1Image(hu_out, hu_img.affine, hu_img.header)
        logging.info(f"HU geclippt auf [{lo}, {hi}] | dtype={'int16' if args.int16_out else 'float32'}")
    else:
        if args.int16_out:
            hu_img = nib.Nifti1Image(hu.astype(np.int16), hu_img.affine, hu_img.header)
            logging.info("HU dtype=int16 (ohne Clip)")

    # --- HU-NIfTI speichern ---
    nib.save(hu_img, args.out_hu)
    logging.info(f"HU-CT gespeichert: {args.out_hu}")

    # --- Quick Coronal-AP PNG ---
    if args.quick_ap_png:
        make_quick_coronal_ap_png(hu_img,
                                  args.quick_ap_png,
                                  win_low=args.win_low, win_high=args.win_high,
                                  base_h_in=args.base_h_in,
                                  do_swap=args.swap, do_flip180=args.flip180,
                                  view=args.view)
        logging.info(f"Quick Coronal PNG gespeichert: {args.quick_ap_png}")

if __name__ == "__main__":
    main()