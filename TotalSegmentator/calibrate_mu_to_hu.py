#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Robuste µ→HU-Kalibrierung (2-Punkt-Fit) + korrekte Coronal-Preview (AP, 800×500 mm).

Idee: Das rekonstruierte Volumen liegt aktuell in "µ"-Einheiten (lineare Schwächungskoeffizienten).
Um es CT-typisch interpretieren zu können, werden die µ linear in Hounsfield Units (HU) skaliert.

Zwei-Punkt-Kalibrierung:
  - Wir nehmen zwei bekannte Referenzpunkte in µ:
      1) Luft (air) soll −1000 HU ergeben
      2) Wasser/Weichteil (tissue) soll "target_tissue_hu" (üblich: 0 HU oder ca. +40 HU) ergeben
  - Daraus bestimmen wir eine lineare Abbildung: HU = alpha * µ + beta

Woher kommen die µ-Referenzwerte?
  - Entweder aus Phantom-ROIs (Masks oder Bounding-Box) → präzise
  - Oder automatisch:
      • Luft: median aus „außerhalb der Körpermaske“ (unteres Dezil)
      • Weichteil: mittlerer Cluster aus k≈3-Clustern (Fett | Soft | Knochen) im Körper

Zusätzlich:
  - Optionales Clipping der HU auf [-1050, 3000]
  - Coronal-Preview (AP) mit echten Millimeter-Extents (X horizontal, Z vertikal)
"""

# --------- Standard-Imports ---------
import argparse, json, os
import numpy as np
import nibabel as nib

# Matplotlib im Headless-Modus (keine GUI erforderlich)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --- optionale SciPy-Features (für Morphologie/Connected Components). Fallback, falls nicht installiert. ---
try:
    from scipy.ndimage import label, binary_closing, generate_binary_structure
    _HAVE_SCIPY = True
except Exception:
    _HAVE_SCIPY = False


# ======================================================================
# =============== Hilfsfunktionen: Masken & Preview =====================
# ======================================================================

def load_mask_or_none(path):
    """
    Lädt eine NIfTI-Maske, falls Pfad angegeben.
    - Akzeptiert auch probabilistische Masken (Float 0..1), die wir binarisieren (>0.5).
    - Rückgabe: uint8 (0/1) oder None.
    """
    if path is None:
        return None
    m = np.asarray(nib.load(path).dataobj)
    return (m > 0.5).astype(np.uint8)


def apply_bbox(shape, bbox):
    """
    Erzeuge eine binäre Maske aus einer Bounding-Box.
    bbox = (x0, x1, z0, z1, y0, y1) in Voxel-Indizes (x1/z1/y1 exklusiv).
    shape = (X, Z, Y), wie dein Volumen.
    """
    x0, x1, z0, z1, y0, y1 = map(int, bbox)
    m = np.zeros(shape, dtype=np.uint8)
    m[x0:x1, z0:z1, y0:y1] = 1
    return m


def save_coronal_png_xzy(nii_path, out_png, base_h_in=8.0, title="Coronal (AP)"):
    """
    Speichert einen korrekten Coronal-Slice (AP) als PNG:
      - Wir lesen das NIfTI (Layout: (X,Z,Y), Affine diag([dx,dz,dy])).
      - Slice: mittlere Y-Schicht → 2D-Bild (X×Z).
      - AP & "Kopf oben": Transponieren + vertikales Flip.
      - Fensterung über 2..98-Perzentil, auf [0,1] normiert (nur Darstellung).
      - mm-Extent aus Affine → keine Verzerrung im Bild (aspect='equal').
    """
    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
    img = nib.load(nii_path)
    vol = np.asarray(img.dataobj, dtype=np.float32)  # (X,Z,Y)
    aff = img.affine
    dx, dz, dy = float(aff[0, 0]), float(aff[1, 1]), float(aff[2, 2])
    X, Z, Y = vol.shape

    # Mitten-Schnitt in Y → coronal
    sl = vol[:, :, Y // 2]           # (X,Z)
    # Für "AP, Kopf oben": (X,Z) → (Z,X) transponieren und dann vertikal spiegeln
    img2 = np.flipud(sl.T)

    # Sanfte Fensterung für Display (keine HU-Änderung, nur Visualisierung)
    v = img2[np.isfinite(img2)]
    p2, p98 = np.percentile(v, [2, 98]) if v.size else (0.0, 1.0)
    if p2 == p98:
        p98 = p2 + np.finfo(np.float32).eps
    shown = np.clip((img2 - p2) / (p98 - p2), 0, 1)

    # Physikalische Größe (mm) des Bildes → X-Breite, Z-Höhe
    width_mm  = dx * X            # ~500 mm
    height_mm = dz * Z            # ~800 mm
    extent    = [0, width_mm, 0, height_mm]
    aspect    = height_mm / max(width_mm, 1e-6)

    # Figure so skalieren, dass Seitenverhältnis = mm-Verhältnis
    h_in = float(base_h_in)
    w_in = h_in / aspect

    fig, ax = plt.subplots(figsize=(w_in, h_in), dpi=200)
    ax.imshow(shown, cmap="gray", origin="lower", extent=extent, aspect="equal")
    ax.set_xlabel("X [mm]"); ax.set_ylabel("Z [mm]"); ax.set_title(title)
    fig.savefig(out_png, dpi=200, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print(f"[OK] Coronal (AP) gespeichert: {out_png}  | extent={width_mm:.1f}×{height_mm:.1f} mm")


# ======================================================================
# =============== Robuste Schätzer für µ-Referenzen ====================
# ======================================================================

def robust_air_from_outside(mu, inside_mask):
    """
    Schätze µ für Luft:
      - „Außerhalb“ der Körpermaske (inside_mask==0) → dort liegt meist Luft.
      - Nimm unteres Dezil (10%) und davon den Median → robust gegen Ausreißer.
    """
    outside = (inside_mask == 0)
    vals = mu[outside]
    vals = vals[np.isfinite(vals)]
    if vals.size < 100:
        raise ValueError("Zu wenige Luft-Voxel außerhalb der Körpermaske.")
    t = np.percentile(vals, 10)
    return float(np.median(vals[vals <= t]))


def robust_tissue_from_inside(mu, inside_mask, prefer_soft_target=True):
    """
    Schätze µ für „Weichteil/Soft Tissue“:
      - Werte *innerhalb* der Körpermaske (inside_mask==1)
      - 1D-KMeans mit k=3 (kleine Eigen-Imp.): grob (Fett | Soft | Knochen)
      - Rückgabe: Median des mittleren Clusters.
    """
    vals = mu[(inside_mask == 1)]
    vals = vals[np.isfinite(vals)]
    if vals.size < 1000:
        raise ValueError("Zu wenige Innen-Voxel für Soft-Tissue-Schätzung.")

    # Initialisiere Clusterzentren grob über Perzentile (robust)
    p = np.percentile(vals, [20, 50, 80])
    centers = np.array(p, dtype=np.float32)

    # Lloyd-ähnliche Iteration (12 Runden reichen hier völlig)
    for _ in range(12):
        d = np.abs(vals[:, None] - centers[None, :])   # Distanz zu allen Zentren
        idx = np.argmin(d, axis=1)                    # nächstes Zentrum
        new_centers = []
        for k in range(3):
            sel = vals[idx == k]
            new_centers.append(np.median(sel) if sel.size > 0 else centers[k])
        new_centers = np.array(new_centers, dtype=np.float32)
        if np.allclose(new_centers, centers, rtol=0, atol=1e-6):
            centers = new_centers
            break
        centers = new_centers

    # Sortiere, damit c0<c1<c2 (Fett | Soft | Knochen grob)
    centers.sort()
    c0, c1, c2 = centers.tolist()
    return float(c1)  # mittlerer Cluster = Soft Tissue


def make_body_mask(mu):
    """
    Grobe Körpermaske (nur µ-Bild, keine HU nötig):
      - Schwellwert bei P60 der Intensitäten → separiert Luft (drunter) vs Körper (drüber).
      - Mit SciPy: kleines Closing + größte zusammenhängende Komponente.
      - Rückgabe: uint8 Maske (0/1).
    """
    v = mu[np.isfinite(mu)]
    if v.size == 0:
        raise ValueError("Volumen enthält keine finiten Werte.")

    thr = np.percentile(v, 60)
    m = (mu > thr).astype(np.uint8)

    if not _HAVE_SCIPY:
        return m  # einfacher Fallback

    # Morphologisches „Aufräumen“ + größte Komponente auswählen
    st = generate_binary_structure(3, 1)
    m = binary_closing(m, structure=st, iterations=1)
    lbl, n = label(m, structure=st)
    if n < 1:
        return m
    sizes = np.bincount(lbl.ravel()); sizes[0] = 0
    keep = np.argmax(sizes)
    return (lbl == keep).astype(np.uint8)


# ======================================================================
# ===================== 2-Punkt-Kalibrierung ===========================
# ======================================================================

def two_point_fit(mu_air, mu_tissue, target_tissue_hu=40.0):
    """
    Bestimme lineare Abbildung HU = alpha*µ + beta mit Bedingungen:
      - µ_air  → -1000 HU
      - µ_tissue → target_tissue_hu  (typisch 0 HU oder 40 HU)
    Algebra:
      alpha = (target - (-1000)) / (µ_tissue - µ_air)
      beta  = -1000 - alpha*µ_air
      µ_water = -beta/alpha  (der µ-Wert, der auf 0 HU abgebildet wird)
    """
    if not np.isfinite(mu_air) or not np.isfinite(mu_tissue) or (mu_tissue == mu_air):
        raise ValueError("Ungültige Kalibrierpunkte.")
    alpha = (target_tissue_hu - (-1000.0)) / (mu_tissue - mu_air)
    beta  = -1000.0 - alpha * mu_air
    mu_water = -beta / alpha
    return float(alpha), float(beta), float(mu_water)


# ======================================================================
# ============================ MAIN ====================================
# ======================================================================

def main():
    # ----------------- CLI-Argumente -----------------
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--in-nii", required=True,
                    help="Eingangs-µ-Volumen (Layout (X,Z,Y), Affine diag([dx,dz,dy]); z.B. aus recon-Skript).")
    ap.add_argument("-o", "--out-hu", required=True,
                    help="Ausgabe HU-Volumen (*.nii[.gz])")
    ap.add_argument("--target-tissue-hu", type=float, default=40.0,
                    help="Ziel-HU für Soft-Tissue (üblich 0 oder ~40). Default 40.")
    ap.add_argument("--clip", nargs=2, type=float, default=(-1050, 3000), metavar=("LOW", "HIGH"),
                    help="Optionales HU-Clipping (Visual/Downstream stabiler).")
    ap.add_argument("--report-json", default=None,
                    help="Optionaler Pfad für Kalibrier-Report (JSON mit Parametern/Stats).")
    ap.add_argument("--coronal-png", default=None,
                    help="Optional: PNG der Coronal-Ansicht (AP, 800×500 mm)")
    ap.add_argument("--base-h-in", type=float, default=8.0,
                    help="Höhe der PNG-Figur in Zoll; Breite wird aus mm-Verhältnis abgeleitet.")

    # Phantom-ROIs (Maske oder BBox). Beides darf kombiniert werden.
    ap.add_argument("--roi-air", default=None, help="NIfTI Maske für Luft-ROI")
    ap.add_argument("--roi-water", default=None, help="NIfTI Maske für Wasser/Soft-ROI")
    ap.add_argument("--bbox-air", nargs=6, type=int, metavar=("x0", "x1", "z0", "z1", "y0", "y1"),
                    help="Voxel-BBox für Luft (X,Z,Y)")
    ap.add_argument("--bbox-water", nargs=6, type=int, metavar=("x0", "x1", "z0", "z1", "y0", "y1"),
                    help="Voxel-BBox für Wasser/Soft-Tissue (X,Z,Y)")

    args = ap.parse_args()

    # ----------------- NIfTI laden -----------------
    img = nib.load(args.in_nii)
    mu  = np.asarray(img.dataobj, dtype=np.float32)  # Layout (X,Z,Y)
    aff = img.affine
    X, Z, Y = mu.shape

    # ----------------- ROIs vorbereiten -----------------
    # Falls Masken vorhanden, nutzen wir deren Median (robust).
    air_mask = load_mask_or_none(args.roi_air)
    wat_mask = load_mask_or_none(args.roi_water)

    # Bounding-Boxes (optional) ergänzen/vereinigen
    if args.bbox_air is not None:
        m = apply_bbox(mu.shape, args.bbox_air)
        air_mask = m if air_mask is None else np.logical_or(air_mask > 0, m > 0).astype(np.uint8)
    if args.bbox_water is not None:
        m = apply_bbox(mu.shape, args.bbox_water)
        wat_mask = m if wat_mask is None else np.logical_or(wat_mask > 0, m > 0).astype(np.uint8)

    # ----------------- Kalibrier-Punkte bestimmen -----------------
    mu_air = None
    mu_tissue = None

    # (1) Luft aus Phantom-ROI?
    if air_mask is not None:
        vals = mu[air_mask > 0]; vals = vals[np.isfinite(vals)]
        if vals.size < 50:
            raise ValueError("ROI Luft enthält zu wenige Werte.")
        mu_air = float(np.median(vals))

    # (2) Wasser/Soft aus Phantom-ROI?
    if wat_mask is not None:
        vals = mu[wat_mask > 0]; vals = vals[np.isfinite(vals)]
        if vals.size < 50:
            raise ValueError("ROI Wasser/Soft enthält zu wenige Werte.")
        mu_tissue = float(np.median(vals))

    # (3) Fehlende Punkte automatisch & robust schätzen
    #     → Körpermaske (aus µ) und dann Luft außen / Soft innen
    if (mu_air is None) or (mu_tissue is None):
        inside = make_body_mask(mu)
        if mu_air is None:
            mu_air = robust_air_from_outside(mu, inside)
        if mu_tissue is None:
            mu_tissue = robust_tissue_from_inside(mu, inside)

    # ----------------- 2-Punkt-Fit & Parameter-Report -----------------
    alpha, beta, mu_water = two_point_fit(mu_air, mu_tissue, args.target_tissue_hu)
    print(f"[CAL] mu_air={mu_air:.6g} | mu_tissue={mu_tissue:.6g} | target={args.target_tissue_hu:.1f} HU")
    print(f"[CAL] alpha={alpha:.6g} | beta={beta:.6g} | mu_water(=HU0)≈{mu_water:.6g}")

    # ----------------- Anwenden + Clipping -----------------
    hu = alpha * mu + beta
    lo, hi = args.clip
    hu = np.clip(hu, lo, hi).astype(np.float32)

    # ----------------- Plausibilitäts-Stats -----------------
    # Wir betrachten HU im Körper (inside) vs. außerhalb (outside):
    body = make_body_mask(mu)
    inside_vals = hu[body > 0]
    out_vals    = hu[body == 0]

    def stats(name, a):
        a = a[np.isfinite(a)]
        if a.size == 0:
            return f"{name}: n=0"
        p = np.percentile(a, [1, 50, 99])
        return (f"{name}: n={a.size}, min={a.min():.1f}, p1={p[0]:.1f}, "
                f"med={p[1]:.1f}, p99={p[2]:.1f}, max={a.max():.1f}, mean={a.mean():.1f}")

    print("[STATS]", stats("inside", inside_vals))
    print("[STATS]", stats("outside", out_vals))

    # ----------------- Speichern (HU) -----------------
    # Wir übernehmen Affine + Header unverändert (nur die Daten sind jetzt HU).
    hu_img = nib.Nifti1Image(hu, aff, img.header)
    nib.save(hu_img, args.out_hu)
    print(f"[OK] HU-Volumen gespeichert: {args.out_hu}")

    # ----------------- Optional: JSON-Report -----------------
    if args.report_json:
        rep = {
            "mu_air": mu_air, "mu_tissue": mu_tissue,
            "alpha": alpha, "beta": beta, "mu_water_est": mu_water,
            "clip": [lo, hi],
            "inside_stats": {
                "min": float(np.nanmin(inside_vals)) if inside_vals.size else None,
                "p1":  float(np.nanpercentile(inside_vals, 1)) if inside_vals.size else None,
                "p50": float(np.nanpercentile(inside_vals, 50)) if inside_vals.size else None,
                "p99": float(np.nanpercentile(inside_vals, 99)) if inside_vals.size else None,
                "max": float(np.nanmax(inside_vals)) if inside_vals.size else None,
                "mean": float(np.nanmean(inside_vals)) if inside_vals.size else None,
            }
        }
        with open(args.report_json, "w") as f:
            json.dump(rep, f, indent=2)
        print(f"[OK] Report geschrieben: {args.report_json}")

    # ----------------- Optional: Coronal-Preview -----------------
    if args.coronal_png:
        save_coronal_png_xzy(args.out_hu, args.coronal_png, base_h_in=args.base_h_in)


if __name__ == "__main__":
    main()