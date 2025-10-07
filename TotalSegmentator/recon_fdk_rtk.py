#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FDK-Rekonstruktion mit ITK-RTK aus xCAT/ct_projector-Projektionen (mu_stack),
plus optionaler Preview (coronal/axial/sagittal) mit mm-Extents.
Headless (Agg). Preview ist strikt RAS (keine Heuristik).

Eingang:
  - ct_proj_stack.mat:
      mu_stack: [nProj, nr, nc] oder [nr, nc, nProj] (wird erkannt)
      angles_deg (optional; sonst 0..360° gleichmäßig)
  - proj.txt: width_mm, height_mm, num_rows (nr), num_channels (nc),
              distance_to_source (DSO), distance_to_detector (DTD), detector_shift (u)

WICHTIG:
  - --swap_uv tauscht NUR die Datenachsen (nr<->nc). width_mm/height_mm werden NICHT getauscht.
    So bleibt width_mm weiterhin an nc (Breite), height_mm an nr (Höhe) gebunden.

NEU (nur Ergänzung, keine Verhaltensänderung):
  - Zusätzliches, kanonisches NIfTI für HU-Weiterverarbeitung: Array=(X,Y,Z), Affine=diag([dx,dy,dz])
    via --out_ras (Default: <out>_RAS.nii)
"""

import argparse, re, os
import numpy as np
import nibabel as nib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------------- I/O-Helper ----------------
def parse_proj_txt(path):                                                                                           # proj.txt einlesen und Geometrie-Parameter extrahieren
    txt = open(path, "r", encoding="utf-8", errors="ignore").read()

    def grab(key):                                                                                                  # suche <zahl> : <key>
        m = re.search(rf'^\s*([+-]?\d+(?:\.\d+)?)\s*:\s*{re.escape(key)}', txt, re.M | re.I)
        return float(m.group(1)) if m else None

    geom = {                                                                                                        # mehrere Schreibweisen erlaubt
        "width_mm":  grab("width(mms)")  or grab("width"),
        "height_mm": grab("height(mms)") or grab("height"),
        "nr": int(grab("num_rows") or 0),                                                                           # Zeilen HÖHE Detektor
        "nc": int(grab("num_channels") or 0),                                                                       # Kanäle BREITE Detektor
        "DSO": grab("distance_to_source(mms)")    or grab("distance_to_source"),
        "DTD": grab("distance_to_detector(mms)")  or grab("distance_to_detector"),
        "det_shift_u": grab("detector_shift") or 0.0,                                                               # lateraler Detektor-Offset (u-Richtung)
    }
    assert geom["DSO"] is not None and geom["DTD"] is not None, "DSO/DTD fehlen in proj.txt"                        # ohne DSO / DTD kann RTK die Geometrie nicht aufbauen
    return geom


def load_mu_stack_any(mat_path):                                                                                    # mu_stack aus .mat einlesen (mu, angles) --> angles ist optional, wenn nicht vorhanden, gleichmäßige Winkel über 0..360°
    """Lädt mu_stack aus .mat (v7/v7.3) und liefert (mu, angles)."""
    mu = None
    ang = None
    try:
        from scipy.io import loadmat
        M = loadmat(mat_path, squeeze_me=True, simplify_cells=True)
        mu = np.asarray(M["mu_stack"], np.float32)

        # <<< Änderung: Winkel-Präferenz angles_out > angles_deg >>>
        if "angles_out" in M:
            ang = np.asarray(M["angles_out"], np.float32)
        elif "angles_deg" in M:
            ang = np.asarray(M["angles_deg"], np.float32)
        else:
            ang = None

    except Exception:
        import h5py
        with h5py.File(mat_path, "r") as f:
            mu = np.array(f["mu_stack"], dtype=np.float32)
            # <<< Änderung: Winkel-Präferenz angles_out > angles_deg >>>
            if "angles_out" in f:
                ang = np.array(f["angles_out"], dtype=np.float32).squeeze()
            elif "angles_deg" in f:
                ang = np.array(f["angles_deg"], dtype=np.float32).squeeze()
            else:
                ang = None

    if mu.ndim != 3:
        raise ValueError(f"mu_stack hat ndims={mu.ndim}, erwarte 3.")
    # häufig [nr, nc, nProj] → nach [nProj, nr, nc] (transponiert wird dann, wenn die erste Dimension NICHT wie nProj aussieht (180 / 360))
    if mu.shape[0] not in (180, 360):
        mu = np.transpose(mu, (2, 0, 1))
    return mu.astype(np.float32, copy=False), (None if ang is None else ang.astype(np.float32, copy=False))


def ensure_axes(mu, nr, nc, nproj):
    """Erzwingt [nProj, nr, nc] – probiert Permutationen, sonst Fehler."""
    want = (nproj, nr, nc)
    if mu.shape == want:
        return mu
    from itertools import permutations
    for p in permutations((0, 1, 2)):
        t = np.transpose(mu, p)
        if t.shape == want:
            print(f"[FIX] permuted mu axes {p} -> {t.shape}")
            return t
    raise ValueError(f"Projektdaten-Form {mu.shape} passt nicht zu {want}.")


# ---------------- Preview (deterministisch in RAS) ----------------
def save_preview_png(nifti_path, out_png, view="coronal",
                     base_h_in=7.0, title=None):
    """
    Nimmt NIfTI und bringt es in RAS (Right-Anterior-Superior) und erzeugt .png der gewünschten Ebene.

    Wichtig: beim Speichern der Rekonstruktion wird bewusst die Affine vertauscht (diag([dx, dz, dy])), damit in der coronal view die vertikale Skala Z ist.

    Strikte RAS-Definitionen:
      axial    = vol[:, :, Z//2]  → X (horiz) × Y (vert)
      coronal  = vol[:, Y//2, :]  → X (horiz) × Z (vert)
      sagittal = vol[X//2, :, :]  → Y (horiz) × Z (vert)
    """
    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)

    img = nib.load(nifti_path)
    img_ras = nib.as_closest_canonical(img)                                                     # Achsen --> RAS-Orientierung
    vol = np.asarray(img_ras.dataobj, dtype=np.float32)                                         # Datenarray in (X,Y,Z)
    aff = img_ras.affine                                                                        # 4x4-Affine
    X, Y, Z = vol.shape

    v = view.lower()
    if v == "coronal":
        # Layout: aff[0,0] entspricht dx (mm pro X-Voxel) und aff[1,1] entspricht dz (mm pro Z-Voxel)
        dx = float(aff[0, 0])
        dz = float(aff[1, 1])
        # Coronal-Slice: alle X, festes Y, alle Z --> vol[:,Y//2,:]
        sl = vol[:, Y // 2, :]
        img2 = np.flipud(sl.T)                                                                  # AP + Kopf oben (transponieren für imshow(y,x) und flip)
        # Fensterung (2..98-Perzentil), skaliert auf [0,1]
        p2, p98 = np.percentile(img2, [2, 98])
        shown = np.clip((img2 - p2) / (p98 - p2 + 1e-6), 0, 1)                                  # Minimum wird auf 0 geschoben, alles über 1 wird auf 1 begrenzt (im Bereich 2..98-Perzentil)
        # physikalische Extents in mm: X-Breite = dx*X, Z-Höhe = dz*Z --> hochkant 500x800
        ex = [0, dx * X, 0, dz * Z]
        xlabel, ylabel = "X [mm]", "Z [mm]"
        ttl = title or "Coronal (AP)"
    elif v == "axial":
        # Axial: X horizontal, Y vertikal --> nutze aff[0,0] = dx, aff[1,1] = dy
        dx = float(aff[0, 0]); dy = float(aff[1, 1])
        sl = vol[:, :, Z // 2]
        img2 = np.flipud(sl.T)
        p2, p98 = np.percentile(img2, [2, 98])
        shown = np.clip((img2 - p2) / (p98 - p2 + 1e-6), 0, 1)
        ex = [0, dx * X, 0, dy * Y]
        xlabel, ylabel = "X [mm]", "Y [mm]"
        ttl = title or "Axial"
    elif v == "sagittal":
        # Sagittal: Y horizontal (dy = aff[2,2]), Z vertikal (dz = aff[1,1])
        dy = float(aff[2, 2]); dz = float(aff[1, 1])
        sl = vol[X // 2, :, :]
        img2 = np.flipud(sl.T)
        p2, p98 = np.percentile(img2, [2, 98])
        shown = np.clip((img2 - p2) / (p98 - p2 + 1e-6), 0, 1)
        ex = [0, dy * Y, 0, dz * Z]
        xlabel, ylabel = "Y [mm]", "Z [mm]"
        ttl = title or "Sagittal"
    else:
        raise ValueError("view must be coronal|axial|sagittal")

    # Aspect Ratio der Figur so wählen, dass mm-Verhältnis bewahrt bleibt
    width_mm  = ex[1] - ex[0]
    height_mm = ex[3] - ex[2]
    aspect = height_mm / max(width_mm, 1e-6)
    h_in = float(base_h_in)
    w_in = h_in / max(aspect, 1e-6)

    # Plotten ohne Verzerrungen (aspect = "equal")
    fig = plt.figure(figsize=(w_in, h_in), dpi=150)
    ax = plt.gca()
    ax.imshow(shown, cmap="gray", origin="lower", extent=ex, aspect="equal")
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    ax.set_title(ttl)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"[INFO] Preview ({view}) gespeichert: {out_png} | extent(mm)={width_mm:.1f}×{height_mm:.1f} | fig≈{w_in:.2f}×{h_in:.2f} in")


# ---------------- Hauptprogramm ----------------
def main():
    # ---------- Argumente/CLI ----------
    ap = argparse.ArgumentParser()
    ap.add_argument("--mat", default="ct_proj_stack.mat")
    ap.add_argument("--proj", default="proj.txt")
    ap.add_argument("--out", default="ct_recon_rtk.nii")
    ap.add_argument("--out_ras", default=None,
                    help="optional: zusätzliches kanonisches NIfTI (X,Y,Z + Affine diag(dx,dy,dz))")
    # Zielraster im Volumen (Voxelanzahl und physikalische Fields of View (FOVs) in mm)
    ap.add_argument("--nx", type=int, default=310)
    ap.add_argument("--ny", type=int, default=310)
    ap.add_argument("--nz", type=int, default=512)
    ap.add_argument("--sx", type=float, default=500.0)
    ap.add_argument("--sy", type=float, default=500.0)
    ap.add_argument("--sz", type=float, default=800.0)
    # Optionale Überschreibung der Detektorgröße (falls proj.txt leer oder unvollständig)
    ap.add_argument("--detw", type=float, default=None)
    ap.add_argument("--deth", type=float, default=None)
    # Steuerflags
    ap.add_argument("--no-swap", dest="swap_uv", action="store_false", help="deaktiviert u/v-Tausch")
    ap.add_argument("--reverse_angles", action="store_true")
    ap.add_argument("--angle_offset", type=float, default=0.0)
    # Preview Optionen
    ap.add_argument("--quick_ap_png", type=str, default=None)
    ap.add_argument("--preview_view", choices=["coronal", "axial", "sagittal"], default="coronal")
    ap.add_argument("--base_h_in", type=float, default=6.0)
    ap.set_defaults(swap_uv=True)
    args = ap.parse_args()

    # ----------- Projektionen & Geometrie -----------
    mu_raw, ang = load_mu_stack_any(args.mat)
    print(f"[INFO] mu_raw shape: {mu_raw.shape}")

    # --- Geometrie lesen ---
    G = parse_proj_txt(args.proj)
    DSO = float(G["DSO"])                                                                                       # Source --> Isocenter Abstand
    DTD = float(G["DTD"])                                                                                       # Isocenter --> Detektor Abstand
    SDD = DSO + DTD                                                                                             # Source --> Detektor Abstand
    # Detektorgröße in mm: bevorzugt CLI-Override, dann proj.txt, sonst fallback auf sx/sy
    width_mm = float(args.detw) if args.detw else (float(G["width_mm"]) if G["width_mm"] else args.sx)
    height_mm = float(args.deth) if args.deth else (float(G["height_mm"]) if G["height_mm"] else args.sy)
    nr_txt = int(G["nr"] or 0)
    nc_txt = int(G["nc"] or 0)
    det_shift_u = float(G["det_shift_u"])

    # --- Zielformen bestimmen ---
    if mu_raw.shape[0] in (180, 360):
        nproj, nr, nc = mu_raw.shape
    elif mu_raw.shape[2] in (180, 360):
        nproj = mu_raw.shape[2]; nr, nc = mu_raw.shape[0], mu_raw.shape[1]
    else:
        raise ValueError("Konnte nProj nicht bestimmen (180/360 erwartet).")

    # ---------- swap_uv (sehr wichtiger Stolperstein) ----------
    # wenn aktiv: NUR die Datenachsen (nr <> nc) werden getauscht; width und heigth bleiben an nc/nr gebunden
    if args.swap_uv:
        nr, nc = nc, nr
        print(f"[INFO] swap_uv aktiv → nr={nr}, nc={nc} (width_mm={width_mm}, height_mm={height_mm})")

    # ---------- Winkel ----------
    # falls Winkel fehlen oder Länge nicht passt --> gleichmäßig auf 0..360° verteilen
    if ang is None or len(ang) != nproj:
        ang = np.linspace(0, 360, nproj, endpoint=False, dtype=np.float32)
        print("[WARN] angles_* fehlte – setze 0..360° gleichmäßig.")
    if args.reverse_angles:
        ang = ang[::-1]; print("[INFO] reverse_angles aktiv")
    if args.angle_offset != 0.0:
        ang = (ang + args.angle_offset) % 360.0; print(f"[INFO] angle_offset = {args.angle_offset}°")

    # <<< nur Logging zur Kontrolle, ändert am Verhalten nichts >>>
    print(f"[INFO] angles[0:5]={np.asarray(ang[:5]).round(1)} ... angles[-5:]={np.asarray(ang[-5:]).round(1)} (n={len(ang)})")

    # ---------- mu nach [nProj, nr, nc] bringen ----------
    mu = mu_raw.copy()
    if args.swap_uv:
        # werden die Achsen getauscht, müssen die Dateien entsprechend transponiert werden
        mu = mu.transpose(0, 2, 1)  # (nProj, nr, nc) -> (nProj, nc, nr)
    mu = ensure_axes(mu, nr, nc, nproj).astype(np.float32, copy=False)
    mu = np.ascontiguousarray(mu)

    # Detektor-Pixelgrößen (mm/Pixel) in u (Breite, nc) und v (Höhe, nr)
    # WICHTIG: width_mm bleibt an nc, height_mm an nr gekoppelt (daher nicht bei swap getauscht)
    du = width_mm / nc
    dv = height_mm / nr
    print(f"[INFO] du={du:.4f} mm, dv={dv:.4f} mm, DSO={DSO} SDD={SDD} (u-shift={det_shift_u})")

    # ---------------- ITK-RTK ----------------
    import itk
    from itk import RTK as rtk

    # RTK-Geometrie: füge für jeden Winkel eine Projektion hinzu (DSO/SDD und Detektor-Shift)
    geometry = rtk.ThreeDCircularProjectionGeometry.New()
    for a in ang:
        geometry.AddProjection(DSO, SDD, float(a), det_shift_u, 0.0)

    # Projektionen (Stack) als ITK-Bild (Spacing = (du,dv,1), Origin so, dass Detektorzentrum bei (0,0) liegt)
    proj_img = itk.GetImageFromArray(mu)                                                                            # (z,y,x) = (nProj,nr,nc) in ITK-Speicherreihenfolge
    proj_img.SetSpacing((du, dv, 1.0))
    proj_img.SetOrigin((-width_mm / 2.0, -height_mm / 2.0, 0.0))
    M = itk.Matrix[itk.D, 3, 3](); M.SetIdentity()
    proj_img.SetDirection(M)

    # Zielvolumen initialisieren (Voxelanzahl und physikalische Größe), ITK braucht (z,y,x)-Reihenfolge
    nx, ny, nz = args.nx, args.ny, args.nz
    sx, sy, sz_mm = args.sx, args.sy, args.sz
    dx, dy, dz = sx / nx, sy / ny, sz_mm / nz                                                                       # Voxelgröße in mm
    vol_np = np.zeros((nz, ny, nx), dtype=np.float32)                                                               # (z,y,x) als leeres Volumen
    vol_img = itk.GetImageFromArray(vol_np)
    vol_img.SetSpacing((dx, dy, dz))
    vol_img.SetOrigin((-sx / 2.0, -sy / 2.0, -sz_mm / 2.0))
    vol_img.SetDirection(M)

    print(f"[INFO] Vol: nVoxel=({nx},{ny},{nz}), FOV(mm)=({sx},{sy},{sz_mm}) → voxel(mm)=({dx:.4f},{dy:.4f},{dz:.4f})")

    # FDK Rekonstruktion (RTK)
    fdk = rtk.FDKConeBeamReconstructionFilter.New()
    fdk.SetInput(0, vol_img)                                                                                        # Zielgitter
    fdk.SetInput(1, proj_img)                                                                                       # Projektionen
    fdk.SetGeometry(geometry)                                                                                       # Geometrie (Winkel, Abstände, Shift)
    fdk.Update()

    out_img = fdk.GetOutput()
    out_np = itk.array_from_image(out_img).astype(np.float32)  # (z,y,x)

    # --------- Ausgabe-Layout (spez. für coronal preview) ----------
    # Speichern: (x,z,y) mit Affine diag([dx, dz, dy]) → Koronal vertikal = dz
    vol_xyz = np.transpose(out_np, (2, 0, 1))   # (x,z,y)
    aff = np.diag([dx, dz, dy, 1.0]).astype(np.float32)
    nib.save(nib.Nifti1Image(vol_xyz, aff), args.out)
    print(f"[OK] saved: {args.out}   voxel(mm)=({dx:.4f},{dz:.4f},{dy:.4f})")

    # --------- ZUSATZ: Kanonisches RAS-File für HU-Weiterverarbeitung ----------
    out_ras = args.out_ras
    if out_ras is None:
        base, ext = os.path.splitext(args.out)
        out_ras = base + "_RAS.nii"
    vol_ras = np.transpose(vol_xyz, (0, 2, 1))                                                                      # (x,z,y) --> (x,y,z)
    aff_ras = np.diag([dx, dy, dz, 1.0]).astype(np.float32)
    nib.save(nib.Nifti1Image(vol_ras, aff_ras), out_ras)
    print(f"[OK] saved (canonical RAS for HU): {out_ras}   voxel(mm)=({dx:.4f},{dy:.4f},{dz:.4f})")

    # Optional: Preview-PNG aus dem bestehenden (x,z,y)-Layout speichern (meist coronal view)
    if args.quick_ap_png:
        save_preview_png(args.out, args.quick_ap_png, view=args.preview_view, base_h_in=args.base_h_in)


if __name__ == "__main__":
    main()