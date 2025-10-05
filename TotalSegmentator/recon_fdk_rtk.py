#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FDK-Rekonstruktion mit ITK-RTK aus xCAT/ct_projector-Projektionen (mu_stack),
inkl. optionalem Quick-Preview als **Coronal-AP** (mittlerer Y-Slice) mit korrekten mm-Extents.

Eingang:
  - ct_proj_stack.mat:
      mu_stack: [nProj, nr, nc] oder [nr, nc, nProj]  (wird erkannt)
      angles_deg (optional) in Grad; sonst gleichmäßig 0..360°
  - proj.txt: width_mm, height_mm, num_rows (nr), num_channels (nc),
              distance_to_source (DSO), distance_to_detector (DTD), detector_shift (u)

Defaults für unsere Daten:
  - u/v werden **standardmäßig getauscht** (swap_uv), da das Artefakte behebt.
    Abschaltbar via --no-swap.

Optionen:
  --no-swap           : deaktiviert u/v-Tausch
  --reverse_angles    : kehrt Winkelreihenfolge um
  --angle_offset a    : addiert 'a' Grad auf alle Winkel
  --detw / --deth     : überschreibt width_mm / height_mm
  --quick_ap_png PATH : speichert einen Coronal-AP-Preview (mittlerer Y-Slice) nach PATH (Agg/headless)
  --no-flip180        : deaktiviert die 180°-Flip-Heuristik für AP
  --no-lrflip         : deaktiviert den L/R-Flip für AP

Beispiel:
  python recon_fdk_rtk.py --out ct_recon_rtk.nii --quick_ap_png ts_preview/reko_coronal_AP.png
"""

import argparse, re, os
import numpy as np
import nibabel as nib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------- I/O-Helper ----------------
def parse_proj_txt(path):
    txt = open(path, "r", encoding="utf-8", errors="ignore").read()
    def grab(key):
        # akzeptiert sowohl "width" als auch "width(mms)"
        m = re.search(rf'^\s*([+-]?\d+(?:\.\d+)?)\s*:\s*{re.escape(key)}', txt, re.M|re.I)
        return float(m.group(1)) if m else None
    geom = {
        "width_mm":  grab("width(mms)")  or grab("width"),
        "height_mm": grab("height(mms)") or grab("height"),
        "nr": int(grab("num_rows") or 0),
        "nc": int(grab("num_channels") or 0),
        "DSO": grab("distance_to_source(mms)")    or grab("distance_to_source"),
        "DTD": grab("distance_to_detector(mms)")  or grab("distance_to_detector"),
        "det_shift_u": grab("detector_shift") or 0.0,
    }
    assert geom["DSO"] is not None and geom["DTD"] is not None, "DSO/DTD fehlen in proj.txt"
    return geom

def load_mu_stack_any(mat_path):
    """Lädt mu_stack aus .mat (v7 / v7.3) und liefert (mu, angles) in numpy."""
    mu = None; ang = None
    try:
        from scipy.io import loadmat
        M = loadmat(mat_path, squeeze_me=True, simplify_cells=True)
        mu = np.asarray(M["mu_stack"], np.float32)
        ang = np.asarray(M.get("angles_deg", None), np.float32) if "angles_deg" in M else None
    except Exception:
        import h5py
        with h5py.File(mat_path, "r") as f:
            mu = np.array(f["mu_stack"], dtype=np.float32)
            if "angles_deg" in f: ang = np.array(f["angles_deg"], dtype=np.float32).squeeze()
    if mu.ndim != 3:
        raise ValueError(f"mu_stack hat ndims={mu.ndim}, erwarte 3.")
    # häufig [nr, nc, nProj] → nach [nProj, nr, nc]
    if mu.shape[0] not in (180, 360):
        mu = np.transpose(mu, (2, 0, 1))
    return mu.astype(np.float32, copy=False), (None if ang is None else ang.astype(np.float32, copy=False))

def ensure_axes(mu, nr, nc, nproj):
    """Erzwingt [nProj, nr, nc] – probiert Permutationen, sonst Fehler."""
    want = (nproj, nr, nc)
    if mu.shape == want: return mu
    from itertools import permutations
    for p in permutations((0,1,2)):
        t = np.transpose(mu, p)
        if t.shape == want:
            print(f"[FIX] permuted mu axes {p} -> {t.shape}")
            return t
    raise ValueError(f"Projektdaten-Form {mu.shape} passt nicht zu {want}.")

# ---------------- Quick-Preview (Coronal-AP) ----------------
def save_quick_coronal_ap_png(nifti_path, out_png, base_h_in=6.0, flip180=True, lrflip=True):
    """
    Nimmt die rekonstruierte NIfTI (x,y,z), wählt den mittleren Y-Slice (coronal),
    stellt **X horizontal** und **Z vertikal** mit korrekten mm-Extents dar (keine Stauchung).
    Heuristik für AP: 180°-Flip + L/R-Flip standardmäßig aktiv.
    """
    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
    img = nib.load(nifti_path)
    vol = np.asarray(img.dataobj, dtype=np.float32)   # (x,y,z)
    dx, dy, dz = float(img.affine[0,0]), float(img.affine[1,1]), float(img.affine[2,2])

    x, y, z = vol.shape
    ymid = y // 2
    cor = vol[:, ymid, :]            # (x, z)

    # Fensterung/Normierung fürs Preview (robust, nur fürs Bild)
    v = cor
    p2, p98 = np.percentile(v, [2, 98])
    if p98 > p2:
        v = np.clip((v - p2) / (p98 - p2), 0, 1)
    else:
        v = (v - v.min()) / (v.ptp() + 1e-6)

    # Heuristik für AP-Orientierung
    if flip180:
        v = np.flipud(np.fliplr(v))  # 180°
    if lrflip:
        v = np.fliplr(v)             # L/R umdrehen → AP statt PA

    # mm-Extents: Breite=X (dx*x), Höhe=Z (dz*z)
    extent = [0, dx * x, 0, dz * z]
    # Seitenverhältnis passend wählen
    w_in = base_h_in * (extent[1] / extent[3])

    plt.figure(figsize=(w_in, base_h_in), dpi=150)
    plt.imshow(v.T, cmap="gray", origin="lower", extent=extent, aspect='equal')
    plt.xlabel("X [mm]"); plt.ylabel("Z [mm]")
    plt.title("Coronal AP (mid-Y)")
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()
    print(f"[INFO] Quick Coronal-AP PNG gespeichert: {out_png}")

# ---------------- Hauptprogramm ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mat", default="ct_proj_stack.mat")
    ap.add_argument("--proj", default="proj.txt")
    ap.add_argument("--out",  default="ct_recon_rtk.nii")
    # Volumengitter (Ausgabe)
    ap.add_argument("--nx", type=int, default=512)
    ap.add_argument("--ny", type=int, default=512)
    ap.add_argument("--nz", type=int, default=512)
    ap.add_argument("--sx", type=float, default=500.0)
    ap.add_argument("--sy", type=float, default=500.0)
    ap.add_argument("--sz", type=float, default=800.0)
    # Geometrie-Override
    ap.add_argument("--detw", type=float, default=None)
    ap.add_argument("--deth", type=float, default=None)
    # Winkel/Orientierung
    ap.add_argument("--no-swap", dest="swap_uv", action="store_false", help="deaktiviert u/v-Tausch (Default: an)")
    ap.add_argument("--reverse_angles", action="store_true")
    ap.add_argument("--angle_offset", type=float, default=0.0)
    # Quick-Preview
    ap.add_argument("--quick_ap_png", type=str, default=None)
    ap.add_argument("--base_h_in", type=float, default=6.0)
    ap.add_argument("--no-flip180", dest="flip180", action="store_false")
    ap.add_argument("--no-lrflip", dest="lrflip", action="store_false")
    ap.set_defaults(swap_uv=True, flip180=True, lrflip=True)
    args = ap.parse_args()

    # --- Projektionen laden ---
    mu_raw, ang = load_mu_stack_any(args.mat)     # [?, ?, ?]
    print(f"[INFO] mu_raw shape: {mu_raw.shape}")

    # --- Geometrie lesen ---
    G = parse_proj_txt(args.proj)
    DSO = float(G["DSO"]); DTD = float(G["DTD"]); SDD = DSO + DTD
    width_mm  = float(args.detw) if args.detw is not None else (float(G["width_mm"])  if G["width_mm"]  is not None else args.sx)
    height_mm = float(args.deth) if args.deth is not None else (float(G["height_mm"]) if G["height_mm"] is not None else args.sy)
    nr_txt = int(G["nr"] or 0); nc_txt = int(G["nc"] or 0)
    det_shift_u = float(G["det_shift_u"])

    # --- Zielformen bestimmen ---
    if mu_raw.shape[0] in (180, 360):
        nproj = int(mu_raw.shape[0]); nr, nc = int(mu_raw.shape[1]), int(mu_raw.shape[2])
    elif mu_raw.shape[2] in (180, 360):
        nproj = int(mu_raw.shape[2]); nr, nc = int(mu_raw.shape[0]), int(mu_raw.shape[1])
    else:
        raise ValueError("Konnte nProj nicht aus mu_stack ableiten (180/360 erwartet).")
    print(f"[INFO] nProj={nproj}, nr={nr}, nc={nc}  (txt: nr={nr_txt}, nc={nc_txt})")

    # --- swap_uv standardmäßig aktiv (abschaltbar via --no-swap) ---
    if args.swap_uv:
        nr, nc = nc, nr
        width_mm, height_mm = height_mm, width_mm
        print(f"[INFO] swap_uv aktiv -> nr={nr}, nc={nc}, width={width_mm}, height={height_mm}")

    # --- Winkel ---
    if ang is None or len(ang) != nproj:
        ang = np.linspace(0, 360, nproj, endpoint=False, dtype=np.float32)
        print("[WARN] angles_deg fehlte/uneindeutig – setze gleichmäßig 0..360°.")
    else:
        ang = np.asarray(ang, dtype=np.float32)
    if args.reverse_angles:
        ang = ang[::-1]; print("[INFO] reverse_angles aktiv")
    if args.angle_offset != 0.0:
        ang = (ang + args.angle_offset) % 360.0
        print(f"[INFO] angle_offset = {args.angle_offset}°")

    # --- mu nach [nProj, nr, nc] bringen (unter Beachtung swap_uv) ---
    mu = mu_raw.copy()
    if args.swap_uv:
        # egal, wie mu_raw gerade steht: letzten beiden Achsen tauschen
        if mu.shape[0] in (180, 360):
            mu = mu.transpose(0,2,1)
        else:
            mu = mu.transpose(2,1,0)  # wird in ensure_axes finalisiert
    mu = ensure_axes(mu, nr, nc, nproj).astype(np.float32, copy=False)
    mu = np.ascontiguousarray(mu)

    # Pixelgrößen am Detektor
    du = float(width_mm)  / float(nc)
    dv = float(height_mm) / float(nr)
    print(f"[INFO] DSO={DSO:.1f} mm, SDD={SDD:.1f} mm, du={du:.4f} mm, dv={dv:.4f} mm, det_shift_u={det_shift_u} mm")
    print(f"[INFO] Detektor(mm): width={width_mm:.1f}, height={height_mm:.1f}")

    # ---------------- ITK-RTK ----------------
    import itk
    from itk import RTK as rtk

    # Geometrie
    geometry = rtk.ThreeDCircularProjectionGeometry.New()
    for a in ang:
        geometry.AddProjection(DSO, SDD, float(a), det_shift_u, 0.0)

    # Proj-Stack: numpy [z,y,x] = [nProj, nr, nc]
    proj_img = itk.GetImageFromArray(mu)      # F32, interpretiert (z,y,x)
    proj_img.SetSpacing((du, dv, 1.0))        # (x,y,z)
    proj_img.SetOrigin((-width_mm/2.0, -height_mm/2.0, 0.0))
    M = itk.Matrix[itk.D,3,3](); M.SetIdentity(); proj_img.SetDirection(M)

    # Sanity: ITK muss Z==nProj sehen
    sz = proj_img.GetLargestPossibleRegion().GetSize()
    assert int(sz[2]) == int(nproj), f"ITK sieht {int(sz[2])} Z-Slices, Geometrie {nproj}!"

    # Ausgangs-Volumen
    nx, ny, nz = int(args.nx), int(args.ny), int(args.nz)
    sx, sy, sz_mm = float(args.sx), float(args.sy), float(args.sz)
    dx, dy, dz = sx/nx, sy/ny, sz_mm/nz

    vol_np = np.zeros((nz, ny, nx), dtype=np.float32)   # (z,y,x)
    vol_img = itk.GetImageFromArray(vol_np)
    vol_img.SetSpacing((dx, dy, dz))
    vol_img.SetOrigin((-sx/2.0, -sy/2.0, -sz_mm/2.0))
    vol_img.SetDirection(M)

    print(f"[INFO] Vol: nVoxel=({nx},{ny},{nz}), FOV(mm)=({sx},{sy},{sz_mm}) → voxel(mm)=({dx:.4f},{dy:.4f},{dz:.4f})")

    # FDK
    fdk = rtk.FDKConeBeamReconstructionFilter.New()
    fdk.SetInput(0, vol_img)
    fdk.SetInput(1, proj_img)
    fdk.SetGeometry(geometry)
    fdk.Update()

    out_img = fdk.GetOutput()
    try:
        out_np  = itk.array_from_image(out_img).astype(np.float32)  # (z,y,x)
    except Exception:
        out_np  = itk.GetArrayFromImage(out_img).astype(np.float32)

    # Als NIfTI speichern (x,y,z)
    vol_xyz = np.transpose(out_np, (2,1,0))
    aff = np.diag([dx, dy, dz, 1.0]).astype(np.float32)
    nii = nib.Nifti1Image(vol_xyz, aff)
    nii.header.set_xyzt_units('mm')
    nib.save(nii, args.out)
    print(f"[OK] saved: {args.out}   voxel(mm)=({dx:.4f},{dy:.4f},{dz:.4f})")

    # Optionaler Quick-Preview (Coronal-AP)
    if args.quick_ap_png:
        save_quick_coronal_ap_png(args.out, args.quick_ap_png, base_h_in=args.base_h_in,
                                  flip180=args.flip180, lrflip=args.lrflip)

if __name__ == "__main__":
    main()