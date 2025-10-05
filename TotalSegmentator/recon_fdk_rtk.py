#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FDK-Rekonstruktion mit ITK-RTK aus xCAT/ct_projector-Projektionen (mu_stack),
plus deterministische Preview (coronal/axial/sagittal) mit korrekten mm-Extents.
Kein Auto-Swap, kein Flip-Chaos – alles strikt anatomisch in RAS definiert.
"""

import argparse, re, os
import numpy as np
import nibabel as nib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ------------------- I/O-Helper -------------------
def parse_proj_txt(path):
    txt = open(path, "r", encoding="utf-8", errors="ignore").read()

    def grab(key):
        m = re.search(rf'^\s*([+-]?\d+(?:\.\d+)?)\s*:\s*{re.escape(key)}', txt, re.M | re.I)
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
    """Lädt mu_stack aus .mat (v7/v7.3) und liefert (mu, angles)."""
    mu = None
    ang = None
    try:
        from scipy.io import loadmat
        M = loadmat(mat_path, squeeze_me=True, simplify_cells=True)
        mu = np.asarray(M["mu_stack"], np.float32)
        ang = np.asarray(M.get("angles_deg", None), np.float32) if "angles_deg" in M else None
    except Exception:
        import h5py
        with h5py.File(mat_path, "r") as f:
            mu = np.array(f["mu_stack"], dtype=np.float32)
            if "angles_deg" in f:
                ang = np.array(f["angles_deg"], dtype=np.float32).squeeze()
    if mu.ndim != 3:
        raise ValueError(f"mu_stack hat ndims={mu.ndim}, erwarte 3.")
    # häufig [nr, nc, nProj] → nach [nProj, nr, nc]
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


# ------------------- Preview -------------------
def save_preview_png(nifti_path, out_png, view="coronal", base_h_in=7.0, title=None):
    """
    Anatomisch korrekte Preview (RAS):
      - axial    = vol[:, :, Z//2]
      - coronal  = vol[:, Y//2, :]  → AP-orientiert, Kopf oben
      - sagittal = vol[X//2, :, :]
    """
    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)

    import nibabel as nib, numpy as np, matplotlib.pyplot as plt

    img = nib.load(nifti_path)
    img_ras = nib.as_closest_canonical(img)
    vol = np.asarray(img_ras.dataobj, dtype=np.float32)
    aff = img_ras.affine
    dx, dy, dz = float(aff[0, 0]), float(aff[1, 1]), float(aff[2, 2])
    X, Y, Z = vol.shape
    v = view.lower()

    if v == "coronal":
        # Mittlere Coronalebene, **AP-orientiert**
        sl = vol[:, Y // 2, :]          # (X, Z)
        img2 = np.flipud(sl.T)          # Kopf nach oben
        # Kein np.fliplr mehr → AP (Patient schaut dich an)
        ex = [0, dx * X, 0, dz * Z]
        xlabel, ylabel = "X [mm]", "Z [mm]"
        portrait_boost = 1.6
    elif v == "axial":
        sl = vol[:, :, Z // 2]
        img2 = np.flipud(sl.T)
        ex = [0, dx * X, 0, dy * Y]
        xlabel, ylabel = "X [mm]", "Y [mm]"
        portrait_boost = 1.0
    elif v == "sagittal":
        sl = vol[X // 2, :, :]
        img2 = np.flipud(sl.T)
        ex = [0, dy * Y, 0, dz * Z]
        xlabel, ylabel = "Y [mm]", "Z [mm]"
        portrait_boost = 1.3
    else:
        raise ValueError("view must be coronal|axial|sagittal")

    # Fensterung
    p2, p98 = np.percentile(img2, [2, 98])
    shown = np.clip((img2 - p2) / (p98 - p2 + 1e-6), 0, 1)

    # Hochkantfigur
    width_mm = ex[1] - ex[0]
    height_mm = ex[3] - ex[2]
    aspect = (height_mm / max(width_mm, 1e-6)) * portrait_boost
    h_in = base_h_in
    w_in = h_in / max(aspect, 1e-6)

    fig = plt.figure(figsize=(w_in, h_in), dpi=150)
    ax = plt.gca()
    ax.imshow(shown, cmap="gray", origin="lower", extent=ex, aspect="auto")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title or f"{view.capitalize()} (AP, mid-slice, RAS)")
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"[INFO] Preview ({view}) gespeichert: {out_png} | fig≈{w_in:.2f}×{h_in:.2f} in (hochkant, AP)")
   
   


# ------------------- Hauptprogramm -------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mat", default="ct_proj_stack.mat")
    ap.add_argument("--proj", default="proj.txt")
    ap.add_argument("--out", default="ct_recon_rtk.nii")
    ap.add_argument("--nx", type=int, default=512)
    ap.add_argument("--ny", type=int, default=512)
    ap.add_argument("--nz", type=int, default=512)
    ap.add_argument("--sx", type=float, default=500.0)
    ap.add_argument("--sy", type=float, default=500.0)
    ap.add_argument("--sz", type=float, default=800.0)
    ap.add_argument("--detw", type=float, default=None)
    ap.add_argument("--deth", type=float, default=None)
    ap.add_argument("--no-swap", dest="swap_uv", action="store_false")
    ap.add_argument("--reverse_angles", action="store_true")
    ap.add_argument("--angle_offset", type=float, default=0.0)
    ap.add_argument("--quick_ap_png", type=str, default=None)
    ap.add_argument("--preview_view", choices=["coronal", "axial", "sagittal"], default="coronal")
    ap.add_argument("--base_h_in", type=float, default=6.0)
    ap.set_defaults(swap_uv=True)
    args = ap.parse_args()

    mu_raw, ang = load_mu_stack_any(args.mat)
    print(f"[INFO] mu_raw shape: {mu_raw.shape}")

    G = parse_proj_txt(args.proj)
    DSO, DTD = float(G["DSO"]), float(G["DTD"])
    SDD = DSO + DTD
    width_mm = float(args.detw) if args.detw is not None else (float(G["width_mm"]) if G["width_mm"] else args.sx)
    height_mm = float(args.deth) if args.deth is not None else (float(G["height_mm"]) if G["height_mm"] else args.sy)
    det_shift_u = float(G["det_shift_u"])

    if mu_raw.shape[0] in (180, 360):
        nproj, nr, nc = mu_raw.shape
    elif mu_raw.shape[2] in (180, 360):
        nproj = mu_raw.shape[2]
        nr, nc = mu_raw.shape[:2]
    else:
        raise ValueError("Konnte nProj nicht aus mu_stack ableiten (180/360 erwartet).")

    if args.swap_uv:
        nr, nc = nc, nr
        width_mm, height_mm = height_mm, width_mm
        print(f"[INFO] swap_uv aktiv → nr={nr}, nc={nc}")

    if ang is None or len(ang) != nproj:
        ang = np.linspace(0, 360, nproj, endpoint=False)
        print("[WARN] angles_deg fehlte – setze 0..360° gleichmäßig.")
    if args.reverse_angles:
        ang = ang[::-1]
    if args.angle_offset != 0.0:
        ang = (ang + args.angle_offset) % 360.0

    mu = mu_raw.copy()
    if args.swap_uv:
        mu = mu.transpose(0, 2, 1)
    mu = ensure_axes(mu, nr, nc, nproj)

    du, dv = width_mm / nc, height_mm / nr
    print(f"[INFO] du={du:.4f} dv={dv:.4f} mm, DSO={DSO:.1f} SDD={SDD:.1f}")

    import itk
    from itk import RTK as rtk

    geometry = rtk.ThreeDCircularProjectionGeometry.New()
    for a in ang:
        geometry.AddProjection(DSO, SDD, float(a), det_shift_u, 0.0)

    proj_img = itk.GetImageFromArray(mu)
    proj_img.SetSpacing((du, dv, 1.0))
    proj_img.SetOrigin((-width_mm / 2.0, -height_mm / 2.0, 0.0))
    M = itk.Matrix[itk.D, 3, 3]()
    M.SetIdentity()
    proj_img.SetDirection(M)

    nx, ny, nz = args.nx, args.ny, args.nz
    sx, sy, sz_mm = args.sx, args.sy, args.sz
    dx, dy, dz = sx / nx, sy / ny, sz_mm / nz
    vol_np = np.zeros((nz, ny, nx), dtype=np.float32)
    vol_img = itk.GetImageFromArray(vol_np)
    vol_img.SetSpacing((dx, dy, dz))
    vol_img.SetOrigin((-sx / 2.0, -sy / 2.0, -sz_mm / 2.0))
    vol_img.SetDirection(M)

    fdk = rtk.FDKConeBeamReconstructionFilter.New()
    fdk.SetInput(0, vol_img)
    fdk.SetInput(1, proj_img)
    fdk.SetGeometry(geometry)
    fdk.Update()

    out_img = fdk.GetOutput()
    out_np = itk.GetArrayFromImage(out_img).astype(np.float32)

    vol_xyz = np.transpose(out_np, (2, 0, 1))
    aff = np.diag([dx, dz, dy, 1.0]).astype(np.float32)
    nii = nib.Nifti1Image(vol_xyz, aff)
    nii.header.set_xyzt_units('mm')
    nib.save(nii, args.out)
    print(f"[OK] saved: {args.out}   voxel(mm)=({dx:.4f},{dz:.4f},{dy:.4f})")

    if args.quick_ap_png:
        save_preview_png(args.out, args.quick_ap_png, view=args.preview_view, base_h_in=args.base_h_in)


if __name__ == "__main__":
    main()