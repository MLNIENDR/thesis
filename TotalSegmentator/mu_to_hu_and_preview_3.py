#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
µ→HU-Kalibrierung + robuste Außenluft-Entfernung + optionale Keep-Box
+ (neu) physisches Cropping auf Keep-Box + HU-Previews + optionales iso-Resample.

Datei: mu_to_hu_and_preview_3.py
"""

import os, re, sys, argparse
import numpy as np
import nibabel as nib
from scipy.ndimage import (
    binary_dilation, binary_closing, label,
    generate_binary_structure, gaussian_filter
)
from nibabel.processing import resample_to_output

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------- Helpers ----------
def mm_to_idx(mm, spacing):  # mm → Voxelindex (gerundet)
    return int(round(mm / float(spacing)))

def parse_mm(token, total_mm):
    """
    Erlaubt Eingaben wie 'x/2', 'y/2', 'z/2' oder Brüche 'a/b'.
    total_mm: (Lx_mm, Ly_mm, Lz_mm) für die jeweilige Achse.
    """
    token = token.strip().lower()
    if token in ("x/2","y/2","z/2"):
        axis = token[0]
        return dict(x=total_mm[0]/2, y=total_mm[1]/2, z=total_mm[2]/2)[axis]
    m = re.fullmatch(r'(\d+(?:\.\d+)?)/(\d+(?:\.\d+)?)', token)
    return float(token) if not m else total_mm[0]*(float(m.group(1))/float(m.group(2)))

def clamp_idx(a, lo, hi):
    return int(max(lo, min(hi, round(a))))

def apply_keep_box_mm(hu, aff, keep):
    """
    Klassisches Verhalten: alles außerhalb Keep-Box → -1024 HU.
    keep: [Y0 Y1 Z0 Z1 X0 X1] (mm) auf RAS (X,Y,Z) bezogen.
    """
    dx, dy, dz = float(aff[0,0]), float(aff[1,1]), float(aff[2,2])
    X,Y,Z = hu.shape
    if len(keep) not in (4,6):
        raise ValueError("--keep_box_mm erwartet 4 oder 6 Werte (Y0 Y1 Z0 Z1 [X0 X1])")
    Y0,Y1,Z0,Z1 = keep[:4]
    if len(keep)==6: X0,X1 = keep[4:6]
    else:            X0,X1 = (0.0, dx*X)

    ix0 = clamp_idx(X0/dx, 0, X); ix1 = clamp_idx(X1/dx, 0, X)
    iy0 = clamp_idx(Y0/dy, 0, Y); iy1 = clamp_idx(Y1/dy, 0, Y)
    iz0 = clamp_idx(Z0/dz, 0, Z); iz1 = clamp_idx(Z1/dz, 0, Z)

    keep_mask = np.zeros_like(hu, bool)
    keep_mask[ix0:ix1, iy0:iy1, iz0:iz1] = True
    hu[~keep_mask] = -1024.0
    return hu

def crop_to_box_mm(hu, aff, keep, pad_mm=0.0):
    """
    NEU: physisches Cropping auf Keep-Box (+ optionales Padding in mm).
    keep: [Y0 Y1 Z0 Z1] oder [Y0 Y1 Z0 Z1 X0 X1] in mm (RAS).
    Rückgabe: (hu_cropped, aff_cropped)
    """
    dx, dy, dz = float(aff[0,0]), float(aff[1,1]), float(aff[2,2])
    X,Y,Z = hu.shape

    if len(keep) not in (4,6):
        raise ValueError("--keep_box_mm erwartet 4 oder 6 Werte (Y0 Y1 Z0 Z1 [X0 X1])")

    Y0,Y1,Z0,Z1 = keep[:4]
    if len(keep)==6: X0,X1 = keep[4:6]
    else:            X0,X1 = (0.0, dx*X)

    # Padding
    X0 -= pad_mm; X1 += pad_mm
    Y0 -= pad_mm; Y1 += pad_mm
    Z0 -= pad_mm; Z1 += pad_mm

    ix0 = max(0, int(np.floor(X0/dx))); ix1 = min(X, int(np.ceil(X1/dx)))
    iy0 = max(0, int(np.floor(Y0/dy))); iy1 = min(Y, int(np.ceil(Y1/dy)))
    iz0 = max(0, int(np.floor(Z0/dz))); iz1 = min(Z, int(np.ceil(Z1/dz)))

    hu_c = hu[ix0:ix1, iy0:iy1, iz0:iz1].copy()

    # Affine verschieben: neuer Ursprung = alter + (ix0,iy0,iz0)
    aff_c = aff.copy()
    aff_c[:3,3] = aff[:3,3] + aff[:3,:3].dot(np.array([ix0, iy0, iz0], float))
    return hu_c, aff_c


# ---------- Außenluft-Entfernung (robust) ----------
def remove_outside_air_keep_inside(hu, air_thr=-750.0, tissue_thr=-550.0, close_iter=2):
    """
    Entfernt Außenluft, behält Innenluft (Lunge/Darm).
    - 'tissue' = hu > tissue_thr → pro axialem Slice schließen (Barriere),
    - 'passable' = (hu < air_thr) & (~tissue_closed),
    - Flood-Fill von den Rändern nur durch 'passable'.
    """
    hu = hu.copy()
    X, Y, Z = hu.shape
    outside = np.zeros((X, Y, Z), dtype=bool)

    for y in range(Y):  # axialer Index entspricht (X,Z)-Slice
        sl = hu[:, y, :]
        tissue = sl > tissue_thr
        tissue_closed = binary_closing(tissue, structure=np.ones((3,3), bool), iterations=close_iter)
        passable = (sl < air_thr) & (~tissue_closed)

        seeds = np.zeros_like(passable, bool)
        seeds[0,:]=True; seeds[-1,:]=True; seeds[:,0]=True; seeds[:,-1]=True

        reach = seeds & passable
        last = -1
        while True:
            neigh = np.zeros_like(reach, bool)
            neigh[:-1,:] |= reach[1:,:]
            neigh[1: ,:] |= reach[:-1,:]
            neigh[:, :-1] |= reach[:, 1:]
            neigh[:, 1: ] |= reach[:, :-1]
            new_reach = (neigh | reach) & passable
            s = int(new_reach.sum())
            if s==last: break
            reach = new_reach; last = s

        outside[:, y, :] = reach

    outside = binary_dilation(outside, iterations=1)
    hu[outside] = -1024.0
    return hu


# ---------- µ -> HU ----------
def calibrate_to_hu(in_path, out_path, anchors, radius_mm=4.0,
                    clip=(-1024,2000), smooth_sigma=0.0, resample_mm=None,
                    air_thr=-750.0, tissue_thr=-550.0, close_iter=2,
                    keep_box_mm=None, crop_keep_box=False, crop_pad_mm=0.0):
    """
    - Lineare Kalibrierung HU = a*µ + b anhand von Ankern (Median über YZ-Kreis, Radius=radius_mm).
    - Außenluft-Entfernung (robust).
    - optional: Glättung, Keep-Box → (entweder klassisch auf -1024) oder NEU physisches Cropping.
    """
    print(f"[INFO] Lade: {in_path}")
    img = nib.load(in_path)                 # µ in RAS (X,Y,Z)
    vol = np.asarray(img.dataobj).astype(np.float32)
    aff = img.affine
    dx, dy, dz = float(aff[0,0]), float(aff[1,1]), float(aff[2,2])
    X, Y, Z = vol.shape
    size_mm = (dx*X, dy*Y, dz*Z)

    # -------- Anker erfassen (Median über Kreis in YZ-Ebene bei festem X) --------
    mus, hus = [], []
    for a in anchors:
        coord, hu_t = a.split(":")
        xs, ys, zs = [t.strip() for t in coord.split(",")]
        x_mm = parse_mm(xs, size_mm)           # absolute mm-Position
        y_mm = parse_mm(ys, (size_mm[1],)*3)
        z_mm = parse_mm(zs, (size_mm[2],)*3)
        ix, iy, iz = mm_to_idx(x_mm, dx), mm_to_idx(y_mm, dy), mm_to_idx(z_mm, dz)

        r_vox = max(1, int(round(float(radius_mm) / min(dy, dz))))
        yy, zz = np.meshgrid(
            np.arange(max(0, iy - r_vox), min(Y, iy + r_vox + 1)),
            np.arange(max(0, iz - r_vox), min(Z, iz + r_vox + 1)),
            indexing="ij"
        )
        mask = (yy - iy)**2 + (zz - iz)**2 <= r_vox**2
        mu_med = float(np.median(vol[ix, yy[mask], zz[mask]]))
        mus.append(mu_med); hus.append(float(hu_t))
        print(f"[ANCHOR] {a} → idx=({ix},{iy},{iz}) r={r_vox} → µ_median={mu_med:.6f}")

    if len(mus) < 2:
        print("[ERR] Mindestens 2 Anker nötig!", file=sys.stderr); sys.exit(1)

    # Least Squares: [mu_i, 1] * [a,b]^T ≈ HU_i
    A = np.stack([mus, np.ones(len(mus), np.float32)], axis=1)
    x_fit, *_ = np.linalg.lstsq(A, np.array(hus, np.float32), rcond=None)
    a, b = float(x_fit[0]), float(x_fit[1])
    rmse = float(np.sqrt(np.mean((a*np.array(mus)+b - np.array(hus))**2)))
    print(f"[FIT] HU = {a:.6f} * µ + {b:.2f} | RMSE={rmse:.2f} HU")

    # anwenden
    hu = a*vol + b

    # Außenluft robust entfernen (Innenluft bleibt)
    hu = remove_outside_air_keep_inside(hu, air_thr=air_thr, tissue_thr=tissue_thr, close_iter=close_iter)

    # optional glätten
    if smooth_sigma and smooth_sigma > 0:
        hu = gaussian_filter(hu, sigma=float(smooth_sigma))

    # Keep-Box: EITHER klassisch od. physisches Cropping
    if keep_box_mm is not None:
        if crop_keep_box:
            print(f"[INFO] Cropping to keep-box (pad={crop_pad_mm:.1f} mm) …")
            hu, aff = crop_to_box_mm(hu, aff, keep_box_mm, pad_mm=float(crop_pad_mm))
            # Update lokaler Geometriedaten
            dx, dy, dz = float(aff[0,0]), float(aff[1,1]), float(aff[2,2])
            X, Y, Z = hu.shape
        else:
            print("[INFO] Applying keep-box as air (-1024 HU) …")
            hu = apply_keep_box_mm(hu, aff, keep_box_mm)

    # Clip & Speichern
    hu_min, hu_max = (clip[0], clip[1]) if isinstance(clip,(list,tuple)) and len(clip)==2 else (-1024,2000)
    hu_f32 = np.clip(hu, hu_min, hu_max).astype(np.float32)
    out_img = nib.Nifti1Image(hu_f32, aff)
    out_img.header.set_qform(aff, code=1); out_img.header.set_sform(aff, code=1)
    nib.save(out_img, out_path)
    print(f"[DONE] gespeichert: {out_path}  | shape={hu_f32.shape}  spacing≈({dx:.3f},{dy:.3f},{dz:.3f}) mm")

    # optional isotrop resample
    resampled_path = None
    if resample_mm and resample_mm > 0:
        print(f"[INFO] Resample auf isotrop {resample_mm:.2f} mm …")
        out_float = nib.Nifti1Image(hu_f32.astype(np.float32), aff)
        rs = resample_to_output(out_float, voxel_sizes=(resample_mm,)*3, order=1)
        rs_data = np.clip(np.asarray(rs.dataobj), hu_min, hu_max).astype(np.int16)
        rs_img = nib.Nifti1Image(rs_data, rs.affine)
        rs_img.header.set_qform(rs.affine, code=1); rs_img.header.set_sform(rs.affine, code=1)
        root, ext = os.path.splitext(out_path)
        if ext.lower()==".gz": root, _ = os.path.splitext(root)
        resampled_path = root + f"_iso{resample_mm:.1f}mm.nii"
        nib.save(rs_img, resampled_path)
        print(f"[DONE] resampled: {resampled_path} | spacing≈({resample_mm},{resample_mm},{resample_mm}) mm")

    return resampled_path if resampled_path else out_path


# ---------- Preview ----------
def render_slice_png(nifti_path, out_png, view="coronal", base_h_in=7.0,
                     hu_window=(-1000,1000), percentile=None,
                     cmap="turbo", coronal_orient="AP", three_views=False):
    img = nib.load(nifti_path)
    img_ras = nib.as_closest_canonical(img)
    vol = np.asarray(img_ras.dataobj, dtype=np.float32)
    aff = img_ras.affine
    X, Y, Z = vol.shape
    dx, dy, dz = float(aff[0,0]), float(aff[1,1]), float(aff[2,2])

    def get_view(v):
        v=v.lower()
        if v=="coronal":
            sl = vol[X//2,:,:]; im = np.flipud(sl)
            if coronal_orient.upper()=="AP": im = np.fliplr(im)
            extent=[0, dz*Z, 0, dy*Y]; xlabel, ylabel, ttl = "Z [mm]","Y [mm]",f"Coronal ({coronal_orient})"
        elif v=="axial":
            sl = vol[:,Y//2,:]; im=np.flipud(sl.T)
            extent=[0, dx*X, 0, dz*Z]; xlabel, ylabel, ttl = "X [mm]","Z [mm]","Axial"
        elif v=="sagittal":
            sl = vol[:,:,Z//2]; im=np.flipud(sl.T)
            extent=[0, dx*X, 0, dy*Y]; xlabel, ylabel, ttl = "X [mm]","Y [mm]","Sagittal"
        else: raise ValueError("view must be coronal|axial|sagittal")
        return im, extent, xlabel, ylabel, ttl

    def window_limits(arr):
        if percentile:
            p1,p99 = np.percentile(arr, percentile); return float(p1), float(p99)
        return hu_window

    if not three_views:
        img2, ex, xlabel, ylabel, ttl = get_view(view)
        width_mm, height_mm = ex[1]-ex[0], ex[3]-ex[2]
        aspect = height_mm/max(width_mm,1e-6)
        h_in=base_h_in; w_in=h_in/max(aspect,1e-6)
        vmin, vmax = window_limits(img2)

        fig, ax = plt.subplots(figsize=(w_in,h_in), dpi=150)
        im = ax.imshow(img2, origin="lower", extent=ex, cmap=cmap,
                       vmin=vmin, vmax=vmax, aspect="equal")
        ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.set_title(ttl)
        cb = fig.colorbar(im, ax=ax, shrink=0.8); cb.set_label("Hounsfield Units [HU]")
        plt.tight_layout(); os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
        plt.savefig(out_png, dpi=150); plt.close(fig)
        print(f"[OK] Preview saved: {out_png} | window=[{vmin:.0f},{vmax:.0f}]")
        return

    views=["coronal","axial","sagittal"]
    metas=[get_view(v) for v in views]
    all_vals=np.concatenate([m[0].ravel() for m in metas])
    vmin,vmax = (np.percentile(all_vals, percentile) if percentile else hu_window)
    h_in=base_h_in; widths=[]
    for _,ex,*_ in metas:
        width_mm, height_mm = ex[1]-ex[0], ex[3]-ex[2]
        widths.append(h_in/max(height_mm/max(width_mm,1e-6),1e-6))
    w_in=sum(widths)+2.0
    fig,axes=plt.subplots(1,3, figsize=(w_in,h_in), dpi=150)
    for ax,(img2,ex,xlabel,ylabel,ttl) in zip(axes, metas):
        im=ax.imshow(img2, origin="lower", extent=ex, cmap=cmap, vmin=vmin, vmax=vmax, aspect="equal")
        ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.set_title(ttl)
    cbar=fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.8); cbar.set_label("Hounsfield Units [HU]")
    plt.tight_layout(); os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
    plt.savefig(out_png, dpi=150); plt.close(fig)
    print(f"[OK] Preview (3) saved: {out_png} | window=[{vmin:.0f},{vmax:.0f}]")


# ---------- CLI ----------
def main():
    base_dir = os.path.expanduser("~/projects/thesis/TotalSegmentator")
    default_in  = os.path.join(base_dir, "runs_360", "ct_recon_rtk360_RAS.nii")
    default_out = os.path.join(base_dir, "runs_360", "ct_recon_rtk360_HU.nii")
    default_png = os.path.join(base_dir, "runs_360", "ct_recon_rtk360_HU_coronal.png")

    ap = argparse.ArgumentParser(description="µ→HU + Outside-Air Removal + Keep-Box (+Crop) + Preview")
    # IO
    ap.add_argument("--in",  dest="infile",  default=default_in)
    ap.add_argument("--out", dest="outfile", default=default_out)
    ap.add_argument("--png", dest="pngfile", default=default_png)
    # anchors
    ap.add_argument("--anchor", action="append",
                    default=[],
                    help='Format "x_mm,y_mm,z_mm:HU" (mehrfach)')
    ap.add_argument("--radius_mm", type=float, default=4.0,
                    help="Anker-Kreisradius (mm) in der YZ-Ebene")
    ap.add_argument("--clip", nargs=2, type=float, default=[-1024,2000])
    # processing
    ap.add_argument("--smooth_sigma", type=float, default=0.0)
    ap.add_argument("--resample_mm", type=float, default=None)
    ap.add_argument("--air_thr", type=float, default=-750.0)
    ap.add_argument("--tissue_thr", type=float, default=-550.0)
    ap.add_argument("--close_iter", type=int, default=2)
    ap.add_argument("--keep_box_mm", nargs="+", type=float,
                    help="Keep-Box (mm): Y0 Y1 Z0 Z1 [X0 X1]")
    ap.add_argument("--crop_keep_box", action="store_true",
                    help="NEU: statt außerhalb -1024 zu setzen → physisch croppen")
    ap.add_argument("--crop_pad_mm", type=float, default=0.0,
                    help="Padding um Keep-Box beim Cropping (mm)")
    # preview
    ap.add_argument("--preview", action="store_true")
    ap.add_argument("--view", choices=["coronal","axial","sagittal"], default="coronal")
    ap.add_argument("--percentile", nargs=2, type=float)
    ap.add_argument("--hu_window", nargs=2, type=float, default=[-1000,1000])
    ap.add_argument("--cmap", default="turbo")
    ap.add_argument("--base_h_in", type=float, default=7.0)
    ap.add_argument("--tri", action="store_true")
    ap.add_argument("--coronal_orient", choices=["AP","PA"], default="AP")

    args = ap.parse_args()

    hu_path = calibrate_to_hu(
        in_path=args.infile, out_path=args.outfile, anchors=args.anchor,
        radius_mm=args.radius_mm, clip=tuple(args.clip),
        smooth_sigma=args.smooth_sigma, resample_mm=args.resample_mm,
        air_thr=args.air_thr, tissue_thr=args.tissue_thr, close_iter=args.close_iter,
        keep_box_mm=args.keep_box_mm, crop_keep_box=bool(args.crop_keep_box),
        crop_pad_mm=float(args.crop_pad_mm),
    )

    if args.preview or args.tri:
        png = args.pngfile if not args.tri else args.pngfile.replace(".png","_preview3.png")
        render_slice_png(
            nifti_path=hu_path, out_png=png, view=args.view, base_h_in=args.base_h_in,
            hu_window=tuple(args.hu_window),
            percentile=tuple(args.percentile) if args.percentile else None,
            cmap=args.cmap, coronal_orient=args.coronal_orient, three_views=bool(args.tri)
        )

if __name__ == "__main__":
    main()