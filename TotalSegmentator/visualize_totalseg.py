#!/usr/bin/env python3
import os
import argparse
import numpy as np
import nibabel as nib
import matplotlib
matplotlib.use("Agg")  # headless ok; nimm weg, wenn du lokal anzeigen willst
import matplotlib.pyplot as plt
from scipy.ndimage import binary_dilation, binary_erosion

def load_slice(vol, axis, idx=None):
    """
    vol in (X, Y, Z) wie von nibabel.get_fdata().
    axis: 0=sagittal (X), 1=coronal (Y), 2=axial (Z).
    """
    if idx is None:
        idx = vol.shape[axis] // 2
    if axis == 0:   # sagittal: X-Konst
        sl = vol[idx, :, :]           # (Y, Z)
        spac = ("Y", "Z")
    elif axis == 1: # coronal: Y-Konst
        sl = vol[:, idx, :]           # (X, Z)
        spac = ("X", "Z")
    elif axis == 2: # axial: Z-Konst
        sl = vol[:, :, idx]           # (X, Y)
        spac = ("X", "Y")
    else:
        raise ValueError("axis muss 0 (sagittal), 1 (koronal) oder 2 (axial) sein")
    return sl, idx, spac

def get_extent(zooms_xyz, shape_2d, spac_names):
    """
    Erzeuge extent in physikalischen Einheiten für plt.imshow, damit aspect='equal' korrekt ist.
    zooms_xyz: (dx, dy, dz) aus NIfTI-Header.
    shape_2d: (W, H) = sl.shape in Pixel.
    spac_names: Tupel aus {"X","Y","Z"} für die Achsen der Slice.
    """
    dx, dy, dz = zooms_xyz
    pix = {"X": dx, "Y": dy, "Z": dz}
    w_mm = pix[spac_names[0]] * shape_2d[0]
    h_mm = pix[spac_names[1]] * shape_2d[1]
    return [0, w_mm, 0, h_mm]

def main():
    p = argparse.ArgumentParser(description="Overlay der TotalSegmentator-Masken auf NIfTI-Slice (physikalisch korrekt)")
    p.add_argument("--nii", required=True, help="Pfad zur NIfTI (HU)")
    p.add_argument("--outdir", required=True, help="Ordner mit TS-Masken (.nii.gz)")
    p.add_argument("--axis", type=int, default=2, help="0=sagittal, 1=koronal, 2=axial (default: 2)")
    p.add_argument("--slice", type=int, default=-1, help="-1 = mittlere Slice")
    p.add_argument("--alpha", type=float, default=0.3, help="Transparenz (bei Flächen)")
    p.add_argument("--filter", nargs='*', default=[], help="Nur Labels, die diese Teilstrings enthalten")
    p.add_argument("--window", nargs=2, type=int, default=[-1000, 400], help="HU-Fenster (min max)")
    p.add_argument("--contours", action="store_true", help="statt Flächen nur Konturen zeichnen")
    p.add_argument("--maxmasks", type=int, default=30, help="max. Anzahl Masken (um Overdraw zu vermeiden)")
    args = p.parse_args()

    # Basisbild
    img = nib.load(args.nii)
    # reorientiere in kanonisches RAS (sicherere Anzeige)
    img = nib.as_closest_canonical(img)
    data = img.get_fdata()  # (X,Y,Z)
    zooms = img.header.get_zooms()[:3]  # (dx,dy,dz)

    base_slice, used_idx, spac = load_slice(data, args.axis, None if args.slice < 0 else args.slice)
    vmin, vmax = args.window

    # Figure größenmäßig an phys. Ausdehnung anlehnen
    ext = get_extent(zooms, base_slice.shape, spac)
    plt.figure(figsize=(8, 8))
    plt.imshow(base_slice.T, cmap="gray", interpolation="bilinear", vmin=vmin, vmax=vmax,
           extent=ext, origin="lower", aspect="equal", resample=True)
    # ^ Achtung: .T + origin="lower" sorgt dafür, dass (X,Y) nicht gespiegelt erscheint

    # Masken einsammeln
    seg_files = sorted([f for f in os.listdir(args.outdir) if f.endswith(".nii.gz")])
    if args.filter:
        seg_files = [f for f in seg_files if any(tok in f for tok in args.filter)]
    if len(seg_files) > args.maxmasks:
        seg_files = seg_files[:args.maxmasks]  # harte Kappung gegen „Lamellen“

    if not seg_files:
        print("Keine Masken gefunden (Filter zu streng oder falscher Ordner?).")

    cmap_base = plt.get_cmap("tab20")

    # Overlays/Contours
    shown = 0
    for i, seg_file in enumerate(seg_files):
        seg_img = nib.load(os.path.join(args.outdir, seg_file))
        seg_img = nib.as_closest_canonical(seg_img)  # gleiche Orientierung wie Basis
        seg = seg_img.get_fdata()

        if seg.shape != data.shape:
            print(f"WARNUNG: Shape passt nicht: {seg_file} {seg.shape} vs {data.shape}")
            continue

        seg_slice, _idx, _spac = load_slice(seg, args.axis, used_idx)
        mask = seg_slice > 0

        if not np.any(mask):
            continue

        color = cmap_base(shown / max(1, min(len(seg_files), 20)-1))
        if args.contours:
            edge = binary_dilation(mask) ^ binary_erosion(mask)
            # Konturen per imshow+alpha: zeichne schmale Maske
            overlay = np.zeros((*edge.T.shape, 4), dtype=np.float32)  # RGBA
            overlay[..., :3] = color[:3]
            overlay[..., 3] = (edge.T.astype(np.float32)) * 1.0  # volle Deckung
            plt.imshow(overlay, extent=ext, origin="lower", aspect="equal", interpolation="nearest")
        else:
            # Fläche
            overlay = np.zeros((*mask.T.shape, 4), dtype=np.float32)
            overlay[..., :3] = color[:3]
            overlay[..., 3] = (mask.T.astype(np.float32)) * args.alpha
            plt.imshow(overlay, extent=ext, origin="lower", aspect="equal", interpolation="nearest")

        shown += 1

    plt.axis("off")
    plt.title(f"Slice {used_idx} (axis={args.axis})")
    out_png = f"visualized_segmentation_slice{used_idx}_axis{args.axis}.png"
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    print(f"Basis shape: {data.shape}, zooms(mm): {zooms}, axis={args.axis}, slice={used_idx}")
    print(f"Gezeichnete Masken: {shown}")
    print(f"Plot gespeichert: {out_png}")

if __name__ == "__main__":
    main()