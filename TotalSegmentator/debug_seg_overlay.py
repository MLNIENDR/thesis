#!/usr/bin/env python3
import os, argparse, numpy as np, nibabel as nib
from nibabel.processing import resample_from_to
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def print_img_info(tag, img):
    A = img.affine
    sp = (A[0,0], A[1,1], A[2,2])
    print(f"[{tag}] shape={img.shape}  spacing={tuple(float(s) for s in sp)}  origin={(A[0,3],A[1,3],A[2,3])}")

def overlay_coronal(hu_img, seg_img, out_png, orient="AP", label=None, alpha=0.4):
    vol = np.asarray(hu_img.dataobj).astype(np.float32)  # (X,Y,Z) RAS
    lab = np.asarray(seg_img.dataobj).astype(np.int32)
    X,Y,Z = vol.shape
    dx,dy,dz = float(hu_img.affine[0,0]), float(hu_img.affine[1,1]), float(hu_img.affine[2,2])

    sl_hu = vol[X//2,:,:]        # (Y,Z)
    sl_lab = lab[X//2,:,:]

    img2 = np.flipud(sl_hu)
    seg2 = np.flipud(sl_lab)
    if orient.upper() == "AP":
        img2 = np.fliplr(img2)
        seg2 = np.fliplr(seg2)

    # Fensterung 2..98 %
    p2,p98 = np.percentile(img2[np.isfinite(img2)], [2,98])
    shown = np.clip((img2-p2)/(p98-p2+1e-6), 0, 1)

    extent = [0, dz*Z, 0, dy*Y]
    width_mm, height_mm = extent[1], extent[3]
    aspect = height_mm/max(width_mm,1e-6)
    h_in=8.0; w_in=h_in/max(aspect,1e-6)

    fig,ax=plt.subplots(figsize=(w_in,h_in), dpi=180)
    ax.imshow(shown, cmap="gray", origin="lower", extent=extent, aspect="equal")

    if label is None:
        mask = (seg2>0).astype(float)
    else:
        mask = (seg2==int(label)).astype(float)
    cmap = matplotlib.colormaps.get("turbo")
    color = cmap(0.15)
    rgba = np.zeros(mask.shape+(4,), np.float32)
    rgba[...,0]=color[0]; rgba[...,1]=color[1]; rgba[...,2]=color[2]; rgba[...,3]=mask*alpha
    ax.imshow(rgba, origin="lower", extent=extent, aspect="equal")

    ax.set_title(f"Coronal {orient}  label={label if label is not None else 'any>0'}")
    ax.set_xlabel("Z [mm]"); ax.set_ylabel("Y [mm]")
    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
    fig.savefig(out_png, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print(f"[OK] saved {out_png}")

def auto_pick_liver_label(hu_img, seg_img):
    hu = np.asarray(hu_img.dataobj).astype(np.float32)
    lab = np.asarray(seg_img.dataobj).astype(np.int32)
    X,Y,Z = hu.shape
    dx,dy,dz = float(hu_img.affine[0,0]), float(hu_img.affine[1,1]), float(hu_img.affine[2,2])

    # Abdomen-Box grob (mm): Y 350..650, Z 120..360
    y0,y1 = int(350/dy), int(650/dy)
    z0,z1 = int(120/dz), int(360/dz)
    x = X//2

    # Kandidaten-Labels in der Box
    sub = lab[x, y0:y1, z0:z1]
    ids, cnt = np.unique(sub[sub>0], return_counts=True)
    if len(ids)==0: return None

    # HU-Mittel pro Label in der Box
    best_id, best_score = None, -1
    for L in ids:
        m = (sub==L)
        hu_vals = hu[x, y0:y1, z0:z1][m]
        if hu_vals.size < 50: continue
        mean = float(np.mean(hu_vals))
        # Scoring: Nähe zu typischer Leber (30..80 HU)
        score = -abs(mean-55.0) + 0.001*hu_vals.size
        if score > best_score:
            best_score, best_id = score, int(L)
    return best_id

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--hu", required=True)
    ap.add_argument("--seg", required=True)  # Datei oder Ordner
    args=ap.parse_args()

    hu_img = nib.load(args.hu)

    seg_path = args.seg
    if os.path.isdir(seg_path):
        from glob import glob
        cands = [os.path.join(seg_path, "segmentation.nii.gz"),
                 os.path.join(seg_path, "segmentation.nii"),
                 os.path.join(seg_path, "ts_output.nii")]
        found = [p for p in cands if os.path.exists(p)]
        if not found:
            nn = glob(os.path.join(seg_path, "*.nii*"))
            if not nn:
                raise FileNotFoundError("Keine Seg-Datei im Ordner gefunden.")
            seg_path = sorted(nn, key=os.path.getsize, reverse=True)[0]
        else:
            seg_path = found[0]
    seg_img0 = nib.load(seg_path)

    print_img_info("HU", hu_img)
    print_img_info("SEG(raw)", seg_img0)

    # resample auf HU grid
    if seg_img0.shape != hu_img.shape or not np.allclose(seg_img0.affine, hu_img.affine, atol=1e-4):
        print("[WARN] resample seg->HU grid (nearest)")
        seg_img = resample_from_to(seg_img0, hu_img, order=0)
    else:
        seg_img = seg_img0

    print_img_info("SEG(onHU)", seg_img)

    # Liver automatisch schätzen
    lid = auto_pick_liver_label(hu_img, seg_img)
    print(f"[INFO] auto liver id: {lid}")

    # Zwei Overlays (AP/PA) schreiben
    base = os.path.splitext(seg_path)[0]
    overlay_coronal(hu_img, seg_img, base+"_overlay_AP.png", orient="AP", label=lid)
    overlay_coronal(hu_img, seg_img, base+"_overlay_PA.png", orient="PA", label=lid)

if __name__ == "__main__":
    main()