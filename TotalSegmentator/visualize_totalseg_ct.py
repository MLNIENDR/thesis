import os, argparse
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

p = argparse.ArgumentParser()
p.add_argument("--nii", default="phantom_01_gt.par_atn_1_ctlike.nii.gz")
p.add_argument("--outdir", default="output_ctlike_abd")
p.add_argument("--axis", type=int, default=1)
p.add_argument("--slice", type=int, default=-1, help="-1 = mittlere Slice")
args = p.parse_args()

nii_path = args.nii
output_folder = args.outdir
slice_axis = args.axis
slice_idx = None if args.slice < 0 else args.slice

# ================== NIfTI laden ==================
img = nib.load(nii_path)
data = img.get_fdata()
print(f"NIfTI geladen: shape={data.shape}, affine=\n{img.affine}")

if slice_idx is None:
    slice_idx = data.shape[slice_axis] // 2

if slice_axis == 0:
    base_slice = data[slice_idx, :, :]
elif slice_axis == 1:
    base_slice = data[:, slice_idx, :]
elif slice_axis == 2:
    base_slice = data[:, :, slice_idx]
else:
    raise ValueError("slice_axis muss 0, 1 oder 2 sein.")

base_slice = np.rot90(base_slice)

# ================== Segmentierungen laden ==================
seg_files = sorted([f for f in os.listdir(output_folder) if f.endswith(".nii.gz")])
n_colors = len(seg_files)
cmap_base = plt.get_cmap("tab20")

plt.figure(figsize=(8, 8))
plt.imshow(base_slice, cmap="gray", aspect=1)

for i, seg_file in enumerate(seg_files):
    seg_path = os.path.join(output_folder, seg_file)
    seg_img = nib.load(seg_path)
    seg_data = seg_img.get_fdata()

    if seg_data.shape != data.shape:
        print(f"WARNUNG: Shape passt nicht: {seg_file} ({seg_data.shape} vs {data.shape})")
        continue

    if slice_axis == 0:
        seg_slice = seg_data[slice_idx, :, :]
    elif slice_axis == 1:
        seg_slice = seg_data[:, slice_idx, :]
    elif slice_axis == 2:
        seg_slice = seg_data[:, :, slice_idx]

    seg_slice = np.rot90(seg_slice)
    mask = seg_slice > 0

    color = cmap_base(i / max(1, n_colors-1))
    plt.imshow(np.ma.masked_where(~mask, seg_slice), alpha=0.5,
               cmap=plt.cm.colors.ListedColormap([color]), aspect=1)

plt.axis('off')
plt.title(f"Slice {slice_idx} (Achse {slice_axis}) mit Segmentierungen")

out_png = f"visualized_segmentation_slice{slice_idx}_axis{slice_axis}.png"
plt.savefig(out_png, dpi=300, bbox_inches='tight')
print(f"Plot gespeichert: {out_png}")