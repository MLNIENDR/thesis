import numpy as np
import nibabel as nib
import subprocess
import os
import argparse
import matplotlib
matplotlib.use("Agg")  # headless: PNG speichern statt Fenster öffnen
import matplotlib.pyplot as plt

def main():
    parser = argparse.ArgumentParser(
        description="XCAT attenuation (.bin, µ in mm^-1 @140 keV) → HU → NIfTI → TotalSegmentator"
    )
    parser.add_argument('-i', '--input', required=True, help='Pfad zur .bin Datei (XCAT attenuation map)')
    parser.add_argument('-o', '--output', required=True, help='Ordner für TotalSegmentator-Ausgabe')

    # WICHTIG: deine Voxel sind 0.15 cm = 1.5 mm → Default in MM
    parser.add_argument('--shape', nargs=3, type=int, default=[256, 256, 651], help='Volumen-Dimensionen x y z')
    parser.add_argument('--voxel', nargs=3, type=float, default=[1.5, 1.5, 1.5],
                        help='Voxelgröße in mm (dx dy dz)')

    # µ_Wasser: manuell oder automatisch
    parser.add_argument('--mu_water_mm', type=float, default=None,
                        help='µ_Wasser in mm^-1 (manuell setzen, falls bekannt)')
    parser.add_argument('--auto_mu', action='store_true',
                        help='µ_Wasser automatisch aus Histogramm bestimmen (größter Nicht-Luft-Peak)')

    # Sonstiges
    parser.add_argument('--clip', nargs=2, type=int, default=[-1000, 2000],
                        help='HU-Clipping [min max]')
    parser.add_argument('--int16_out', action='store_true',
                        help='NIfTI als int16 speichern (typisch für CT)')
    parser.add_argument('--plot_hist', action='store_true',
                        help='Histogramm der HU-Werte als PNG speichern')
    parser.add_argument('--nr', type=int, default=1, help='Threads für Resampling (TotalSegmentator)')
    parser.add_argument('--ns', type=int, default=1, help='Threads für Saving (TotalSegmentator)')
    parser.add_argument('--device', choices=['gpu','cpu','mps'], default='gpu',
                        help='TotalSegmentator Gerät (gpu, gpu:X, cpu, mps)')
    parser.add_argument('--skip_seg', action='store_true', help='Nur NIfTI erzeugen, Segmentierung überspringen')
    args = parser.parse_args()

    # ----------------- .bin laden -----------------
    print(f"Lade {args.input} ...")
    raw = np.fromfile(args.input, dtype=np.float32)
    expected = np.prod(args.shape)
    if raw.size != expected:
        raise ValueError(f"Elementanzahl passt nicht zu shape {args.shape}: {raw.size} vs {expected}")
    mu_mm = raw.reshape(args.shape, order='F')  # XCAT → Fortran-Order
    print(f"Volumen geladen: shape={mu_mm.shape}, dtype={mu_mm.dtype}")
    print(f"µ(mm^-1): min={mu_mm.min():.6f}, max={mu_mm.max():.6f}, mean={mu_mm.mean():.6f}, unique≈{len(np.unique(mu_mm))}")

    # ----------------- µ_Wasser bestimmen -----------------
    if args.auto_mu:
        vals = mu_mm[mu_mm > 0.001].flatten()
        hist, bin_edges = np.histogram(vals, bins=500)
        idx_max = np.argmax(hist)
        mu_water = float((bin_edges[idx_max] + bin_edges[idx_max+1]) / 2.0)
        print(f"Automatisch bestimmtes µ_Wasser ≈ {mu_water:.6f} mm^-1")
    else:
        if args.mu_water_mm is None:
            mu_water = 0.0153  # Fallback (monoenergetisch 140 keV, mm^-1)
            print(f"µ_Wasser nicht gesetzt → Fallback {mu_water:.4f} mm^-1")
        else:
            mu_water = float(args.mu_water_mm)
            print(f"Manuell gesetztes µ_Wasser = {mu_water:.6f} mm^-1")

    # ----------------- HU-Umrechnung -----------------
    volume_hu = 1000.0 * (mu_mm - mu_water) / mu_water
    volume_hu[mu_mm == 0.0] = -1000.0  # Luft explizit setzen
    hu_min, hu_max = args.clip
    volume_hu = np.clip(volume_hu, hu_min, hu_max)
    print(f"HU: min={volume_hu.min():.1f}, max={volume_hu.max():.1f}, mean={volume_hu.mean():.1f}")

    # ----------------- Histogramm -----------------
    if args.plot_hist:
        plt.figure(figsize=(8, 4))
        plt.hist(volume_hu.flatten(), bins=200)
        plt.xlabel("HU"); plt.ylabel("Voxelanzahl")
        plt.title(f"HU-Histogramm (µ_Wasser={mu_water:.5f} mm^-1)")
        out_hist = os.path.splitext(os.path.basename(args.input))[0] + "_HU_histogram.png"
        plt.savefig(out_hist, dpi=150, bbox_inches='tight')
        print(f"Histogramm gespeichert: {out_hist}")

    # ----------------- NIfTI speichern -----------------
    affine = np.diag(list(args.voxel) + [1.0])
    nii_path = os.path.splitext(os.path.basename(args.input))[0] + "_hu_scaled.nii.gz"
    data_to_save = np.round(volume_hu).astype(np.int16) if args.int16_out else volume_hu.astype(np.float32)
    img = nib.Nifti1Image(data_to_save, affine)
    img.set_qform(affine); img.set_sform(affine)
    nib.save(img, nii_path)
    print(f"NIfTI gespeichert: {nii_path}")
    print(f"Affine/Spacing (mm): diag = {args.voxel}")

    # ----------------- Segmentierung -----------------
    if args.skip_seg:
        print("Segmentierung übersprungen (--skip_seg).")
        return

    cmd = ["TotalSegmentator","-i",nii_path,"-o",args.output,
           "-nr",str(args.nr),"-ns",str(args.ns),"-d",args.device]
    print("Starte TotalSegmentator:", " ".join(cmd))
    subprocess.run(cmd, check=True)
    print(f"Segmentierung fertig in {args.output}")

if __name__ == "__main__":
    main()





#!/bin/bash
#SBATCH --job-name=totalseg
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --output=/home/mnguest12/slurm/totalseg.%j.out
#SBATCH --error=/home/mnguest12/slurm/totalseg.%j.err
# Optional/cluster-spezifisch:
# SBATCH --partition=gpu
# SBATCH --constraint=a100

#echo "SLURM_JOB_GPUS: ${SLURM_JOB_GPUS}"
#echo "SLURM_CPUS_PER_TASK: ${SLURM_CPUS_PER_TASK}"
#echo "CUDA_VISIBLE_DEVICES (gesetzt von SLURM): ${CUDA_VISIBLE_DEVICES}"

# Umgebung aktivieren
#source /home/mnguest12/mambaforge/bin/activate totalseg

# Threads sauber setzen
#export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
#export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
#export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}

# Start
#srun --gpu-bind=closest python /home/mnguest12/projects/tools/TotalSegmentator/totalseg.py \
#  -i /home/mnguest12/projects/tools/TotalSegmentator/phantom_01_gt.par_atn_1.bin \
#  -o /home/mnguest12/projects/tools/TotalSegmentator/output \
#  --auto_mu \
#  --voxel 1.5 1.5 1.5 \
#  --nr ${SLURM_CPUS_PER_TASK} \
#  --ns ${SLURM_CPUS_PER_TASK} \
#  --plot_hist \
#  --int16_out \
#  --device gpu




import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

# ================== Einstellungen ==================
nii_path = "phantom_01_gt.par_atn_1_hu_scaled.nii.gz"  # NIfTI-Datei laden
output_folder = "output"                               # TotalSegmentator output
slice_axis = 1                                        # 0=axial, 1=coronal, 2=sagittal
slice_idx = None                                      # None = mittlere Slice

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