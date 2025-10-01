#!/bin/bash
#SBATCH --job-name=totalseg_qual
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --output=/home/mnguest12/slurm/totalseg_qual.%j.out
#SBATCH --error=/home/mnguest12/slurm/totalseg_qual.%j.err
# SBATCH --partition=gpu
# SBATCH --constraint=a100

echo "SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

source /home/mnguest12/mambaforge/bin/activate totalseg
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}

# Eingaben/Outputs
IN=/home/mnguest12/projects/tools/TotalSegmentator/phantom_01_gt.par_atn_1.bin
OUT=/home/mnguest12/projects/tools/TotalSegmentator/output_qual

# Start (WICHTIG: keine --int16_out, statt auto_mu manuell; engeres HU-Window)
srun --gpu-bind=closest python /home/mnguest12/projects/tools/TotalSegmentator/totalseg.py \
  -i "${IN}" \
  -o "${OUT}" \
  --mu_water_mm 0.0190 \
  --voxel 1.5 1.5 1.5 \
  --clip -1000 1000 \
  --nr ${SLURM_CPUS_PER_TASK} \
  --ns ${SLURM_CPUS_PER_TASK} \
  --plot_hist \
  --device gpu

# Body-Crop in TS (entfernt Luft/Schwarzrand → stabilere Normalisierung)
# Wenn du direkt in TS croppen möchtest, kannst du zusätzlich mit Subprozess laufen lassen:
# TotalSegmentator -i phantom_01_gt.par_atn_1_hu_scaled.nii.gz -o "${OUT}" -bs -d gpu