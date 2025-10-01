#!/bin/bash
#SBATCH --job-name=totalseg
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --output=/home/mnguest12/slurm/totalseg.%j.out
#SBATCH --error=/home/mnguest12/slurm/totalseg.%j.err
# SBATCH --partition=gpu
# SBATCH --constraint=a100

set -euo pipefail

echo "SLURM_JOB_GPUS: ${SLURM_JOB_GPUS:-?}"
echo "SLURM_CPUS_PER_TASK: ${SLURM_CPUS_PER_TASK:-?}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-?}"

# Env
source /home/mnguest12/mambaforge/bin/activate totalseg
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}

IN=/home/mnguest12/projects/thesis/TotalSegmentator/phantom_01_gt.par_atn_1.bin
OUT=/home/mnguest12/projects/thesis/TotalSegmentator/output
SCRIPT=/home/mnguest12/projects/thesis/TotalSegmentator/totalseg.py

mkdir -p "$OUT"

# Hinweis zu mu_water:
# - Dein Script unterstützt entweder --mu_water_mm <float> ODER --auto_mu
#   Für reproduzierbare Runs hier manuell gesetzt (0.030). Auto-Option unten auskommentiert.
# - Wenn du Auto willst, ersetze die Zeile mit --mu_water_mm durch: --auto_mu

srun --gpu-bind=closest \
  python "$SCRIPT" \
    -i "$IN" \
    -o "$OUT" \
    --shape 256 256 651 \
    --voxel 1.5 1.5 1.5 \
    --mu_water_mm 0.030 \
    # --auto_mu \
    --clip -1024 1500 \
    --int16_out \
    --plot_hist \
    --ts_body_crop \
    --nr ${SLURM_CPUS_PER_TASK} \
    --ns ${SLURM_CPUS_PER_TASK} \
    --device gpu
    # Optional: nur bestimmte Regionen (Beispiel Lunge-Lappen)
    # --roi_subset lung_upper_lobe_left lung_upper_lobe_right lung_middle_lobe_right lung_lower_lobe_left lung_lower_lobe_right

# Ende