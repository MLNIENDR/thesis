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

IN=/home/mnguest12/projects/tools/TotalSegmentator/phantom_01_gt.par_atn_1.bin
OUT=/home/mnguest12/projects/tools/TotalSegmentator/output
SCRIPT=/home/mnguest12/projects/tools/TotalSegmentator/totalseg.py

mkdir -p "$OUT"

srun --gpu-bind=closest python "$SCRIPT" \
  -i "$IN" \
  -o "$OUT" \
  --mu_water_mm 0.030 \
  --clip -1024 1500 \
  --crop \
  --bias_amp_hu 12 --bias_fwhm_mm 120 \
  --psf_fwhm_mm 1.2 \
  --noise_hu 4.0 \
  --plot_hist \
  --nr ${SLURM_CPUS_PER_TASK} \
  --ns ${SLURM_CPUS_PER_TASK}
  # Hinweis: --device gpu ist nicht nötig; SLURM setzt CUDA_VISIBLE_DEVICES.
  # Optional erzwingen:
  # --device cpu