#!/bin/bash
#SBATCH --job-name=ts_ctlike_abd2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --output=/home/mnguest12/slurm/ts_ctlike_abd2.%j.out
#SBATCH --error=/home/mnguest12/slurm/ts_ctlike_abd2.%j.err

source /home/mnguest12/mambaforge/bin/activate totalseg
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}

IN=/home/mnguest12/projects/tools/TotalSegmentator/phantom_01_gt.par_atn_1.bin
OUT=/home/mnguest12/projects/tools/TotalSegmentator/output_ctlike_abd2

srun --gpu-bind=closest python /home/mnguest12/projects/tools/TotalSegmentator/totalseg_ctprep.py \
  -i "${IN}" -o "${OUT}" \
  --mu_water_mm 0.0205 \
  --voxel 0.8 0.8 0.8 \
  --clip -150 250 \
  --psf_fwhm_mm 1.6 \
  --noise_gauss_hu 6.0 \
  --noise_poisson_hu 2.0 \
  --bias_amp_hu 8.0 \
  --bias_fwhm_mm 120.0 \
  --ts_body_crop \
  --roi_subset liver spleen kidney_left kidney_right aorta pancreas stomach gallbladder \
  --nr ${SLURM_CPUS_PER_TASK} --ns ${SLURM_CPUS_PER_TASK} \
  --device gpu --plot_hist