#!/bin/bash
#SBATCH --job-name=ct_pipeline
#SBATCH --output=/home/mnguest12/slurm/ct_pipeline.%j.out
#SBATCH --error=/home/mnguest12/slurm/ct_pipeline.%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --mem=64G

echo "========================================================"
echo "[INFO] Starting CT reconstruction + calibration pipeline"
echo "========================================================"
echo "GPUs available: ${SLURM_JOB_GPUS}"
echo "Running on node: $(hostname)"
echo "Start time: $(date)"
echo "--------------------------------------------------------"

# === 1. Environment setup ===
source /home/mnguest12/mambaforge/bin/activate totalseg
cd /home/mnguest12/projects/thesis/TotalSegmentator

# === 2. Step 1: FDK Reconstruction ===
echo "[STEP 1] Running FDK reconstruction ..."
python recon_fdk_rtk.py \
  --mat ct_proj_stack.mat \
  --proj proj.txt \
  --out ct_recon_rtk.nii \
  --out_ras ct_recon_rtk_RAS.nii \
  --quick_ap_png ts_preview/coronal_AP_recon.png \
  --preview_view coronal \
  --base_h_in 6.0

# === 3. Step 2: µ → HU calibration ===
echo "[STEP 2] Calibrating µ to HU using phantom ROIs ..."
python calibrate_mu_to_hu.py \
  -i ct_recon_rtk_RAS.nii \
  -o ct_recon_rtk_RAS_HU_calib.nii.gz \
  --roi-air phantom_air_mask.nii.gz \
  --roi-water phantom_water_mask.nii.gz \
  --target-tissue-hu 0 \
  --clip -1050 3000 \
  --report-json ts_preview/hu_calibration_report.json \
  --coronal-png ts_preview/coronal_AP_HU_calib.png

# === 4. Step 3: Segmentation and overlay ===
echo "[STEP 3] Running TotalSegmentator segmentation ..."
python hu_seg_and_preview.py \
  --in-hu ct_recon_rtk_RAS_HU_calib.nii.gz \
  --seg-dir ts_output_calib \
  --coronal-out ts_preview/coronal_AP_overlay_HUcalib.png \
  --force

echo "--------------------------------------------------------"
echo "[INFO] Pipeline finished successfully at $(date)"
echo "Output files:"
echo "  - HU volume: ct_recon_rtk_RAS_HU_calib.nii.gz"
echo "  - Segmentation: ts_output_calib/segmentation.nii.gz (or ts_output_calib.nii)"
echo "  - Overlay: ts_preview/coronal_AP_overlay_HUcalib.png"
echo "--------------------------------------------------------"