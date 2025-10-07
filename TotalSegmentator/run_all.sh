#!/usr/bin/env bash
#SBATCH --job-name=ct_pipeline
#SBATCH --output=/home/mnguest12/slurm/ct_pipeline.%j.out
#SBATCH --error=/home/mnguest12/slurm/ct_pipeline.%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --mem=64G

set -euo pipefail
set -x  # Debug-Ausgaben ins Slurm-Log

echo "========================================================"
echo "[INFO] Starting CT reconstruction + calibration pipeline"
echo "========================================================"
echo "GPUs available: ${SLURM_JOB_GPUS:-unknown}"
echo "Running on node: $(hostname)"
echo "Start time: $(date)"
echo "--------------------------------------------------------"

# === 1) Environment ===
source /home/mnguest12/mambaforge/bin/activate totalseg
cd /home/mnguest12/projects/thesis/TotalSegmentator

# --- Argumente ---
MAT_FILE=${1:-ct_proj_stack.mat}
PROJ_FILE=${2:-proj.txt}
PREFIX=${3:-ct_recon_rtk}
RUN_TAG=${4:-default}  # z. B. "180" oder "360"
OUTDIR="runs_${RUN_TAG}"

mkdir -p "${OUTDIR}/ts_preview" "${OUTDIR}/ts_output_calib"

ROI_AIR="phantom_air_mask.nii.gz"
ROI_WATER="phantom_water_mask.nii.gz"
USE_PHANTOM="no"
if [[ -f "$ROI_AIR" && -f "$ROI_WATER" ]]; then
  USE_PHANTOM="yes"
fi

echo "[INFO] Input MAT=${MAT_FILE}, PROJ=${PROJ_FILE}, PREFIX=${PREFIX}, RUN=${RUN_TAG}"

# === 2) FDK Reconstruction ===
echo "[STEP 1] Running FDK reconstruction ..."
python recon_fdk_rtk.py \
  --mat "$MAT_FILE" \
  --proj "$PROJ_FILE" \
  --out "${OUTDIR}/${PREFIX}.nii" \
  --out_ras "${OUTDIR}/${PREFIX}_RAS.nii" \
  --quick_ap_png "${OUTDIR}/ts_preview/coronal_AP_recon.png" \
  --preview_view coronal \
  --base_h_in 6.0

test -f "${OUTDIR}/${PREFIX}_RAS.nii" || { echo "[FATAL] missing: ${OUTDIR}/${PREFIX}_RAS.nii"; exit 2; }

# === 3) µ→HU Calibration ===
echo "[STEP 2] Calibrating µ→HU ..."
HU_OUT="${OUTDIR}/${PREFIX}_RAS_HU_calib.nii.gz"

if [[ "$USE_PHANTOM" == "yes" ]]; then
  echo "[INFO] Using phantom ROIs"
  python calibrate_mu_to_hu.py \
    -i "${OUTDIR}/${PREFIX}_RAS.nii" \
    -o "$HU_OUT" \
    --roi-air "$ROI_AIR" \
    --roi-water "$ROI_WATER" \
    --target-tissue-hu 40 \
    --clip -1050 3000 \
    --report-json "${OUTDIR}/ts_preview/hu_calibration_report.json" \
    --coronal-png "${OUTDIR}/ts_preview/coronal_AP_HU_calib.png"
else
  echo "[WARN] No ROIs found → Auto calibration"
  python calibrate_mu_to_hu.py \
    -i "${OUTDIR}/${PREFIX}_RAS.nii" \
    -o "$HU_OUT" \
    --target-tissue-hu 40 \
    --clip -1050 3000 \
    --report-json "${OUTDIR}/ts_preview/hu_calibration_report_auto.json" \
    --coronal-png "${OUTDIR}/ts_preview/coronal_AP_HU_calib_auto.png"
fi

test -f "$HU_OUT" || { echo "[FATAL] missing HU output: $HU_OUT"; exit 3; }

# === 4) TotalSegmentator + Overlay ===
echo "[STEP 3] Running TotalSegmentator ..."
python hu_seg_and_preview.py \
  --in-hu "$HU_OUT" \
  --seg-dir "${OUTDIR}/ts_output_calib" \
  --coronal-out "${OUTDIR}/ts_preview/coronal_AP_overlay_HUcalib.png" \
  --force

echo "--------------------------------------------------------"
echo "[INFO] Pipeline finished successfully at $(date)"
echo "Results in: ${OUTDIR}/"
echo "--------------------------------------------------------"