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
set -x

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
PREFIX=${3:-ct_recon_rtk360}
RUN_TAG=${4:-360}  # z. B. "180" oder "360"
OUTDIR="runs_${RUN_TAG}"

mkdir -p "${OUTDIR}/ts_preview"

# Phantom-ROIs optional
ROI_AIR="phantom_air_mask.nii.gz"
ROI_WATER="phantom_water_mask.nii.gz"
USE_PHANTOM="no"
if [[ -f "$ROI_AIR" && -f "$ROI_WATER" ]]; then
  USE_PHANTOM="yes"
fi

echo "[INFO] Input MAT=${MAT_FILE}, PROJ=${PROJ_FILE}, PREFIX=${PREFIX}, RUN=${RUN_TAG}"

# === 2) FDK Reconstruction (RAS) ===
echo "[STEP 1] Running FDK reconstruction ..."
python recon_fdk_rtk.py \
  --mat "$MAT_FILE" \
  --proj "$PROJ_FILE" \
  --out "${OUTDIR}/${PREFIX}_RAS.nii" \
  --quick_ap_png "${OUTDIR}/ts_preview/" \
  --preview_view coronal \
  --base_h_in 6.0

RAS_NII="${OUTDIR}/${PREFIX}_RAS.nii"
test -f "${RAS_NII}" || { echo "[FATAL] missing: ${RAS_NII}"; exit 2; }

# === 3) µ→HU Calibration ===
echo "[STEP 2] Calibrating µ→HU ..."
HU_OUT="${OUTDIR}/${PREFIX}_RAS_HU.nii.gz"

if [[ "$USE_PHANTOM" == "yes" ]]; then
  echo "[INFO] Using phantom ROIs"
  python calibrate_mu_to_hu.py \
    -i "${RAS_NII}" \
    -o "$HU_OUT" \
    --roi-air "$ROI_AIR" \
    --roi-water "$ROI_WATER" \
    --target-tissue-hu 40 \
    --clip -1050 3000 \
    --report-json "${OUTDIR}/ts_preview/hu_calibration_report.json" \
    --coronal-png "${OUTDIR}/ts_preview/${PREFIX}_HU_coronal_phantom.png"
else
  echo "[WARN] No ROIs found → Auto calibration"
  python calibrate_mu_to_hu.py \
    -i "${RAS_NII}" \
    -o "$HU_OUT" \
    --target-tissue-hu 40 \
    --clip -1050 3000 \
    --report-json "${OUTDIR}/ts_preview/hu_calibration_report_auto.json" \
    --coronal-png "${OUTDIR}/ts_preview/${PREFIX}_HU_coronal_auto.png"
fi

test -f "$HU_OUT" || { echo "[FATAL] missing HU output: $HU_OUT"; exit 3; }

# === 4) TotalSegmentator (Multi-Label ins OUTDIR) ===
echo "[STEP 3] Running TotalSegmentator ..."
TotalSegmentator -i "$HU_OUT" -o "${OUTDIR}" --ml
echo "[OK] Segmentation finished → ${OUTDIR}"

# Multi-Label-Datei konsolidieren (einheitlicher Name)
if   [[ -f "${OUTDIR}/segmentations.nii.gz" ]]; then
  SEG_IN="${OUTDIR}/segmentations.nii.gz"
elif [[ -f "${OUTDIR}/segmentation.nii.gz" ]]; then
  SEG_IN="${OUTDIR}/segmentation.nii.gz"
elif [[ -f "${OUTDIR}/ts_output_calib.nii.gz" ]]; then
  SEG_IN="${OUTDIR}/ts_output_calib.nii.gz"
elif [[ -f "${OUTDIR}/ts_output_calib.nii" ]]; then
  SEG_IN="${OUTDIR}/ts_output_calib.nii"
else
  echo "[FATAL] Konnte Multi-Label nicht finden in ${OUTDIR}."
  ls -lh "${OUTDIR}" || true
  exit 4
fi

SEG_STD="${OUTDIR}/ts_output_calib.nii.gz"
if [[ "${SEG_IN}" == *.nii.gz ]]; then
  cp -f "${SEG_IN}" "${SEG_STD}"
else
  gzip -c "${SEG_IN}" > "${SEG_STD}"
fi
echo "[OK] Multi-Label bereit: ${SEG_STD}"

# === 5) Overlay rendern ===
echo "[STEP 4] Making coronal AP overlay ..."
python hu_seg_and_preview.py \
  --in-hu "$HU_OUT" \
  --seg-path "${SEG_STD}" \
  --coronal-out "${OUTDIR}/ts_preview/${PREFIX}_overlay_HUcalib.png" \
  --force

echo "--------------------------------------------------------"
echo "[INFO] Pipeline finished successfully at $(date)"
echo "Results in: ${OUTDIR}/"
echo "  - HU volume:        ${HU_OUT}"
echo "  - Seg multilabel:   ${SEG_STD}"
echo "  - Preview PNGs:     ${OUTDIR}/ts_preview/"
echo "--------------------------------------------------------"