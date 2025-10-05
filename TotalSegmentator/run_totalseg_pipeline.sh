#!/bin/bash
#SBATCH --job-name=ts_pipeline
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --output=/home/mnguest12/slurm/ts_pipeline.%j.out
#SBATCH --error=/home/mnguest12/slurm/ts_pipeline.%j.err
# SBATCH --partition=gpu
# SBATCH --constraint=a100

set -euo pipefail

# --- Env ---
source /home/mnguest12/mambaforge/bin/activate totalseg
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}

PY="/home/mnguest12/mambaforge/envs/totalseg/bin/python"
TS="/home/mnguest12/mambaforge/envs/totalseg/bin/TotalSegmentator"
echo "[DBG] python: $($PY -V)"
echo "[DBG] TS: $TS"
echo "[DBG] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"

# --- IO (absolute Pfade) ---
cd /home/mnguest12/projects/thesis/TotalSegmentator
PWD_ABS="$(pwd)"

IN_MU="${PWD_ABS}/ct_recon_rtk.nii"        # rekonstruierter µ-CT
OUT_HU="${PWD_ABS}/ct_hu.nii"              # wird neu erstellt
TS_OUT="${PWD_ABS}/${TS_OUT:-totalseg_out}"
PREVIEW_DIR="${PWD_ABS}/${PREVIEW_DIR:-ts_preview}"

# frisch aufsetzen
rm -rf "$TS_OUT"
mkdir -p "$TS_OUT" "$PREVIEW_DIR"

# Vor dem Run: verirrte ML-Dateien im Top-Level weg (alte Reste verhindern)
for STRAY in segmentations.nii.gz segmentations.nii \
             labels.nii.gz labels.nii \
             multilabel.nii.gz multilabel.nii \
             segmentation.nii.gz segmentation.nii \
             totalseg_out.nii totalseg_out.nii.gz; do
  if [[ -f "${PWD_ABS}/${STRAY}" ]]; then
    echo "[CLEAN] Entferne altes Top-Level: ${STRAY}"
    rm -f "${PWD_ABS}/${STRAY}"
  fi
done

# --- STEP 1: μ -> HU ---
if [[ -f "${PWD_ABS}/mu_to_hu_and_precheck.py" ]]; then
  echo "[STEP 1/3] μ -> HU (auto μ_water)…"
  $PY "${PWD_ABS}/mu_to_hu_and_precheck.py" \
    -i "$IN_MU" \
    -o "$OUT_HU" \
    --auto_mu \
    --clip -1024 3071 \
    --int16_out \
    --hist_png "${PREVIEW_DIR}/mu_hist.png"
else
  echo "[WARN] mu_to_hu_and_precheck.py fehlt – nutze µ als HU."
  OUT_HU="$IN_MU"
fi

# --- STEP 2: TotalSegmentator ---
echo "[STEP 2/3] TotalSegmentator (--ml, --fast)…"
srun --gpu-bind=closest "$TS" \
  -i "$OUT_HU" \
  -o "$TS_OUT" \
  --ml \
  --fast \
  --device gpu

echo "[DBG] Inhalt von $TS_OUT nach TS:"
ls -lh "$TS_OUT" | sed -n '1,60p' || true

# --- Multi-Label robust finden/normalisieren (Endung beibehalten) ---
echo "[CHK] Suche Multi-Label-Datei…"
pick_ml () {
  find "$1" -maxdepth "$2" -type f \( \
    -iname 'segmentations.nii'     -o -iname 'segmentations.nii.gz' \
 -o -iname 'labels.nii'            -o -iname 'labels.nii.gz' \
 -o -iname 'multilabel.nii'        -o -iname 'multilabel.nii.gz' \
 -o -iname 'segmentation.nii'      -o -iname 'segmentation.nii.gz' \
 -o -iname 'totalseg_out.nii'      -o -iname 'totalseg_out.nii.gz' \
  \) | head -n1
}

# 1) zuerst im Zielordner
ML="$(pick_ml "$TS_OUT" 2 || true)"
# 2) falls dort nichts: auch Top-Level inspizieren (manche Builds legen dort ab)
if [[ -z "${ML:-}" ]]; then
  ML="$(pick_ml "$PWD_ABS" 1 || true)"
  if [[ -n "${ML:-}" ]]; then
    case "$ML" in
      *.nii.gz) ML_STD="$TS_OUT/segmentations.nii.gz" ;;
      *.nii)    ML_STD="$TS_OUT/segmentations.nii" ;;
      *)        ML_STD="$TS_OUT/segmentations.nii" ;;
    esac
    echo "[FIX] Top-Level ML gefunden: $ML → kopiere nach $ML_STD"
    cp -f "$ML" "$ML_STD"
    ML="$ML_STD"
  fi
fi

if [[ -n "${ML:-}" ]]; then
  # falls der Name nicht standardisiert ist, aber im TS_OUT liegt: normalisieren mit gleicher Endung
  case "$ML" in
    "$TS_OUT/segmentations.nii"|"$TS_OUT/segmentations.nii.gz") : ;;
    *.nii.gz) cp -f "$ML" "$TS_OUT/segmentations.nii.gz"; ML="$TS_OUT/segmentations.nii.gz" ;;
    *.nii)    cp -f "$ML" "$TS_OUT/segmentations.nii";    ML="$TS_OUT/segmentations.nii" ;;
    *)        cp -f "$ML" "$TS_OUT/segmentations.nii";    ML="$TS_OUT/segmentations.nii" ;;
  esac
  echo "[OK] Multi-Label: $ML"

  # kurze Lesbarkeitsprobe mit nibabel
  "$PY" - <<PY "$ML"
import sys, nibabel as nib
p=sys.argv[1]
img=nib.load(p)
print("[OK] nibabel kann lesen:", p, "shape=", img.shape, "vox=", img.header.get_zooms()[:3])
PY
else
  echo "[WARN] Kein Multi-Label gefunden – baue eins aus Einzelmasken…"
  $PY - <<'PY'
import os, glob, numpy as np, nibabel as nib
seg_dir="totalseg_out"
ct_path="ct_hu.nii" if os.path.isfile("ct_hu.nii") else "ct_recon_rtk.nii"
out_nii=os.path.join(seg_dir,"segmentations.nii")
ct=nib.as_closest_canonical(nib.load(ct_path))
shape,aff=ct.shape,ct.affine
masks=sorted(p for p in glob.glob(os.path.join(seg_dir,"*.nii*"))
             if os.path.basename(p).lower() not in (
               "segmentations.nii","segmentations.nii.gz",
               "labels.nii","labels.nii.gz",
               "multilabel.nii","multilabel.nii.gz",
               "segmentation.nii","segmentation.nii.gz",
               "totalseg_out.nii","totalseg_out.nii.gz"))
if not masks:
    raise SystemExit("[ERR] Keine Einzelmasken in "+seg_dir)
lab=np.zeros(shape,np.int32); cur=1
for i,p in enumerate(masks,1):
    m=nib.as_closest_canonical(nib.load(p)).get_fdata().astype(np.uint8)
    if m.shape!=shape:
        raise SystemExit(f"[ERR] Shape mismatch {p}: {m.shape} vs {shape}")
    lab[(lab==0)&(m>0)]=cur; cur+=1
nib.save(nib.Nifti1Image(lab,aff),out_nii)
print("[OK] geschrieben:", out_nii)
PY
  ML="$TS_OUT/segmentations.nii"
fi

# --- STEP 3: Coronal AP Overlay (ein PNG) ---
echo "[STEP 3/3] Coronal AP Overlay…"
$PY "${PWD_ABS}/make_coronal_overlay.py" \
  --ct "$OUT_HU" \
  --seg_dir "$TS_OUT" \
  --out "${PREVIEW_DIR}/coronal_AP_overlay.png"

echo "[OK] Pipeline fertig."
echo " - HU-CT:             $OUT_HU"
echo " - TS-Ausgabe:        $TS_OUT"
echo " - Multi-Label:       $(ls -1 "$TS_OUT"/segmentations.nii* 2>/dev/null | head -n1 || echo 'n/a')"
echo " - Overlay:           ${PREVIEW_DIR}/coronal_AP_overlay.png"
echo " - Histogramm:        ${PREVIEW_DIR}/mu_hist.png (falls erstellt)"