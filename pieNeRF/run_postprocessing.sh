#!/bin/bash
#SBATCH --job-name=postproc_baseline_calib
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=dgx
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:A100:1
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/home/mnguest12/slurm/postprocessing.%j.out
#SBATCH --error=/home/mnguest12/slurm/postprocessing.%j.err
#SBATCH --chdir=/home/mnguest12/projects/thesis/pieNeRF

set -euo pipefail

SWEEP_ROOT="${SWEEP_ROOT:-/home/mnguest12/projects/thesis/pieNeRF/results_spect_sweep_projW}"
SWEEP_TAGS="${SWEEP_TAGS:-}"
MANIFEST="/home/mnguest12/projects/thesis/pieNeRF/data/manifest_abs.csv"

MASK_PATTERN='/home/mnguest12/projects/thesis/Data_Processing/{phantom}/out/mask.npy'
DEVICE="cuda"

CONDA_ENV="totalseg"
CONDA_ACTIVATE="/home/mnguest12/mambaforge/bin/activate"
PYTHON_BIN="python"

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK}

echo "🛠 Starting postprocessing job on ${HOSTNAME}"
echo "📅 Job started at: $(date)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "sweep_root=${SWEEP_ROOT}"
echo "sweep_tags=${SWEEP_TAGS:-<all>}"
echo "manifest=${MANIFEST}"
echo "mask_pattern=${MASK_PATTERN}"
echo "device=${DEVICE}"

source "${CONDA_ACTIVATE}"
conda activate "${CONDA_ENV}"

echo "python=$(which "${PYTHON_BIN}")"
"${PYTHON_BIN}" -V

echo "PWD=$(pwd)"
nvidia-smi || true

# --- sanity checks ---
test -d "${SWEEP_ROOT}" || { echo "[ERROR] SWEEP_ROOT not found: ${SWEEP_ROOT}"; exit 2; }
test -f "${MANIFEST}" || { echo "[ERROR] MANIFEST not found: ${MANIFEST}"; exit 2; }

# ensure imports work (Data_Processing + repo)
export PYTHONPATH="/home/mnguest12/projects/thesis:/home/mnguest12/projects/thesis/pieNeRF:${PYTHONPATH:-}"

declare -a TAG_DIRS=()
if [[ -n "${SWEEP_TAGS}" ]]; then
  for tag in ${SWEEP_TAGS}; do
    TAG_DIRS+=("${SWEEP_ROOT}/${tag}")
  done
else
  while IFS= read -r d; do
    TAG_DIRS+=("${d}")
  done < <(find "${SWEEP_ROOT}" -mindepth 1 -maxdepth 1 -type d | sort)
fi

if [[ ${#TAG_DIRS[@]} -eq 0 ]]; then
  echo "[ERROR] No sweep tag directories found in ${SWEEP_ROOT}"
  exit 2
fi

FAIL_COUNT=0

for tag_dir in "${TAG_DIRS[@]}"; do
  SWEEP_TAG="$(basename "${tag_dir}")"
  RUN_DIR="${tag_dir}/results_spect"
  SPLIT_JSON="${RUN_DIR}/split.json"
  CONFIG="${tag_dir}/spect_${SWEEP_TAG}.yaml"
  OUT_DIR="${RUN_DIR}/postproc_baseline_calib"

  if [[ ! -f "${CONFIG}" ]]; then
    CONFIG_CANDIDATE="$(find "${tag_dir}" -maxdepth 1 -type f -name '*.yaml' | head -n 1 || true)"
    if [[ -n "${CONFIG_CANDIDATE}" ]]; then
      CONFIG="${CONFIG_CANDIDATE}"
    fi
  fi

  echo ""
  echo "============================================================"
  echo "🔁 Running postprocessing for ${SWEEP_TAG}"
  echo "run_dir=${RUN_DIR}"
  echo "split_json=${SPLIT_JSON}"
  echo "config=${CONFIG}"
  echo "out_dir=${OUT_DIR}"
  echo "============================================================"

  if [[ ! -d "${RUN_DIR}" ]]; then
    echo "[ERROR] RUN_DIR not found: ${RUN_DIR}"
    FAIL_COUNT=$((FAIL_COUNT + 1))
    continue
  fi
  if [[ ! -f "${SPLIT_JSON}" ]]; then
    echo "[ERROR] SPLIT_JSON not found: ${SPLIT_JSON}"
    FAIL_COUNT=$((FAIL_COUNT + 1))
    continue
  fi
  if [[ ! -f "${CONFIG}" ]]; then
    echo "[ERROR] CONFIG not found: ${CONFIG}"
    FAIL_COUNT=$((FAIL_COUNT + 1))
    continue
  fi

  mkdir -p "${OUT_DIR}"
  echo "[DBG] Listing ${RUN_DIR}/test_slices:"
  ls -lah "${RUN_DIR}/test_slices" || true

  set -x
  if ! srun /usr/bin/time -v "${PYTHON_BIN}" -u postprocessing.py \
    --run-dir "${RUN_DIR}" \
    --split-json "${SPLIT_JSON}" \
    --manifest "${MANIFEST}" \
    --config "${CONFIG}" \
    --out-dir "${OUT_DIR}" \
    --mask-path-pattern "${MASK_PATTERN}" \
    --device "${DEVICE}" \
    --pred-act-per-phantom \
    --pred-act-pattern "test_slices/{phantom}/activity_pred.npy" \
    --calibrate-scale \
    --render-projections \
    --save-proj-npy \
    --save-proj-png \
    --timing \
    --skip-plots \
    --save-active-organ-plots; then
    set +x
    echo "[ERROR] postprocessing failed for ${SWEEP_TAG}"
    FAIL_COUNT=$((FAIL_COUNT + 1))
    continue
  fi
  set +x
done

if [[ ${FAIL_COUNT} -ne 0 ]]; then
  echo "[ERROR] Finished with ${FAIL_COUNT} failed sweep tag(s)."
  exit 1
fi

echo "✅ Postprocessing finished at: $(date)"
