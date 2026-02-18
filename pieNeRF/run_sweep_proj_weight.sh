#!/bin/bash
#SBATCH --job-name=projW_sweep
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=dgx
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:A100:1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/home/mnguest12/slurm/sweep_projW.%j.out
#SBATCH --error=/home/mnguest12/slurm/sweep_projW.%j.err
#SBATCH --chdir=/home/mnguest12/projects/thesis/pieNeRF

set -euo pipefail

# --- env ---
source /home/mnguest12/mambaforge/bin/activate
conda activate totalseg
PYTHON_BIN=${PYTHON_BIN:-python}

ROOT="/home/mnguest12/projects/thesis/pieNeRF"
TRAIN_PY="${ROOT}/train_emission.py"
BASE_CFG="${ROOT}/configs/spect.yaml"

export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"

BASE_DIR="${ROOT}/results_spect_wCT_sweep_projW"
mkdir -p "${BASE_DIR}"

# --- sweep values (only thing that changes) ---
PROJ_WEIGHTS=(0.0 0.001 0.0025 0.005 0.01)

# --- fixed baseline parameters (must match baseline) ---
SEED=0
MAX_STEPS=8000
LOG_EVERY=200
SAVE_EVERY=1000

PROJ_TARGET_SOURCE="counts"
POISSON_RATE_MODE="identity"
PROJ_LOSS_TYPE="poisson"
PROJ_WARMUP_STEPS=1500
PROJ_RAMP_STEPS=20000

ACT_LOSS_WEIGHT="3.0"
ACT_POS_FRACTION="0.05"
ACT_POS_WEIGHT="10.0"
ACT_SPARSITY_WEIGHT="2e-3"
ACT_TV_WEIGHT="1e-4"
ACT_SAMPLES="32768"
CT_LOSS_WEIGHT="1e-4"

Z_ENC_ALPHA="0.5"

# --- robust cleanup for results_spect symlink ---
RESTORE_DIR=""
cleanup_link() {
  rm -f "${ROOT}/results_spect" || true
  if [ -n "${RESTORE_DIR:-}" ] && [ -e "${RESTORE_DIR}" ]; then
    mv "${RESTORE_DIR}" "${ROOT}/results_spect"
  fi
}
trap cleanup_link EXIT

for W in "${PROJ_WEIGHTS[@]}"; do
  TAG="projW_${W}"
  RUN_DIR="${BASE_DIR}/${TAG}"
  RUN_OUT="${RUN_DIR}/results_spect"
  RUN_CFG="${RUN_DIR}/spect_${TAG}.yaml"
  TRAIN_LOG="${RUN_DIR}/train.log"

  echo "=============================="
  echo "Starting run with proj-loss-weight=${W}"
  echo "RUN_OUT=${RUN_OUT}"
  echo "=============================="

  mkdir -p "${RUN_DIR}" "${RUN_OUT}"

  if ls -1 "${RUN_OUT}/checkpoints"/checkpoint_step08000.pt >/dev/null 2>&1; then
    echo "[sweep] SKIP ${TAG}: checkpoint_step08000.pt exists"
    continue
  fi

  # Write a per-run config (only for naming; training controlled by CLI flags)
  ${PYTHON_BIN} - <<PY
import os, yaml
base_cfg = r"${BASE_CFG}"
run_cfg  = r"${RUN_CFG}"
tag      = r"${TAG}"

with open(base_cfg, "r") as f:
    cfg = yaml.safe_load(f)

cfg["expname"] = f"spect_emission_{tag}"

os.makedirs(os.path.dirname(run_cfg), exist_ok=True)
with open(run_cfg, "w") as f:
    yaml.safe_dump(cfg, f, sort_keys=False)

print("[sweep] wrote", run_cfg)
print("[sweep] expname =", cfg.get("expname"))
PY

  # --- robust per-run outdir via symlink ---
  RESTORE_DIR=""
  if [ -e "${ROOT}/results_spect" ] && [ ! -L "${ROOT}/results_spect" ]; then
    RESTORE_DIR="${ROOT}/results_spect.__backup__.$(date +%Y%m%d_%H%M%S)"
    echo "⚠️ Backing up existing ${ROOT}/results_spect -> ${RESTORE_DIR}"
    mv "${ROOT}/results_spect" "${RESTORE_DIR}"
  fi

  rm -f "${ROOT}/results_spect"
  ln -s "${RUN_OUT}" "${ROOT}/results_spect"
  echo "🔗 Linked ${ROOT}/results_spect -> ${RUN_OUT}"

  set -x
  ( cd "${ROOT}" && stdbuf -oL -eL srun --export=ALL,PYTHONPATH="${PYTHONPATH}" ${PYTHON_BIN} -u "${TRAIN_PY}" \
      --config "${RUN_CFG}" \
      --hybrid \
      --encoder-use-ct \
      --seed "${SEED}" \
      --max-steps "${MAX_STEPS}" \
      --log-every "${LOG_EVERY}" \
      --save-every "${SAVE_EVERY}" \
      --proj-target-source "${PROJ_TARGET_SOURCE}" \
      --poisson-rate-mode "${POISSON_RATE_MODE}" \
      --proj-loss-type "${PROJ_LOSS_TYPE}" \
      --proj-loss-weight "${W}" \
      --proj-warmup-steps "${PROJ_WARMUP_STEPS}" \
      --proj-ramp-steps "${PROJ_RAMP_STEPS}" \
      --act-loss-weight "${ACT_LOSS_WEIGHT}" \
      --act-pos-fraction "${ACT_POS_FRACTION}" \
      --act-pos-weight "${ACT_POS_WEIGHT}" \
      --act-sparsity-weight "${ACT_SPARSITY_WEIGHT}" \
      --act-tv-weight "${ACT_TV_WEIGHT}" \
      --act-samples "${ACT_SAMPLES}" \
      --ct-loss-weight "${CT_LOSS_WEIGHT}" \
      --z-enc-alpha "${Z_ENC_ALPHA}" \
      --final-act-compare \
    |& tee "${TRAIN_LOG}" )
  set +x

  cleanup_link

  if [ -d "${RUN_OUT}/checkpoints" ]; then
    echo "[guard] checkpoints dir exists ✅"
  else
    echo "[guard][FATAL] checkpoints dir missing at ${RUN_OUT}/checkpoints"
    exit 3
  fi

  FINAL_CKPT="${RUN_OUT}/checkpoints/checkpoint_step08000.pt"
  if [ -f "${FINAL_CKPT}" ]; then
    echo "[sweep] finished ${TAG} ✅ (${FINAL_CKPT})"
  else
    last_ckpt="$(ls -1 "${RUN_OUT}/checkpoints"/checkpoint_step*.pt 2>/dev/null | tail -n 1 || true)"
    if [ -n "${last_ckpt}" ]; then
      echo "[sweep][WARN] ${TAG}: checkpoint_step08000.pt missing; last ckpt: ${last_ckpt}"
    else
      echo "[sweep][WARN] ${TAG}: no checkpoints found"
    fi
  fi
done

echo "SWEEP FINISHED (training only)"