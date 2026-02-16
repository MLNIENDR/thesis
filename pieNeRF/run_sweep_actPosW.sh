#!/bin/bash
#SBATCH --job-name=posW_sweep
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=dgx
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:A100:1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/home/mnguest12/slurm/sweep_posW.%j.out
#SBATCH --error=/home/mnguest12/slurm/sweep_posW.%j.err
#SBATCH --chdir=/home/mnguest12/projects/thesis/pieNeRF

set -euo pipefail

source /home/mnguest12/mambaforge/bin/activate
conda activate totalseg
PYTHON_BIN=python

ROOT="/home/mnguest12/projects/thesis/pieNeRF"
TRAIN_PY="${ROOT}/train_emission.py"
BASE_CFG="${ROOT}/configs/spect.yaml"

export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"

BASE_DIR="${ROOT}/results_spect_sweep_posW"
mkdir -p "${BASE_DIR}"

# 🔥 varying parameter
POS_WEIGHTS=(5 10 20 40)

# ===== BASELINE FIXED =====
SEED=0
MAX_STEPS=8000
LOG_EVERY=200
SAVE_EVERY=500

PROJ_LOSS_WEIGHT=0.1
PROJ_WARMUP_STEPS=150
PROJ_RAMP_STEPS=1000

ACT_LOSS_WEIGHT=3.0
ACT_POS_FRACTION=0.05
ACT_SPARSITY_WEIGHT=5e-4
ACT_TV_WEIGHT=1e-6
ACT_SAMPLES=32768
CT_LOSS_WEIGHT=1e-4

for P in "${POS_WEIGHTS[@]}"; do
  TAG="posW_${P}"
  RUN_DIR="${BASE_DIR}/${TAG}"
  RUN_OUT="${RUN_DIR}/results_spect"
  TRAIN_LOG="${RUN_DIR}/train.log"

  mkdir -p "${RUN_DIR}" "${RUN_OUT}"

  set -x
  ( cd "${ROOT}" && srun ${PYTHON_BIN} "${TRAIN_PY}" \
      --config "${BASE_CFG}" \
      --hybrid \
      --seed "${SEED}" \
      --max-steps "${MAX_STEPS}" \
      --log-every "${LOG_EVERY}" \
      --save-every "${SAVE_EVERY}" \
      --proj-loss-weight "${PROJ_LOSS_WEIGHT}" \
      --proj-warmup-steps "${PROJ_WARMUP_STEPS}" \
      --proj-ramp-steps "${PROJ_RAMP_STEPS}" \
      --act-loss-weight "${ACT_LOSS_WEIGHT}" \
      --act-pos-fraction "${ACT_POS_FRACTION}" \
      --act-pos-weight "${P}" \
      --act-sparsity-weight "${ACT_SPARSITY_WEIGHT}" \
      --act-tv-weight "${ACT_TV_WEIGHT}" \
      --act-samples "${ACT_SAMPLES}" \
      --ct-loss-weight "${CT_LOSS_WEIGHT}" \
    |& tee "${TRAIN_LOG}" )
  set +x
done

echo "POS WEIGHT SWEEP DONE"