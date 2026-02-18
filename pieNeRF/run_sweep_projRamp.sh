#!/bin/bash
#SBATCH --job-name=ramp_sweep
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=dgx
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:A100:1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/home/mnguest12/slurm/sweep_ramp.%j.out
#SBATCH --error=/home/mnguest12/slurm/sweep_ramp.%j.err
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

BASE_DIR="${ROOT}/results_spect_sweep_ramp"
mkdir -p "${BASE_DIR}"

# 🔥 varying parameter
RAMP_STEPS=(250 500 1000 2000)

# ===== BASELINE FIXED =====
SEED=0
MAX_STEPS=8000
LOG_EVERY=200
SAVE_EVERY=500

PROJ_LOSS_WEIGHT=0.1
PROJ_WARMUP_STEPS=150

ACT_LOSS_WEIGHT=3.0
ACT_POS_FRACTION=0.05
ACT_POS_WEIGHT=10
ACT_SPARSITY_WEIGHT=5e-4
ACT_TV_WEIGHT=1e-6
ACT_SAMPLES=32768
CT_LOSS_WEIGHT=1e-4

for R in "${RAMP_STEPS[@]}"; do
  TAG="ramp_${R}"
  RUN_DIR="${BASE_DIR}/${TAG}"
  RUN_OUT="${RUN_DIR}/results_spect"
  RUN_CFG="${RUN_DIR}/spect_${TAG}.yaml"
  TRAIN_LOG="${RUN_DIR}/train.log"
  FINAL_CKPT="${RUN_OUT}/checkpoints/checkpoint_step08000.pt"

  echo "=============================="
  echo "Starting run with proj-ramp-steps=${R}"
  echo "RUN_OUT=${RUN_OUT}"
  echo "=============================="

  mkdir -p "${RUN_DIR}" "${RUN_OUT}"

  if [ -f "${FINAL_CKPT}" ]; then
    echo "[sweep] SKIP ${TAG}: already finished"
    continue
  fi

  ${PYTHON_BIN} - <<PY
import os, yaml
base_cfg = r"${BASE_CFG}"
run_cfg  = r"${RUN_CFG}"
run_out  = r"${RUN_OUT}"
tag      = r"${TAG}"
ramp     = int(r"${R}")

with open(base_cfg, "r") as f:
    cfg = yaml.safe_load(f)

def set_if_path(cfg, path, value):
    cur = cfg
    for k in path[:-1]:
        if not isinstance(cur, dict) or k not in cur:
            return False
        cur = cur[k]
    if isinstance(cur, dict) and path[-1] in cur:
        cur[path[-1]] = value
        return True
    return False

cfg["expname"] = f"spect_emission_{tag}"

patched = []
cfg["outdir"] = run_out; patched.append("outdir")
for p in [("training","outdir"),("logging","outdir"),("trainer","outdir"),("experiment","outdir"),("paths","outdir")]:
    if set_if_path(cfg, list(p), run_out):
        patched.append(".".join(p))

cfg.setdefault("training", {})
cfg["training"]["proj_ramp_steps"] = ramp

os.makedirs(os.path.dirname(run_cfg), exist_ok=True)
with open(run_cfg, "w") as f:
    yaml.safe_dump(cfg, f, sort_keys=False)

print("[sweep] wrote", run_cfg)
print("[sweep] patched:", patched)
print("[sweep] training.proj_ramp_steps =", cfg["training"]["proj_ramp_steps"])
PY

  set -x
  ( cd "${ROOT}" && stdbuf -oL -eL srun --export=ALL,PYTHONPATH="${PYTHONPATH}" ${PYTHON_BIN} -u "${TRAIN_PY}" \
      --config "${RUN_CFG}" \
      --hybrid \
      --seed "${SEED}" \
      --max-steps "${MAX_STEPS}" \
      --log-every "${LOG_EVERY}" \
      --save-every "${SAVE_EVERY}" \
      --proj-loss-weight "${PROJ_LOSS_WEIGHT}" \
      --proj-warmup-steps "${PROJ_WARMUP_STEPS}" \
      --proj-ramp-steps "${R}" \
      --act-loss-weight "${ACT_LOSS_WEIGHT}" \
      --act-pos-fraction "${ACT_POS_FRACTION}" \
      --act-pos-weight "${ACT_POS_WEIGHT}" \
      --act-sparsity-weight "${ACT_SPARSITY_WEIGHT}" \
      --act-tv-weight "${ACT_TV_WEIGHT}" \
      --act-samples "${ACT_SAMPLES}" \
      --ct-loss-weight "${CT_LOSS_WEIGHT}" \
    |& tee "${TRAIN_LOG}" )
  set +x

  if grep -q "Output-Ordner: ${RUN_OUT}" "${TRAIN_LOG}"; then
    echo "[guard] TRAIN wrote into sweep outdir ✅"
  else
    echo "[guard][FATAL] WRONG outdir"
    grep -m 1 "Output-Ordner:" "${TRAIN_LOG}" || true
    exit 3
  fi

  [ -f "${FINAL_CKPT}" ] && echo "[sweep] finished ${TAG} ✅" || echo "[sweep][WARN] final ckpt missing"
done

echo "RAMP SWEEP DONE"
