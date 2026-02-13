#!/bin/bash
#SBATCH --job-name=actW_sweep
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=dgx
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:A100:1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/home/mnguest12/slurm/sweep_actW.%j.out
#SBATCH --error=/home/mnguest12/slurm/sweep_actW.%j.err
#SBATCH --chdir=/home/mnguest12/projects/thesis/pieNeRF

set -euo pipefail

source /home/mnguest12/mambaforge/bin/activate
conda activate totalseg
PYTHON_BIN=python

ROOT="/home/mnguest12/projects/thesis/pieNeRF"
TRAIN_PY="${ROOT}/train_emission.py"
BASE_CFG="${ROOT}/configs/spect.yaml"

# make repo importable
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"

BASE_DIR="${ROOT}/results_spect_sweep"
mkdir -p "${BASE_DIR}"

# ✅ choose act loss weights to sweep (baseline you used often is 3.0)
ACT_WEIGHTS=(1.0 2.0 3.0 5.0)

for W in "${ACT_WEIGHTS[@]}"; do
  TAG="actW_${W}"
  RUN_DIR="${BASE_DIR}/${TAG}"
  RUN_OUT="${RUN_DIR}/results_spect"
  RUN_CFG="${RUN_DIR}/spect_${TAG}.yaml"
  TRAIN_LOG="${RUN_DIR}/train.log"
  FINAL_CKPT="${RUN_OUT}/checkpoints/checkpoint_step08000.pt"

  echo "=============================="
  echo "Starting run with act-loss-weight=${W}"
  echo "RUN_DIR=${RUN_DIR}"
  echo "RUN_OUT=${RUN_OUT}"
  echo "RUN_CFG=${RUN_CFG}"
  echo "FINAL_CKPT=${FINAL_CKPT}"
  echo "PYTHONPATH=${PYTHONPATH}"
  echo "=============================="

  mkdir -p "${RUN_DIR}" "${RUN_OUT}"

  # ✅ skip if already finished
  if [ -f "${FINAL_CKPT}" ]; then
    echo "[sweep] SKIP ${TAG}: already finished (${FINAL_CKPT} exists)"
    continue
  fi

  # ---- write per-run config (patch multiple possible outdir keys) ----
  ${PYTHON_BIN} - <<PY
import os, yaml

base_cfg = r"${BASE_CFG}"
run_cfg  = r"${RUN_CFG}"
run_out  = r"${RUN_OUT}"
tag      = r"${TAG}"
actW     = float(r"${W}")

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

for p in [
    ("training","outdir"),
    ("logging","outdir"),
    ("trainer","outdir"),
    ("experiment","outdir"),
    ("paths","outdir"),
]:
    if set_if_path(cfg, list(p), run_out):
        patched.append(".".join(p))

# set act loss weight in config too (belt + suspenders)
cfg.setdefault("training", {})
cfg["training"]["act_loss_weight"] = actW

os.makedirs(os.path.dirname(run_cfg), exist_ok=True)
with open(run_cfg, "w") as f:
    yaml.safe_dump(cfg, f, sort_keys=False)

print(f"[sweep] wrote {run_cfg}")
print(f"[sweep] expname = {cfg.get('expname')}")
print(f"[sweep] outdir  = {run_out}")
print(f"[sweep] patched keys: {patched}")
print(f"[sweep] training.act_loss_weight = {cfg['training'].get('act_loss_weight')}")
PY

  # ---- TRAIN ----
  echo ""
  echo "[sweep] TRAIN (live log -> slurm + ${TRAIN_LOG})"
  echo ""

  set -x
  ( cd "${ROOT}" && stdbuf -oL -eL srun --export=ALL,PYTHONPATH="${PYTHONPATH}" ${PYTHON_BIN} -u "${TRAIN_PY}" \
      --config "${RUN_CFG}" \
      --hybrid \
      --seed 0 \
      --max-steps 8000 \
      --log-every 200 \
      --save-every 500 \
      --proj-target-source counts \
      --poisson-rate-mode identity \
      --proj-loss-type poisson \
      --proj-loss-weight 0.1 \
      --proj-warmup-steps 150 \
      --proj-ramp-steps 1000 \
      --act-loss-weight "${W}" \
      --act-pos-fraction 0.05 \
      --act-pos-weight 10.0 \
      --act-sparsity-weight 5e-4 \
      --act-tv-weight 1e-6 \
      --act-samples 16384 \
      --ct-loss-weight 1e-4 \
      --final-act-compare \
    |& tee "${TRAIN_LOG}" )
  set +x

  # ---- HARD GUARD: must write into sweep outdir ----
  if grep -q "Output-Ordner: ${RUN_OUT}" "${TRAIN_LOG}"; then
    echo "[guard] TRAIN wrote into sweep outdir ✅"
  else
    echo "[guard][FATAL] TRAIN wrote to WRONG outdir. Found:"
    grep -m 1 "Output-Ordner:" "${TRAIN_LOG}" || true
    echo "Aborting to prevent baseline overwrite."
    exit 3
  fi

  # ---- finished marker for downstream postproc jobs ----
  if [ -f "${FINAL_CKPT}" ]; then
    echo "[sweep] finished ${TAG} ✅  (${FINAL_CKPT})"
  else
    echo "[sweep][WARN] ${TAG} ended but final ckpt missing: ${FINAL_CKPT}"
  fi

  echo ""
  echo "Training finished for ${TAG}"
  echo "NOTE: postprocessing is decoupled (submit separately)."
  echo ""
done

echo "SWEEP FINISHED (training only)"