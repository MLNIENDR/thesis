#!/bin/bash
#SBATCH --job-name=projRamp_sweep
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=dgx
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:A100:1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/home/mnguest12/slurm/sweep_projRamp.%j.out
#SBATCH --error=/home/mnguest12/slurm/sweep_projRamp.%j.err
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

BASE_DIR="${ROOT}/results_spect_sweep"
mkdir -p "${BASE_DIR}"

# =========================
# SWEEP: projection ramp steps
# =========================
RAMP_STEPS=(300 600 1000 2000)

for R in "${RAMP_STEPS[@]}"; do
  TAG="projRamp_${R}"
  RUN_DIR="${BASE_DIR}/${TAG}"
  RUN_OUT="${RUN_DIR}/results_spect"
  RUN_CFG="${RUN_DIR}/spect_${TAG}.yaml"
  TRAIN_LOG="${RUN_DIR}/train.log"
  FINAL_CKPT="${RUN_OUT}/checkpoints/checkpoint_step08000.pt"

  echo "=============================="
  echo "Starting run with proj_ramp_steps=${R}"
  echo "RUN_DIR=${RUN_DIR}"
  echo "RUN_OUT=${RUN_OUT}"
  echo "RUN_CFG=${RUN_CFG}"
  echo "FINAL_CKPT=${FINAL_CKPT}"
  echo "=============================="

  mkdir -p "${RUN_DIR}" "${RUN_OUT}"

  # Skip if already finished
  if [ -f "${FINAL_CKPT}" ]; then
    echo "[sweep] SKIP ${TAG}: already finished"
    continue
  fi

  # ---- write per-run config ----
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

for p in [
    ("training","outdir"),
    ("logging","outdir"),
    ("trainer","outdir"),
    ("experiment","outdir"),
    ("paths","outdir"),
]:
    if set_if_path(cfg, list(p), run_out):
        patched.append(".".join(p))

cfg.setdefault("hybrid", {})
cfg["hybrid"]["proj_ramp_steps"] = ramp

os.makedirs(os.path.dirname(run_cfg), exist_ok=True)
with open(run_cfg, "w") as f:
    yaml.safe_dump(cfg, f, sort_keys=False)

print(f"[sweep] wrote {run_cfg}")
print(f"[sweep] expname = {cfg.get('expname')}")
print(f"[sweep] outdir  = {run_out}")
print(f"[sweep] hybrid.proj_ramp_steps = {ramp}")
PY

  # ---- TRAIN ----
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
      --proj-ramp-steps "${R}" \
      --act-loss-weight 3.0 \
      --act-pos-fraction 0.05 \
      --act-pos-weight 10.0 \
      --act-sparsity-weight 5e-4 \
      --act-tv-weight 1e-6 \
      --act-samples 16384 \
      --ct-loss-weight 1e-4 \
      --final-act-compare \
    |& tee "${TRAIN_LOG}" )
  set +x

  # Guard against wrong outdir
  if grep -q "Output-Ordner: ${RUN_OUT}" "${TRAIN_LOG}"; then
    echo "[guard] TRAIN wrote into sweep outdir ✅"
  else
    echo "[guard][FATAL] Wrong outdir detected"
    grep -m 1 "Output-Ordner:" "${TRAIN_LOG}" || true
    exit 3
  fi

  echo "Training finished for ${TAG}"
  echo "Postprocessing submit separately."
  echo ""
done

echo "SWEEP FINISHED (training only)"