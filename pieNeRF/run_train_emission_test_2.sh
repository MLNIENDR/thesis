#!/bin/bash
#SBATCH --job-name=emission-train-test2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=dgx
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:A100:1
#SBATCH --mem=64G
#SBATCH --time=10:00:00
#SBATCH --output=/home/mnguest12/slurm/emission_train_test2.%j.out
#SBATCH --error=/home/mnguest12/slurm/emission_train_test2.%j.err
#SBATCH --chdir=/home/mnguest12/projects/thesis/pieNeRF

set -euo pipefail

PYTHON_BIN=${PYTHON_BIN:-python3}

echo "🚀 Starting Emission-NeRF training job (TEST 2: encoder counts input) on $HOSTNAME"
echo "📅 Job started at: $(date)"
echo "🧠 GPUs assigned: ${SLURM_JOB_GPUS:-<unset>}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"

# 1) Conda env
source /home/mnguest12/mambaforge/bin/activate totalseg

# 2) Project root
cd /home/mnguest12/projects/thesis/pieNeRF

# 3) GPU info
nvidia-smi

# 4) Unique results dir for this run (job-id based)
RESULTS_DIR="/home/mnguest12/projects/thesis/pieNeRF/results_spect_test_2/run_${SLURM_JOB_ID}"
mkdir -p "${RESULTS_DIR}/checkpoints" "${RESULTS_DIR}/logs"
echo "📁 RESULTS_DIR=${RESULTS_DIR}"

# Record basic metadata
{
  echo "date: $(date)"
  echo "host: ${HOSTNAME}"
  echo "job_id: ${SLURM_JOB_ID}"
  echo "cuda_visible_devices: ${CUDA_VISIBLE_DEVICES:-<unset>}"
  echo "git_rev: $(git rev-parse --short HEAD 2>/dev/null || echo '<no-git>')"
  echo "python: $(${PYTHON_BIN} -V 2>&1 || true)"
} > "${RESULTS_DIR}/run_meta.txt"

# 5) Robust symlink redirect for ./results_spect
RESTORE_DIR=""
cleanup_link() {
  rm -f "results_spect" || true
  if [ -n "${RESTORE_DIR:-}" ] && [ -e "${RESTORE_DIR}" ]; then
    mv "${RESTORE_DIR}" "results_spect"
  fi
}
trap cleanup_link EXIT

if [ -e "results_spect" ] && [ ! -L "results_spect" ]; then
  RESTORE_DIR="results_spect.__backup__.$(date +%Y%m%d_%H%M%S)"
  echo "⚠️ Found existing ./results_spect (not symlink). Backing up to: ${RESTORE_DIR}"
  mv "results_spect" "${RESTORE_DIR}"
fi

rm -f "results_spect"
ln -s "${RESULTS_DIR}" "results_spect"
echo "🔗 Linked ./results_spect -> ${RESULTS_DIR}"

# 6) Ensure TEST2 config exists: encoder input comes from counts
CFG_IN="configs/spect.yaml"
CFG_OUT="configs/spect_encoderCounts_projCounts.yaml"

if [ ! -f "${CFG_OUT}" ]; then
  echo "🛠️ Creating ${CFG_OUT} from ${CFG_IN} (set data.proj_input_source=counts)"
  ${PYTHON_BIN} - <<'PY'
import yaml
cfg_in  = "configs/spect.yaml"
cfg_out = "configs/spect_encoderCounts_projCounts.yaml"

with open(cfg_in, "r") as f:
    cfg = yaml.safe_load(f)

cfg.setdefault("data", {})
cfg["data"]["proj_input_source"] = "counts"

with open(cfg_out, "w") as f:
    yaml.safe_dump(cfg, f, sort_keys=False)

print("wrote", cfg_out)
print("data.proj_input_source =", cfg["data"]["proj_input_source"])
PY
fi

echo "🏋️ Running train_emission.py (TEST 2)..."

exit_code=0

# --- core schedule knobs (after proj-ramp fix) ---
# max proj weight = --proj-loss-weight
# min proj weight = --proj-weight-min (starts at min after warmup, ramps to max)
# warmup: 1500 steps (proj inactive)
# ramp: 5000 steps (reasonable within an 8000-step run)

CMD=(srun ${PYTHON_BIN} -u train_emission.py
  --config "${CFG_OUT}"
  --hybrid
  --encoder-use-ct
  --seed 0
  --max-steps 8000
  --log-every 200
  --save-every 1000

  --proj-target-source counts
  --poisson-rate-mode identity
  --proj-loss-type poisson
  --proj-loss-weight 2e-3
  --proj-weight-min 1e-4
  --proj-warmup-steps 1500
  --proj-ramp-steps 5000

  --act-loss-weight 3.0
  --act-pos-fraction 0.05
  --act-pos-weight 10.0
  --act-sparsity-weight 2e-3
  --act-tv-weight 1e-5
  --act-samples 32768

  --ct-loss-weight 1e-4

  --z-enc-alpha 0.5
  --encoder-proj-transform none
  --proj-scale-source compute_p99

  --tv-weight 5e-4

  --debug-latent-stats
  --grad-stats-every 50
  --debug-grad-terms-every 50
  --clip-grad-decoder 1.0

  --final-act-compare
)

# Save command for reproducibility
printf '%q ' "${CMD[@]}" > "${RESULTS_DIR}/command.sh"
echo >> "${RESULTS_DIR}/command.sh"

"${CMD[@]}" || exit_code=$?

echo "python_exit=$exit_code"

if [ $exit_code -ne 0 ]; then
  echo "[ERROR] Training failed with exit_code=$exit_code"
  exit $exit_code
fi

echo "✅ Training finished at: $(date)"
echo "📁 Outputs under: ${RESULTS_DIR}"