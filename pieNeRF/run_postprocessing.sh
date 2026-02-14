#!/bin/bash
#SBATCH --job-name=postproc_projW005
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

RUN_DIR="/home/mnguest12/projects/thesis/pieNeRF/results_spect_sweep/gainReg_1e-5/results_spect"
SPLIT_JSON="${RUN_DIR}/split.json"
MANIFEST="/home/mnguest12/projects/thesis/pieNeRF/data/manifest_abs.csv"
CONFIG="/home/mnguest12/projects/thesis/pieNeRF/results_spect_sweep/gainReg_1e-5/spect_gainReg_1e-5.yaml"
OUT_DIR="/home/mnguest12/projects/thesis/pieNeRF/results_spect_sweep/gainReg_1e-5/postproc"
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
echo "run_dir=${RUN_DIR}"
echo "split_json=${SPLIT_JSON}"
echo "manifest=${MANIFEST}"
echo "config=${CONFIG}"
echo "out_dir=${OUT_DIR}"
echo "mask_pattern=${MASK_PATTERN}"
echo "device=${DEVICE}"

source "${CONDA_ACTIVATE}"
conda activate "${CONDA_ENV}"

echo "python=$(which "${PYTHON_BIN}")"
"${PYTHON_BIN}" -V

echo "PWD=$(pwd)"
nvidia-smi || true

# --- sanity checks ---
test -d "${RUN_DIR}" || { echo "[ERROR] RUN_DIR not found: ${RUN_DIR}"; exit 2; }
test -f "${SPLIT_JSON}" || { echo "[ERROR] SPLIT_JSON not found: ${SPLIT_JSON}"; exit 2; }
test -f "${MANIFEST}" || { echo "[ERROR] MANIFEST not found: ${MANIFEST}"; exit 2; }
test -f "${CONFIG}" || { echo "[ERROR] CONFIG not found: ${CONFIG}"; exit 2; }
mkdir -p "${OUT_DIR}"

# ensure imports work (Data_Processing + repo)
export PYTHONPATH="/home/mnguest12/projects/thesis:/home/mnguest12/projects/thesis/pieNeRF:${PYTHONPATH:-}"

echo "🔁 Running postprocessing.py..."
echo "[DBG] Listing ${RUN_DIR}/test_slices:"
ls -lah "${RUN_DIR}/test_slices" || true

# ---- run ----
set -x
srun /usr/bin/time -v "${PYTHON_BIN}" -u postprocessing.py \
  --run-dir "${RUN_DIR}" \
  --split-json "${SPLIT_JSON}" \
  --manifest "${MANIFEST}" \
  --config "${CONFIG}" \
  --out-dir "${OUT_DIR}" \
  --mask-path-pattern "${MASK_PATTERN}" \
  --device "${DEVICE}" \
  --pred-act-per-phantom \
  --pred-act-pattern "test_slices/{phantom}/activity_pred.npy" \
  --render-projections \
  --save-proj-npy \
  --save-proj-png \
  --timing \
  --skip-plots \
  --save-active-organ-plots
set +x

echo "✅ Postprocessing finished at: $(date)"