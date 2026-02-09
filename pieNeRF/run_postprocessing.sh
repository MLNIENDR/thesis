#!/bin/bash
#
# SLURM job wrapper to run inference postprocessing (pieNeRF/postprocessing.py) on GPU.
#

#SBATCH --job-name=postprocessing
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

PYTHON_BIN=${PYTHON_BIN:-python3}
RUN_DIR=${RUN_DIR:-results_spect}
SPLIT_JSON=${SPLIT_JSON:-"$RUN_DIR/split.json"}
MANIFEST=${MANIFEST:-data/manifest_abs.csv}
CONFIG=${CONFIG:-configs/spect.yaml}
OUT_DIR=${OUT_DIR:-"$RUN_DIR/postproc"}

# Important: mask path pattern uses {phantom} placeholder
MASK_PATTERN=${MASK_PATTERN:-'/home/mnguest12/projects/thesis/Data_Processing/{phantom}/out/mask.npy'}

# Use GPU in postprocessing.py
DEVICE=${DEVICE:-cuda}

# Threading hygiene (prevents oversubscription on CPU-heavy parts)
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK}

echo "🛠 Starting postprocessing job on ${HOSTNAME}"
echo "📅 Job started at: $(date)"
echo "🧠 GPUs assigned: ${SLURM_JOB_GPUS:-unknown}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "🗂 run_dir=${RUN_DIR}"
echo "🧾 split_json=${SPLIT_JSON}"
echo "🧾 manifest=${MANIFEST}"
echo "🧾 config=${CONFIG}"
echo "🧹 out_dir=${OUT_DIR}"
echo "🩻 mask_pattern=${MASK_PATTERN}"
echo "🧮 device=${DEVICE}"
echo "🧵 cpus_per_task=${SLURM_CPUS_PER_TASK}"

# 1) Activate env
echo "⏱ Activate start: $(date +%s)"
source /home/mnguest12/mambaforge/bin/activate totalseg
echo "⏱ Activate end:   $(date +%s)"

# 2) Project dir
cd /home/mnguest12/projects/thesis/pieNeRF

# 3) GPU info
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi || true

# 4) Run postprocessing
echo "🔁 Running postprocessing.py..."
echo "⏱ Python start:   $(date +%s)"

exit_code=0
srun /usr/bin/time -v ${PYTHON_BIN} -u postprocessing.py \
  --run-dir "$RUN_DIR" \
  --split-json "$SPLIT_JSON" \
  --manifest "$MANIFEST" \
  --config "$CONFIG" \
  --out-dir "$OUT_DIR" \
  --mask-path-pattern "$MASK_PATTERN" \
  --device "$DEVICE" \
  --save-proj-npy \
  --timing \
  || exit_code=$?

echo "python_exit=$exit_code"
echo "⏱ Python end:     $(date +%s)"

if [ $exit_code -ne 0 ]; then
  echo "[ERROR] Postprocessing failed with exit_code=$exit_code"
  exit $exit_code
fi

echo "✅ Postprocessing finished at: $(date)"
