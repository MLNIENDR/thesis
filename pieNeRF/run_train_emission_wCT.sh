#!/bin/bash
#SBATCH --job-name=emission-train-wCT
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=dgx
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:A100:1
#SBATCH --mem=64G
#SBATCH --time=10:00:00
#SBATCH --output=/home/mnguest12/slurm/emission_train_wCT.%j.out
#SBATCH --error=/home/mnguest12/slurm/emission_train_wCT.%j.err
#SBATCH --chdir=/home/mnguest12/projects/thesis/pieNeRF

set -euo pipefail

PYTHON_BIN=${PYTHON_BIN:-python3}

echo "🚀 Starting Emission-NeRF training job (wCT encoder) on $HOSTNAME"
echo "📅 Job started at: $(date)"
echo "🧠 GPUs assigned: ${SLURM_JOB_GPUS}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

# 1️⃣ Conda-Umgebung aktivieren
source /home/mnguest12/mambaforge/bin/activate totalseg

# 2️⃣ Ins Projektverzeichnis
cd /home/mnguest12/projects/thesis/pieNeRF

# 3️⃣ GPU-Info ausgeben
nvidia-smi

# 4️⃣ Eigener Results-Ordner
RESULTS_DIR="/home/mnguest12/projects/thesis/pieNeRF/results_spect_wCT"
mkdir -p "${RESULTS_DIR}"
mkdir -p "${RESULTS_DIR}/checkpoints" "${RESULTS_DIR}/logs"

echo "📁 RESULTS_DIR=${RESULTS_DIR}"

# 5️⃣ Robustheits-Fallback: falls Code hart nach results_spect schreibt,
#    leiten wir results_spect -> results_spect_wCT um (nur für diesen Run).
#    Wenn results_spect bereits existiert, sichern wir es weg und stellen es nach dem Run wieder her.
RESTORE_DIR=""
if [ -e "results_spect" ] && [ ! -L "results_spect" ]; then
  RESTORE_DIR="results_spect.__backup__.$(date +%Y%m%d_%H%M%S)"
  echo "⚠️ Found existing ./results_spect (not a symlink). Backing up to: ${RESTORE_DIR}"
  mv "results_spect" "${RESTORE_DIR}"
fi

# Erzeuge/ersetze Symlink
if [ -L "results_spect" ] || [ ! -e "results_spect" ]; then
  rm -f "results_spect"
  ln -s "${RESULTS_DIR}" "results_spect"
  echo "🔗 Linked ./results_spect -> ${RESULTS_DIR}"
else
  echo "❌ Unexpected state for ./results_spect"
  exit 1
fi

# 6️⃣ Training starten
echo "🏋️ Running train_emission.py (wCT)..."
exit_code=0

srun ${PYTHON_BIN} -u train_emission.py \
  --config configs/spect.yaml \
  --hybrid \
  --encoder-use-ct \
  --seed 0 \
  --max-steps 8000 \
  --log-every 200 \
  --save-every 1000 \
  --proj-target-source counts \
  --poisson-rate-mode identity \
  --proj-loss-type poisson \
  --proj-loss-weight 0.005 \
  --proj-warmup-steps 1500 \
  --proj-ramp-steps 20000 \
  --act-loss-weight 3.0 \
  --act-pos-fraction 0.05 \
  --act-pos-weight 10.0 \
  --act-sparsity-weight 2e-3 \
  --act-tv-weight 1e-4 \
  --act-samples 32768 \
  --ct-loss-weight 1e-4 \
  --z-enc-alpha 0.5 \
  --debug-latent-stats \
  --grad-stats-every 50 \
  --final-act-compare || exit_code=$?

echo "python_exit=$exit_code"

# 7️⃣ Symlink zurückbauen / Backup restaurieren
echo "🧹 Restoring results_spect path..."
rm -f "results_spect"

if [ -n "${RESTORE_DIR}" ]; then
  echo "↩️ Restoring original ./results_spect from ${RESTORE_DIR}"
  mv "${RESTORE_DIR}" "results_spect"
fi

if [ $exit_code -ne 0 ]; then
  echo "[ERROR] Training failed with exit_code=$exit_code"
  exit $exit_code
fi

echo "✅ Training finished at: $(date)"
echo "📁 Outputs should be under: ${RESULTS_DIR}"