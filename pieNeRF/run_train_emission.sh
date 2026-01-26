#!/bin/bash
#SBATCH --job-name=emission-train
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=/home/mnguest12/slurm/emission_train.%j.out
#SBATCH --error=/home/mnguest12/slurm/emission_train.%j.err
#SBATCH --chdir=/home/mnguest12/projects/thesis/pieNeRF

PYTHON_BIN=${PYTHON_BIN:-python3}

echo "🚀 Starting Emission-NeRF training job on $HOSTNAME"
echo "📅 Job started at: $(date)"
echo "🧠 GPUs assigned: ${SLURM_JOB_GPUS}"

# 1️⃣ Conda-Umgebung aktivieren
source /home/mnguest12/mambaforge/bin/activate totalseg

# 2️⃣ Ins Projektverzeichnis
cd /home/mnguest12/projects/thesis/pieNeRF

# 3️⃣ Optional: GPU-Info ausgeben
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
nvidia-smi

# 4️⃣ Training starten
echo "🏋️ Running train_emission.py..."
srun ${PYTHON_BIN} -u train_emission.py \
  --config configs/spect.yaml \
  --hybrid \
  --max-steps 500 \
  --act-loss-weight 1.0 \
  --act-norm-source none \
  --act-pos-fraction 0.8 \
  --act-pos-threshold 1e-6 \
  --act-pos-weight 5.0 \
  \
  --proj-loss-weight 0.02 \
  --proj-weight-min 0.0 \
  --proj-warmup-steps 200 \
  --proj-ramp-steps 800 \
  --proj-gain-source z_enc \
  --gain-reg-weight 1e-4
echo "✅ Training finished at: $(date)"

# --- Hybrid-Optionen (AP/PA -> Encoder -> Conditioning) ---
#   --hybrid
#   --proj-loss-type poisson|sqrt_mse
#   --proj-loss-weight 0.1
#   --proj-warmup-steps 0
#   --proj-weight-min 0.005
#   --proj-ramp-steps 200
#   # Tipp (kleine Datensaetze): --proj-loss-weight 0.05 --proj-ramp-steps 500
#   --proj-target-source counts|norm
#   --proj-gain-source z_enc|scalar|none
#   --encoder-proj-transform log1p|sqrt|none
#   --proj-scale-source meta_p99|compute_p99|sumcounts|none
#   --act-norm-source p99_global|p99_scan|fixed
#   --act-norm-value 1.0
#   --encoder-use-ct
#   --z-enc-alpha 0.1
#   --smoke-test
