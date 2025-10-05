#!/bin/bash
#SBATCH --job-name=recon_fbp
#SBATCH --output=/home/<USER>/slurm/recon_fbp.%j.out
#SBATCH --error=/home/<USER>/slurm/recon_fbp.%j.err
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

set -e
echo "[INFO] SLURM Job gestartet auf $(hostname)"

# Environment aktivieren
source ~/mambaforge/etc/profile.d/conda.sh
conda activate totalseg

# Prüfe Python-Pakete
python - <<'PYCHECK'
import importlib
for pkg in ["itk","itk-rtk","nibabel","numpy","scipy","h5py","matplotlib"]:
    try:
        importlib.import_module(pkg)
        print(f"[OK] {pkg}")
    except Exception as e:
        print(f"[WARN] {pkg} fehlt: {e}")
PYCHECK

# Ausführung
cd /home/<USER>/projects/thesis/TotalSegmentator

python recon_fbp_manual.py \
    --mat ct_proj_stack.mat \
    --proj proj.txt \
    --out ct_recon_fbp.nii \
    --nx 512 --ny 512 --nz 512 \
    --sx 500 --sy 500 --sz 800 \
    --swap_uv --window shepp-logan --cutoff 1.0

echo "[INFO] FBP-Rekonstruktion abgeschlossen."