#!/bin/bash
#SBATCH --job-name=sw_agg
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=dgx
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --output=/home/mnguest12/slurm/sw_agg.%j.out
#SBATCH --error=/home/mnguest12/slurm/sw_agg.%j.err
#SBATCH --chdir=/home/mnguest12/projects/thesis/pieNeRF

set -euo pipefail

CONDA_ENV=${CONDA_ENV:-totalseg}
CONDA_ACTIVATE=${CONDA_ACTIVATE:-/home/mnguest12/mambaforge/bin/activate}
source "${CONDA_ACTIVATE}"
conda activate "${CONDA_ENV}"

python scripts/aggregate_sweep.py