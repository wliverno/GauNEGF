#!/bin/bash
#SBATCH --job-name=3DFermi
#SBATCH --account=anantram
#SBATCH --partition=ckpt-g2
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --time=02:00:00
#SBATCH --mem=64GB
#SBATCH --output=/gscratch/anantram/willll/NEGFCode/tests/3d_fermi_%j.out
#SBATCH --mail-type=END
#SBATCH --mail-user=willll@uw.edu

set -e

module load anantram/gdv
module load cuda
unset LD_LIBRARY_PATH

source /gscratch/anantram/willll/miniconda3/etc/profile.d/conda.sh
conda activate /gscratch/anantram/willll/NEGF

echo "=== Node: $(hostname) ==="
echo "=== GPU: ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "=== Starting at $(date) ==="

cd /gscratch/anantram/willll/NEGFCode
python3 -u tests/run_3d_fermi_check.py

echo "=== Done at $(date) ==="
exit 0
