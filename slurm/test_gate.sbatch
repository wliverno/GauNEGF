#!/bin/bash
#SBATCH --job-name=crossterm_gate
#SBATCH --account=anantram
#SBATCH --partition=ckpt-all
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gpus=1
#SBATCH --time=2:00:00
#SBATCH --requeue
#SBATCH --mem=64G
#SBATCH --output=/mmfs1/gscratch/anantram/willll/NEGFCode-crossterm/slurm-%j.out
set -eo pipefail
module load anantram/gdv/j26 || true
if ! command -v testrt >/dev/null 2>&1; then
    echo "GUARD: gdv missing after module load; requeueing in 5 min"
    sleep 300; scontrol requeue $SLURM_JOB_ID; exit 0
fi
unset LD_LIBRARY_PATH
source /gscratch/anantram/willll/miniconda3/etc/profile.d/conda.sh
conda activate /gscratch/anantram/willll/NEGF
nvidia-smi --query-gpu=name --format=csv,noheader | sed "s/^/GPU: /"
export PYTHONPATH=/mmfs1/gscratch/anantram/willll/NEGFCode-crossterm
cd /mmfs1/gscratch/anantram/willll/NEGFCode-crossterm/tests
python3 -m pytest . --ignore=test_kpoint_convergence.py -q ${SUITE_ARGS:-}
# Submit with: sbatch --export=ALL,SUITE_ARGS="<pytest args>" slurm/test_gate.sbatch
# (empty SUITE_ARGS = full suite)
