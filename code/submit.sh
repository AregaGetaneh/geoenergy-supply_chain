#!/bin/bash
# ============================================================================
# DTU HPC (LSF) submission for any experiment in code/experiments.py.
#
# Set EXPERIMENT to a subcommand of experiments.py, then submit from the
# repository root:
#
#     EXPERIMENT=reconcile bsub < code/submit.sh
#     EXPERIMENT=isoelastic bsub < code/submit.sh
#
# For the sharded, certified core-stability computation, submit as an 8-element
# array and then a dependent combine job:
#
#     bsub -J "geoenergy[1-8]" -env "all, EXPERIMENT=core-shard" < code/submit.sh
#     bsub -w "done(geoenergy)" -env "all, EXPERIMENT=core-combine" < code/submit.sh
#
# Only PYTHON_MODULE (and, if needed, the gurobi version) ever need editing.
# ============================================================================
#BSUB -J geoenergy
#BSUB -q hpc
#BSUB -n 8
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=32GB]"
#BSUB -M 32GB
#BSUB -W 24:00
#BSUB -o logs/%J.out
#BSUB -e logs/%J.err
#BSUB -u ageab@dtu.dk

set -euo pipefail

EXPERIMENT="${EXPERIMENT:-reconcile}"
PROJECT_DIR="$HOME/projects/geoenergy-supply_chain-mip-v4"
PYTHON_MODULE="python3/3.10.13"     # match `module avail python3` on the node

cd "$PROJECT_DIR"
mkdir -p logs outputs outputs/shards tmp

module purge
module load "$PYTHON_MODULE"
module load gurobi/13.0.0 2>/dev/null || echo "NOTE: gurobi module absent, HiGHS fallback will be used"
source .venv/bin/activate

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export GUROBI_THREADS=8
export GUROBI_MEM_GB=24
export GUROBI_METHOD=-1

# Experiment configuration (override on the bsub command line as needed).
export COALITION_TARGET_COUNT="${COALITION_TARGET_COUNT:-30}"
export CORE_MIP_GAP="${CORE_MIP_GAP:-1e-4}"
export SHARD_INDEX="${LSB_JOBINDEX:-1}"     # 1-based array index, 1 for a single job
export SHARD_COUNT="${SHARD_COUNT:-8}"      # must match the array range for core-shard

TMP_JOB_DIR="$PROJECT_DIR/tmp/${LSB_JOBID}_${LSB_JOBINDEX:-0}"
mkdir -p "$TMP_JOB_DIR"
export TMPDIR="$TMP_JOB_DIR"
cleanup() { rm -rf "$TMP_JOB_DIR"; }
trap cleanup EXIT

echo "Experiment: ${EXPERIMENT}"
echo "Job: ${LSB_JOBID}${LSB_JOBINDEX:+[${LSB_JOBINDEX}]}  Host: $(hostname)  Started: $(date)"
python -u code/experiments.py "${EXPERIMENT}"
echo "Finished: $(date)"
