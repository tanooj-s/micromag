#!/bin/bash
# Run the test suite on one GPU. Build first, on the login node (see CLAUDE.md).
#SBATCH --job-name=micromag-test
#SBATCH --gres=gpu:1
#SBATCH --time=00:10:00
#SBATCH --output=%x-%j.out
##SBATCH --partition=TBD   # uncomment if the cluster has no default GPU partition
##SBATCH --account=TBD
set -euo pipefail

# Must be the same modules the build used, or libstdc++ / libcudart won't match.
module load gcc/8.4.0/1 cuda/12.1

nvidia-smi -L
uv run --no-sync python -c "import micromag, sys; sys.exit(0 if micromag.has_cuda else 'built without CUDA; rebuild per CLAUDE.md')"
uv run --no-sync pytest -v -m ""  # -m "": include the slow tests
