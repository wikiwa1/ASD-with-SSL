#!/usr/bin/env bash
# One-shot: allocate an interactive Perlmutter GPU node, activate the asd-ssl env
# ON the compute node, and launch AE training there. Releases the allocation when
# training finishes.
#
# Usage:
#   bash audio_ssl/scripts/run_training_salloc.sh [TARGET_DIR] [EPOCHS] [WALLTIME]
#
# Defaults run a quick "does it go through" check on one complete target:
#   bash audio_ssl/scripts/run_training_salloc.sh
#   bash audio_ssl/scripts/run_training_salloc.sh dataset/6_dB/pump/id_00 50 01:00:00
set -euo pipefail

export REPO=/pscratch/sd/d/dfarough/ASD-with-SSL
export TARGET="${1:-dataset/0_dB/fan/id_00}"   # complete, good data (not -6_dB/valve)
export EPOCHS="${2:-5}"                          # short by default — just a pipeline check
WALLTIME="${3:-00:30:00}"
ACCOUNT="${NERSC_ACCOUNT:-m4539}"

echo "Requesting interactive GPU node (account=$ACCOUNT, t=$WALLTIME)"
echo "  -> training $TARGET for $EPOCHS epoch(s), Comet tracking per config"

# salloc grabs the allocation on the login node; srun places the work on the GPU
# node. --export=ALL propagates TARGET/EPOCHS/REPO. bash -lc runs a login shell so
# `module`/`conda` initialise exactly like an interactive session.
salloc -N 1 -C gpu -G 1 -q interactive -t "$WALLTIME" -A "$ACCOUNT" \
  srun --ntasks=1 --export=ALL bash -lc '
    set -e
    module load conda
    conda activate asd-ssl
    cd "$REPO"
    echo "=== compute node: $(hostname) ==="
    echo "=== python: $(which python) ==="
    python -c "import torch; print(\"torch\", torch.__version__, \"cuda\", torch.cuda.is_available())"
    python -m audio_ssl.scripts.train_autoencoder \
      --target-dir "$TARGET" \
      --max-epochs "$EPOCHS" \
      --accelerator gpu --devices 1 --num-nodes 1
  '
