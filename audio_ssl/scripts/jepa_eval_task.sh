#!/usr/bin/env bash
# srun task for JEPA eval (single process, one GPU). No DDP. Requires env: REPO, CONFIG, RUNDIR.
set -e
module load conda
conda activate asd-ssl
cd "$REPO"
python -m audio_ssl.scripts.eval_jepa --config "$CONFIG" --run-dir "$RUNDIR"
