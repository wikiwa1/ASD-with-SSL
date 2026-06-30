#!/usr/bin/env bash
# srun task for JEPA pretraining. One process per GPU; Lightning + SLURM coordinate DDP
# (SLURM_* env is intentionally kept). Requires env: REPO, CONFIG, RUNDIR, NODES, GPUS_PER_NODE.
set -e
module load conda
conda activate asd-ssl
cd "$REPO"
python -m audio_ssl.scripts.train_jepa \
  --config "$CONFIG" --run-dir "$RUNDIR" \
  --accelerator gpu --devices "$GPUS_PER_NODE" --num-nodes "$NODES" --strategy ddp
