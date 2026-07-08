#!/usr/bin/env bash
# Runs inside the salloc allocation: pretrain the global JEPA (DDP across all GPUs),
# evaluate it (single GPU), then aggregate the machine x SNR table.
# Requires env: REPO, CONFIG, RUNDIR, NODES, GPUS_PER_NODE, NTASKS.
set -uo pipefail

echo "=== JEPA pretrain: DDP over $NTASKS GPUs ($NODES nodes x $GPUS_PER_NODE) -> $RUNDIR ==="
# Each task must SEE all GPUs on its node (Lightning picks one per local rank); do NOT
# bind 1 GPU/task here or `--devices N` fails with "machine only has [0]".
srun -N "$NODES" --ntasks="$NTASKS" --ntasks-per-node="$GPUS_PER_NODE" \
     --cpus-per-task=16 --gpus-per-node="$GPUS_PER_NODE" --gpu-bind=none --export=ALL \
     bash "$REPO/audio_ssl/scripts/jepa_train_task.sh"

echo "=== JEPA eval: latent prediction error over all targets (single GPU) ==="
srun -N1 --ntasks=1 --gpus-per-task=1 --cpus-per-task=16 --export=ALL \
     bash "$REPO/audio_ssl/scripts/jepa_eval_task.sh"

echo "=== aggregate machine x SNR table ==="
module load conda
conda activate asd-ssl
cd "$REPO"
python -m audio_ssl.scripts.aggregate_results --config "$CONFIG" --run-dir "$RUNDIR"
echo "=== done. run folder: $RUNDIR ==="
