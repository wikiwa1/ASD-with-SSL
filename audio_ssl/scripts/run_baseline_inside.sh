#!/usr/bin/env bash
# Runs inside the salloc allocation: one worker task per GPU, then aggregate the table.
# Requires env: REPO, CONFIG, RUNDIR, TARGETS_FILE, NODES, GPUS_PER_NODE, NTASKS.
set -uo pipefail

echo "=== launching $NTASKS workers ($NODES nodes x $GPUS_PER_NODE GPUs) into $RUNDIR ==="
srun -N "$NODES" --ntasks="$NTASKS" --ntasks-per-node="$GPUS_PER_NODE" \
     --gpu-bind=none --export=ALL \
     bash "$REPO/audio_ssl/scripts/run_baseline_worker.sh"

echo "=== all workers returned; aggregating results ==="
module load conda
conda activate asd-ssl
cd "$REPO"
python -m audio_ssl.scripts.aggregate_results --config "$CONFIG" --run-dir "$RUNDIR"
echo "=== done. run folder: $RUNDIR ==="
