#!/usr/bin/env bash
# Per-task worker for the multi-GPU MIMII baseline sweep. srun launches one of these
# per GPU. Each worker trains + evaluates its round-robin slice of the target list into
# the shared run folder ($RUNDIR). Requires env: REPO, CONFIG, RUNDIR, TARGETS_FILE.
set -uo pipefail

SHARD="${SLURM_PROCID:-0}"     # 0 .. NTASKS-1, this task's global index
NSHARD="${SLURM_NTASKS:-1}"    # total workers
LOCALID="${SLURM_LOCALID:-0}"  # 0 .. gpus_per_node-1, this task's index on its node

# Pin to one GPU, then scrub SLURM_* so each per-target run is a standalone single-GPU
# job — not DDP across the 16 tasks, and the feature-cache rank logic stays rank 0.
export CUDA_VISIBLE_DEVICES="$LOCALID"
for v in $(compgen -v | grep '^SLURM_'); do unset "$v"; done

module load conda
conda activate asd-ssl
cd "$REPO"

mapfile -t ALL < "$TARGETS_FILE"
N=${#ALL[@]}
echo "[worker $SHARD/$NSHARD] node=$(hostname) gpu=$CUDA_VISIBLE_DEVICES  ${N} total targets"

count=0
i=$SHARD
while [ "$i" -lt "$N" ]; do
  TARGET="${ALL[$i]}"
  echo "[worker $SHARD] === train+eval $TARGET ==="
  if python -m audio_ssl.scripts.train_autoencoder \
        --config "$CONFIG" --run-dir "$RUNDIR" --target-dir "$TARGET" \
        --accelerator gpu --devices 1 --num-nodes 1 \
     && python -m audio_ssl.scripts.eval_autoencoder \
        --config "$CONFIG" --run-dir "$RUNDIR" --target-dir "$TARGET" --fragments; then
    count=$(( count + 1 ))
  else
    echo "[worker $SHARD] FAILED on $TARGET (continuing)"
  fi
  i=$(( i + NSHARD ))
done
echo "[worker $SHARD] finished $count target(s)"
