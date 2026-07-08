#!/usr/bin/env bash
# Runs inside the salloc allocation: DINO pretrain (DDP) -> embedding-distance eval ->
# aggregate. Requires env: REPO, CONFIG, RUNDIR, NODES, GPUS_PER_NODE, NTASKS.
# Optional: CKPT, TAG, METHOD, PCADIM.
set -uo pipefail
mkdir -p "$RUNDIR/logs"

echo "=== DINO pretrain: DDP over $NTASKS GPUs ($NODES nodes x $GPUS_PER_NODE) -> $RUNDIR ==="
TRAIN_MODULE=audio_ssl.scripts.train_dino \
srun -N "$NODES" --ntasks="$NTASKS" --ntasks-per-node="$GPUS_PER_NODE" \
     --cpus-per-task=16 --gpus-per-node="$GPUS_PER_NODE" --gpu-bind=none --export=ALL \
     bash "$REPO/audio_ssl/scripts/jepa_train_task.sh" 2>&1 | tee "$RUNDIR/logs/train.log"

echo "=== embedding-distance eval (single GPU) ==="
srun -N1 --ntasks=1 --gpus-per-task=1 --cpus-per-task=16 --export=ALL \
     bash -lc 'set -e; module load conda; conda activate asd-ssl; cd "$REPO";
               python -m audio_ssl.scripts.eval_jepa_embedding --config "$CONFIG" --run-dir "$RUNDIR" ${CKPT:+--checkpoint "$CKPT"} ${TAG:+--tag "$TAG"} ${METHOD:+--method "$METHOD"} ${PCADIM:+--pca-dim "$PCADIM"}'

echo "=== aggregate machine x SNR table ==="
module load conda
conda activate asd-ssl
cd "$REPO"
python -m audio_ssl.scripts.aggregate_results --config "$CONFIG" --run-dir "$RUNDIR" --result-file "result_embedding${TAG:+_$TAG}.yaml"
echo "=== done. run folder: $RUNDIR ==="
