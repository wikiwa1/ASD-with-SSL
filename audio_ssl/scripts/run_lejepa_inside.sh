#!/usr/bin/env bash
# Runs inside the salloc allocation: pretrain the global LeJEPA (DDP across all GPUs),
# then embedding-distance eval (single GPU) + aggregate. LeJEPA has no predictor, so
# there is no latent-prediction-error eval step (unlike run_jepa_inside.sh).
# Requires env: REPO, CONFIG, RUNDIR, NODES, GPUS_PER_NODE, NTASKS.
set -uo pipefail
mkdir -p "$RUNDIR/logs"

echo "=== LeJEPA pretrain: DDP over $NTASKS GPUs ($NODES nodes x $GPUS_PER_NODE) -> $RUNDIR ==="
# Each task must SEE all GPUs on its node (Lightning picks one per local rank); do NOT
# bind 1 GPU/task here or `--devices N` fails with "machine only has [0]".
# Tee to a persistent log so a crash traceback survives (pipefail keeps srun's exit code).
TRAIN_MODULE=audio_ssl.scripts.train_lejepa \
srun -N "$NODES" --ntasks="$NTASKS" --ntasks-per-node="$GPUS_PER_NODE" \
     --cpus-per-task=16 --gpus-per-node="$GPUS_PER_NODE" --gpu-bind=none --export=ALL \
     bash "$REPO/audio_ssl/scripts/jepa_train_task.sh" 2>&1 | tee "$RUNDIR/logs/train.log"

echo "=== LeJEPA embedding-distance eval (single GPU) ==="
srun -N1 --ntasks=1 --gpus-per-task=1 --cpus-per-task=16 --export=ALL \
     bash -lc 'set -e; module load conda; conda activate asd-ssl; cd "$REPO";
               python -m audio_ssl.scripts.eval_jepa_embedding --config "$CONFIG" --run-dir "$RUNDIR" ${CKPT:+--checkpoint "$CKPT"} ${TAG:+--tag "$TAG"} ${METHOD:+--method "$METHOD"} ${PCADIM:+--pca-dim "$PCADIM"}'

echo "=== aggregate machine x SNR table ==="
module load conda
conda activate asd-ssl
cd "$REPO"
python -m audio_ssl.scripts.aggregate_results --config "$CONFIG" --run-dir "$RUNDIR" --result-file "result_embedding${TAG:+_$TAG}.yaml"
echo "=== done. run folder: $RUNDIR ==="
