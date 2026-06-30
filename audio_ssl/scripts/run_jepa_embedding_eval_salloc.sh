#!/usr/bin/env bash
# One-shot embedding-distance evaluation of an already-trained JEPA run (no retraining):
# allocate 1 GPU, score every target by one-class distance on frozen-encoder embeddings,
# and aggregate the machine x SNR table into summary_embedding.yaml.
#
# Usage:
#   bash audio_ssl/scripts/run_jepa_embedding_eval_salloc.sh [RUN_DIR] [WALLTIME]
#   bash audio_ssl/scripts/run_jepa_embedding_eval_salloc.sh                     # latest run
#   bash audio_ssl/scripts/run_jepa_embedding_eval_salloc.sh audio_ssl/outputs/jepa_baseline_20260629_vivid_delta_1678
set -euo pipefail

export REPO=/pscratch/sd/d/dfarough/ASD-with-SSL
export CONFIG="${CONFIG:-audio_ssl/configs/jepa_baseline.yaml}"
export RUNDIR="${1:-audio_ssl/outputs/jepa_baseline_latest}"
WALLTIME="${2:-00:40:00}"
ACCOUNT="${NERSC_ACCOUNT:-m4539}"

cd "$REPO"
echo "RUN DIR : $RUNDIR"
echo "WALLTIME: $WALLTIME   ACCOUNT: $ACCOUNT   CONFIG: $CONFIG"

salloc -N1 -C gpu -G 1 -q interactive -t "$WALLTIME" -A "$ACCOUNT" \
  bash "$REPO/audio_ssl/scripts/run_jepa_embedding_inside.sh"
