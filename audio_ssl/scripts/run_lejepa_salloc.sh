#!/usr/bin/env bash
# One-shot global LeJEPA (Balestriero & LeCun 2025): allocate N interactive GPU nodes,
# pretrain one encoder on all normal MIMII via view-invariance + SIGReg (DDP), then score
# every target by one-class distance on frozen embeddings and aggregate the machine x SNR
# table. Comparable to the JEPA embedding eval (AUC ~0.82) and AE baseline (~0.717).
#
# Usage:
#   bash audio_ssl/scripts/run_lejepa_salloc.sh [NODES] [WALLTIME]
#   bash audio_ssl/scripts/run_lejepa_salloc.sh              # 1 node (4 GPUs), 4h
#   bash audio_ssl/scripts/run_lejepa_salloc.sh 4 04:00:00   # 4 nodes (16 GPUs)
#   CONFIG=audio_ssl/configs/lejepa_fan.yaml bash audio_ssl/scripts/run_lejepa_salloc.sh
set -euo pipefail

export REPO=/pscratch/sd/d/dfarough/ASD-with-SSL
export CONFIG="${CONFIG:-audio_ssl/configs/lejepa_baseline.yaml}"
export NODES="${1:-1}"
export GPUS_PER_NODE=4
export NTASKS=$(( NODES * GPUS_PER_NODE ))
export CKPT="${CKPT:-}"       # optional: explicit checkpoint for the eval step
export TAG="${TAG:-}"         # optional: eval output suffix
export METHOD="${METHOD:-}"   # optional: override embedding.method
export PCADIM="${PCADIM:-}"   # optional: override embedding.pca_dim
WALLTIME="${2:-04:00:00}"
ACCOUNT="${NERSC_ACCOUNT:-m4539}"
PY=/global/homes/d/dfarough/.conda/envs/asd-ssl/bin/python

cd "$REPO"
export RUNDIR="$("$PY" -c 'import sys; from audio_ssl.src.utils.config import load_config; from audio_ssl.src.utils.runs import create_run_dir; print(create_run_dir(load_config(sys.argv[1])["output"]["directory"]))' "$CONFIG")"

echo "RUN DIR : $RUNDIR"
echo "PRETRAIN: DDP over $NTASKS GPUs ($NODES nodes x $GPUS_PER_NODE)"
echo "WALLTIME: $WALLTIME   ACCOUNT: $ACCOUNT   CONFIG: $CONFIG"

salloc -N "$NODES" -C gpu --gpus-per-node="$GPUS_PER_NODE" -q interactive -t "$WALLTIME" -A "$ACCOUNT" \
  bash "$REPO/audio_ssl/scripts/run_lejepa_inside.sh"
