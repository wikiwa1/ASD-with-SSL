#!/usr/bin/env bash
# One-shot global audio-JEPA: allocate N interactive GPU nodes, pretrain one JEPA encoder
# on all normal MIMII (DDP), then score every target by latent prediction error and
# aggregate the machine x SNR AUC/pAUC table. Comparable to the AE baseline (AUC ~0.717).
#
# Usage:
#   bash audio_ssl/scripts/run_jepa_salloc.sh [NODES] [WALLTIME]
#   bash audio_ssl/scripts/run_jepa_salloc.sh            # 1 node (4 GPUs), 4h
#   bash audio_ssl/scripts/run_jepa_salloc.sh 4 04:00:00 # 4 nodes (16 GPUs)
set -euo pipefail

export REPO=/pscratch/sd/d/dfarough/ASD-with-SSL
export CONFIG="${CONFIG:-audio_ssl/configs/jepa_baseline.yaml}"
export NODES="${1:-1}"
export GPUS_PER_NODE=4
export NTASKS=$(( NODES * GPUS_PER_NODE ))
WALLTIME="${2:-04:00:00}"
ACCOUNT="${NERSC_ACCOUNT:-m4539}"
PY=/global/homes/d/dfarough/.conda/envs/asd-ssl/bin/python

cd "$REPO"
# Run-dir base comes from the config's output.directory so experiment configs (e.g.
# jepa_fan.yaml -> outputs/jepa_fan_<...>) are named consistently with their feature cache.
export RUNDIR="$("$PY" -c 'import sys; from audio_ssl.src.utils.config import load_config; from audio_ssl.src.utils.runs import create_run_dir; print(create_run_dir(load_config(sys.argv[1])["output"]["directory"]))' "$CONFIG")"

echo "RUN DIR : $RUNDIR"
echo "PRETRAIN: DDP over $NTASKS GPUs ($NODES nodes x $GPUS_PER_NODE)"
echo "WALLTIME: $WALLTIME   ACCOUNT: $ACCOUNT   CONFIG: $CONFIG"

salloc -N "$NODES" -C gpu --gpus-per-node="$GPUS_PER_NODE" -q interactive -t "$WALLTIME" -A "$ACCOUNT" \
  bash "$REPO/audio_ssl/scripts/run_jepa_inside.sh"
