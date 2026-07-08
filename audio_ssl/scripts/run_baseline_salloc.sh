#!/usr/bin/env bash
# One-shot full MIMII autoencoder baseline across many GPUs.
#
# Allocates N interactive GPU nodes (4 GPUs each), creates ONE shared run folder,
# trains + evaluates all 48 targets in parallel (one single-GPU worker per GPU), then
# aggregates the machine x SNR AUC/pAUC table. Reproduces the baseline at 50 epochs.
#
# Usage:
#   bash audio_ssl/scripts/run_baseline_salloc.sh [NODES] [WALLTIME]
#   bash audio_ssl/scripts/run_baseline_salloc.sh            # 4 nodes (16 GPUs), 4h
#   bash audio_ssl/scripts/run_baseline_salloc.sh 2 04:00:00 # 2 nodes (8 GPUs)
set -euo pipefail

export REPO=/pscratch/sd/d/dfarough/ASD-with-SSL
export CONFIG="${CONFIG:-audio_ssl/configs/autoencoder_baseline.yaml}"
export NODES="${1:-4}"
export GPUS_PER_NODE=4
export NTASKS=$(( NODES * GPUS_PER_NODE ))
WALLTIME="${2:-04:00:00}"
ACCOUNT="${NERSC_ACCOUNT:-m4539}"
PY=/global/homes/d/dfarough/.conda/envs/asd-ssl/bin/python

cd "$REPO"

# Shared run folder (created once) + deterministic target list (48 = 3 SNR x 4 type x 4 id).
export RUNDIR="$("$PY" -c 'from audio_ssl.src.utils.runs import create_run_dir; print(create_run_dir("audio_ssl/outputs/autoencoder_baseline"))')"
export TARGETS_FILE="$RUNDIR/targets.txt"
find dataset -mindepth 3 -maxdepth 3 -type d -name 'id_*' | sort > "$TARGETS_FILE"
NTARGETS=$(wc -l < "$TARGETS_FILE")

echo "RUN DIR : $RUNDIR"
echo "TARGETS : $NTARGETS across $NTASKS GPUs ($NODES nodes x $GPUS_PER_NODE)"
echo "WALLTIME: $WALLTIME   ACCOUNT: $ACCOUNT   CONFIG: $CONFIG"
if [ "$NTARGETS" -eq 0 ]; then
  echo "ERROR: no targets found under dataset/ (expected 48)"; exit 1
fi

salloc -J asd-ae -N "$NODES" -C gpu --gpus-per-node="$GPUS_PER_NODE" -q interactive -t "$WALLTIME" -A "$ACCOUNT" \
  bash "$REPO/audio_ssl/scripts/run_baseline_inside.sh"
