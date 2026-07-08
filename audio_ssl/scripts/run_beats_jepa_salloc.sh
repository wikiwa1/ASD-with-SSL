#!/usr/bin/env bash
# One-shot BEATs-JEPA: allocate N interactive GPU nodes, run continued SSL on normal
# MIMII from the pretrained BEATs checkpoint (DDP), then embedding-distance eval +
# machine x SNR table. Set INIT_ONLY=1 to skip training and evaluate the FROZEN
# pretrained BEATs (the zero-training baseline).
#
# Usage:
#   bash audio_ssl/scripts/run_beats_jepa_salloc.sh [NODES] [WALLTIME]
#   CONFIG=audio_ssl/configs/beats_fan.yaml bash audio_ssl/scripts/run_beats_jepa_salloc.sh
#   INIT_ONLY=1 bash audio_ssl/scripts/run_beats_jepa_salloc.sh 1 01:00:00
set -euo pipefail

export REPO=/pscratch/sd/d/dfarough/ASD-with-SSL
export CONFIG="${CONFIG:-audio_ssl/configs/beats_jepa.yaml}"
export NODES="${1:-1}"
export GPUS_PER_NODE=4
export NTASKS=$(( NODES * GPUS_PER_NODE ))
export INIT_ONLY="${INIT_ONLY:-}"
export CKPT="${CKPT:-}"       # optional: explicit checkpoint for the eval step
export TAG="${TAG:-}"         # optional: eval output suffix
export METHOD="${METHOD:-}"   # optional: override embedding.method
export PCADIM="${PCADIM:-}"   # optional: override embedding.pca_dim
WALLTIME="${2:-04:00:00}"
ACCOUNT="${NERSC_ACCOUNT:-m4539}"
PY=/global/homes/d/dfarough/.conda/envs/asd-ssl/bin/python

cd "$REPO"
python -m audio_ssl.scripts.download_beats >/dev/null 2>&1 || true  # no-op if present
export RUNDIR="$("$PY" -c 'import sys; from audio_ssl.src.utils.config import load_config; from audio_ssl.src.utils.runs import create_run_dir; print(create_run_dir(load_config(sys.argv[1])["output"]["directory"]))' "$CONFIG")"

echo "RUN DIR : $RUNDIR"
if [ -n "${INIT_ONLY}" ]; then
  echo "MODE    : frozen-probe (no training)"
else
  echo "MODE    : continued SSL, DDP over $NTASKS GPUs"
fi
echo "WALLTIME: $WALLTIME   ACCOUNT: $ACCOUNT   CONFIG: $CONFIG"

salloc -N "$NODES" -C gpu --gpus-per-node="$GPUS_PER_NODE" -q interactive -t "$WALLTIME" -A "$ACCOUNT" \
  bash "$REPO/audio_ssl/scripts/run_beats_jepa_inside.sh"
