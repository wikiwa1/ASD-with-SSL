#!/usr/bin/env bash
# One-shot: allocate an interactive Perlmutter GPU node, activate the asd-ssl env on
# the compute node, and evaluate trained AE checkpoint(s) on the held-out MIMII test
# split — writing AUC + pAUC to result.yaml and a ROC curve (png + npz) per target.
#
# Usage:
#   bash audio_ssl/scripts/run_eval_salloc.sh [TARGET_DIR] [WALLTIME]
#
#   bash audio_ssl/scripts/run_eval_salloc.sh                         # the smoke target
#   bash audio_ssl/scripts/run_eval_salloc.sh dataset/0_dB/fan/id_00  # one target
#   bash audio_ssl/scripts/run_eval_salloc.sh ALL 00:30:00           # every trained target
set -euo pipefail

export REPO=/pscratch/sd/d/dfarough/ASD-with-SSL
export TARGET="${1:-dataset/0_dB/fan/id_00}"   # use "ALL" to evaluate every checkpoint
WALLTIME="${2:-00:20:00}"
ACCOUNT="${NERSC_ACCOUNT:-m4539}"

# "ALL" -> no --target-dir, so eval discovers every trained target under base_directory.
export TARGET_FLAG="--target-dir $TARGET"
[ "$TARGET" = "ALL" ] && export TARGET_FLAG=""

echo "Requesting interactive GPU node (account=$ACCOUNT, t=$WALLTIME)"
echo "  -> evaluating ${TARGET} (ROC + AUC + pAUC)"

salloc -N 1 -C gpu -G 1 -q interactive -t "$WALLTIME" -A "$ACCOUNT" \
  srun --ntasks=1 --export=ALL bash -lc '
    set -e
    module load conda
    conda activate asd-ssl
    cd "$REPO"
    echo "=== compute node: $(hostname) | python: $(which python) ==="
    python -m audio_ssl.scripts.eval_autoencoder \
      --config audio_ssl/configs/autoencoder_baseline.yaml \
      $TARGET_FLAG
    RUN=audio_ssl/outputs/autoencoder_baseline_latest
    echo "=== results ($RUN) ==="
    cat "$RUN/result.yaml"
    echo "=== ROC artifacts ==="
    ls -1 "$RUN/roc/"
  '
