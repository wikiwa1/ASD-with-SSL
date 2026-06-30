#!/usr/bin/env bash
# Inside the salloc allocation: embedding-distance eval of a trained JEPA run, then
# aggregate its table. Requires env: REPO, CONFIG, RUNDIR.
set -uo pipefail

echo "=== JEPA embedding-distance eval on $RUNDIR (single GPU) ==="
srun -N1 --ntasks=1 --gpus-per-task=1 --cpus-per-task=16 --export=ALL \
     bash -lc 'set -e; module load conda; conda activate asd-ssl; cd "$REPO";
               python -m audio_ssl.scripts.eval_jepa_embedding --config "$CONFIG" --run-dir "$RUNDIR"'

echo "=== aggregate embedding table (summary_embedding.yaml) ==="
module load conda
conda activate asd-ssl
cd "$REPO"
python -m audio_ssl.scripts.aggregate_results --config "$CONFIG" --run-dir "$RUNDIR" --result-file result_embedding.yaml
echo "=== done. run folder: $RUNDIR ==="
