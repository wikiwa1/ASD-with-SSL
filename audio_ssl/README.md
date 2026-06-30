# Audio SSL Experiments

Shared infrastructure for MIMII audio anomaly detection experiments.

The first implemented method is the MIMII dense autoencoder baseline, rewritten
with PyTorch Lightning. The package layout is intentionally method-agnostic so a
future JEPA/SSL method can reuse the same data loading, feature extraction,
checkpointing, and anomaly evaluation code.

## Environment

```bash
module load conda
conda activate asd-ssl
```

For Perlmutter GPU training, the environment needs CUDA-enabled PyTorch wheels:

```bash
python -m pip install -r requirements-gpu-cu128.txt
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

## Autoencoder Baseline

Train all detected MIMII target directories:

```bash
python -m audio_ssl.scripts.train_autoencoder \
  --config audio_ssl/configs/autoencoder_baseline.yaml
```

Evaluate all trained checkpoints (writes AUC + pAUC and a ROC curve png+npz per
target into the run folder's `roc/`):

```bash
python -m audio_ssl.scripts.eval_autoencoder \
  --config audio_ssl/configs/autoencoder_baseline.yaml
```

### Run folders

Each training invocation creates a uniquely named run folder under `outputs/`, e.g.
`autoencoder_baseline_<date>_<bigram>_<nnnn>`, so reruns never overwrite each other.
A `autoencoder_baseline_latest` symlink points at the newest run; `eval` defaults to
it. Pin a specific run with `--run-dir <path>` on either script. The logmel feature
cache is shared across runs (`autoencoder_baseline_feature_cache/`) so features are
not recomputed each run. Quick interactive-node one-shots:

```bash
bash audio_ssl/scripts/run_training_salloc.sh                 # salloc -> train (smoke)
bash audio_ssl/scripts/run_eval_salloc.sh                     # salloc -> eval latest run
```

### Full baseline sweep (all 48 targets, multi-GPU)

The MIMII baseline trains one autoencoder per target (3 SNR x 4 machine types x 4
model IDs = 48), independently, at 50 epochs. `run_baseline_salloc.sh` does the whole
sweep in one shot: it allocates N interactive GPU nodes (4 GPUs each), creates one
shared run folder, then runs one single-GPU worker per GPU — each training + evaluating
its round-robin slice of the 48 targets (data-parallel across targets, not DDP). When
all workers finish it aggregates the machine x SNR AUC/pAUC table.

```bash
bash audio_ssl/scripts/run_baseline_salloc.sh            # 4 nodes (16 GPUs), 4h
bash audio_ssl/scripts/run_baseline_salloc.sh 2 04:00:00 # 2 nodes (8 GPUs)
```

Outputs in the run folder: per-target `roc/*.png|npz`, merged `result.yaml`, and
`summary.yaml` (the paper-style table). Each target is a linked Comet experiment
(train curves + test AUC/pAUC + ROC). Re-aggregate any run without recomputing:

```bash
python -m audio_ssl.scripts.aggregate_results --run-dir audio_ssl/outputs/autoencoder_baseline_<...>
```

For Perlmutter multi-node interactive jobs, set the trainer section in the
config, or override from the CLI:

```bash
srun -N 4 --ntasks-per-node=4 --gpus-per-node=4 --gpu-bind=none \
  python -m audio_ssl.scripts.train_autoencoder \
  --config audio_ssl/configs/autoencoder_baseline.yaml \
  --accelerator gpu \
  --devices 4 \
  --num-nodes 4 \
  --strategy ddp
```

Each task must see all GPUs on its node (Lightning assigns one per local rank), so use
`--gpus-per-node` + `--gpu-bind=none`, not `--gpus-per-task=1` (which would make Lightning
see only one GPU and fail).

For Lightning on SLURM, the number of tasks per node should match `devices`.
The shared code does not assume one method: autoencoder and future JEPA modules
will use the same trainer configuration path.

## JEPA (latent prediction error)

A self-supervised alternative to the AE: train **one global audio-JEPA** (I-JEPA style)
on all normal MIMII clips, then score anomalies by **latent prediction error** — how badly
the predictor reconstructs masked time-frequency block embeddings (no per-target fitting).
Normal machine sound is periodic/predictable; anomalies break predictability -> higher error.

- Input: full-clip log-mel (64 x 313), patch 16x16, small ViT context encoder + EMA target
  encoder + narrow predictor, smooth-L1 latent loss, I-JEPA block masking (`src/models/jepa/`,
  `src/lightning/jepa_module.py`, config `configs/jepa_baseline.yaml`).
- Score: mean prediction error over N fixed-seed masks per clip (`src/evaluation/jepa_scores.py`).
- Eval reuses the same per-target AUC/pAUC + ROC + `aggregate_results`, so its machine x SNR
  table is directly comparable to the AE baseline.

One-shot (allocate GPU nodes -> DDP pretrain -> eval -> aggregate):

```bash
bash audio_ssl/scripts/run_jepa_salloc.sh            # 1 node (4 GPUs), DDP, 4h
bash audio_ssl/scripts/run_jepa_salloc.sh 4 04:00:00 # 4 nodes (16 GPUs)
```

Pretraining is one model = **real DDP** (unlike the AE sweep). The ~50k spectrograms are
extracted in parallel and cached once (`jepa_baseline_feature_cache/`). All per-target test
metrics + ROC log into the single global Comet experiment.

### Second scorer: embedding distance (no retraining)

The same frozen encoder can be scored a second way — one-class distance on its mean-pooled
clip embeddings. For each target, fit Mahalanobis (Ledoit-Wolf) or k-NN (cosine) on the
normal-train embeddings, then score the test split by distance. Configure under
`embedding:` in the config; outputs land beside the prediction-error ones
(`result_embedding.yaml`, `roc_embedding/`, Comet prefix `test_emb`) so they're comparable,
not overwritten. Reuses the existing checkpoint — point it at a trained run:

```bash
bash audio_ssl/scripts/run_jepa_embedding_eval_salloc.sh                 # latest JEPA run
bash audio_ssl/scripts/run_jepa_embedding_eval_salloc.sh <run-folder>    # a specific run
# table: python -m audio_ssl.scripts.aggregate_results --run-dir <run> --result-file result_embedding.yaml
```

## Experiment tracking (Comet-ML)

Training logs to Comet alongside the local CSV logger. Credentials are read from a
repo-root `.env` (gitignored), loaded automatically at the start of training:

```
COMET_API_KEY="..."
COMET_WORKSPACE="dfaroughy"
COMET_PROJECT_NAME="Audio_ssl"
```

Each MIMII target trains as its own Comet experiment named by its key
(e.g. `fan_id_00_-6_dB`), tagged per the config, with machine/SNR/feature/fit
hyperparameters logged. Toggle and configure under `logging.comet` in the config.

The training experiment key is saved next to the checkpoint, so `eval` reattaches the
test metrics (`test_AUC`, `test_pAUC`) and the ROC curve + plot to the **same**
experiment — training curves and test results live together. (Eval logging requires
`online: true`; it is skipped for offline runs.)

Comet's git/conda/system/CO2 auto-collection is disabled by default: it scans the
filesystem and makes network calls at experiment creation, which is slow and
hang-prone on the NERSC home GPFS. Re-enable any of it via `logging.comet.options`.

Compute nodes without outbound internet should run offline and upload afterwards:

```bash
# config: logging.comet.online -> false   (writes <output.directory>/comet_offline/*.zip)
comet upload audio_ssl/outputs/autoencoder_baseline/comet_offline/*.zip
```

The logger wiring lives in `audio_ssl/src/utils/loggers.py` and is method-agnostic,
so the future JEPA/SSL training script can call `build_loggers(...)` the same way.
