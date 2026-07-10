# Self-Supervised Sound Anomaly Detection (MIMII)

Detecting anomalous machine sounds **without labelled anomalies**, using
self-supervised learning. Anomalous examples are rare and unpredictable in the
real world, so we train only on *normal* sound and flag clips that the model
finds surprising.

This repository is a comparative study of four approaches on the
[MIMII dataset](https://zenodo.org/record/3384388) (fan machine), from a simple
supervised baseline up to two modern SSL methods (an adaptation of **DINO** and
an adaptation of **JEPA**).

> **Erdős Institute Deep Learning Bootcamp project.** The primary deliverables
> are the annotated notebooks in [`notebooks/`](notebooks/).

---

## The dataset

[MIMII](https://zenodo.org/record/3384388) — *Malfunctioning Industrial Machine
Investigation and Inspection* — is a collection of real machine-operating sounds
(fan, pump, slider, valve) recorded with background factory noise mixed in at
three signal-to-noise ratios (−6 dB, 0 dB, +6 dB). Each machine type has several
individual units (`id_00`, `id_02`, …), and each unit has `normal/` and
`abnormal/` clips. We train on `normal` only and use `abnormal` purely for
evaluation.

The raw audio is **not** committed (it is large and gitignored). See
[Getting the data](#getting-the-data) below.

## Methods

Each method learns "what normal sounds like" differently, then turns that into a
per-clip **anomaly score** evaluated with ROC **AUC / pAUC**.

| # | Method | Idea | Code | Notebook |
|---|--------|------|------|----------|
| 1 | **Log-mel + ResNet-18 baseline** | Supervised reference point: classify normal vs. abnormal from log-mel spectrograms. | notebooks | [`03`](notebooks/03_baseline_logmel_avg.ipynb), [`04`](notebooks/04_resnet18_classifier.ipynb) |
| 2 | **Dense autoencoder** | Reconstruct normal log-mel frames; high reconstruction error ⇒ anomaly. The classic MIMII baseline, in PyTorch Lightning. | [`audio_ssl/`](audio_ssl/) | (scripts) |
| 3 | **DINO** | Self-distillation: a student matches a teacher's embeddings of augmented crops; anomalies fall far from the normal-sound manifold. | [`dino_asd/`](dino_asd/) | [`05`](notebooks/05_dino.ipynb) |
| 4 | **JEPA / LeJEPA** | Predict masked time–frequency block *embeddings* (I-JEPA style); latent prediction error scores anomalies. Also a BEATs-backbone variant. | [`audio_ssl/`](audio_ssl/) | (scripts) |

### Results

| model                     | id 00 | id 02 | id 04 | id 06 | mean  |
|---------------------------|-------|-------|-------|-------|-------|
| JEPA-ViT (all machines)   | **0.761** | **0.965** | 0.901 | 0.983 | **0.903** |
| JEPA-ViT (fans only)      | 0.719 | 0.958 | **0.903** | **0.987** | 0.892 |
| DINO-BEATs                | 0.757 | 0.895 | 0.878 | 0.967 | 0.874 |
| DINO-ResNet               | 0.681 | 0.911 | 0.818 | 0.904 | 0.829 |

*fan ROC-AUC, averaged over −6/0/+6 dB; best per column in bold*

## Repository layout

```
.
├── README.md                 ← you are here
├── notebooks/                ← annotated deliverables, in reading order
│   ├── 01_eda.ipynb                    exploratory data analysis
│   ├── 02_features_stft_logmel.ipynb   STFT / log-mel feature building
│   ├── 03_baseline_logmel_avg.ipynb    log-mel-average baseline
│   ├── 04_resnet18_classifier.ipynb    supervised ResNet-18 reference
│   └── 05_dino.ipynb                   DINO SSL method
│
├── audio_ssl/                ← Lightning package: autoencoder + JEPA + LeJEPA + BEATs-JEPA
│   ├── src/                    data / features / models / lightning / evaluation / utils
│   ├── scripts/                train / eval / plot + SLURM (Perlmutter) launchers
│   ├── configs/                per-experiment YAML
│   └── README.md               developer & HPC guide for this package
│
├── dino_asd/                 ← DINO method (Colab-oriented package, used by notebook 05)
├── docs/                     ← background notes (JEPA literature review)
├── pyproject.toml            ← packages dino_asd (see note inside)
├── requirements.txt          ← full environment (notebooks + audio_ssl)
└── requirements-gpu-cu128.txt ← CUDA-enabled torch wheels for GPU nodes
```

### Why two Python packages?

The two SSL code bases grew for different environments and are intentionally kept
separate:

- **`audio_ssl/`** — the autoencoder and JEPA family. Built for multi-GPU HPC
  (SLURM/Perlmutter) with PyTorch Lightning, YAML configs, run folders, and
  Comet logging. Its own [README](audio_ssl/README.md) is the developer guide.
- **`dino_asd/`** — the DINO method. A small, flat package meant to be `pip`-
  installed and driven from the Colab notebook (`notebooks/05_dino.ipynb`).

They share the same dataset and evaluation protocol (AUC/pAUC) so results are
comparable, even though the code is not unified.

## Getting started

### Environment

```bash
python -m venv .venv && source .venv/bin/activate    # or: source .venv/bin/activate.fish
pip install -r requirements.txt                      # notebooks + audio_ssl
# On a CUDA GPU node, add the matching torch wheels:
pip install -r requirements-gpu-cu128.txt
```

The DINO package can also be installed on its own (e.g. in Colab):

```bash
pip install -e .        # installs the dino_asd package (see pyproject.toml)
```

### Getting the data

MIMII fan audio is not committed. Download the SNR slice(s) you want (e.g.
`-6_dB_fan.zip`) from the [MIMII Zenodo record](https://zenodo.org/record/3384388)
and unzip so the tree looks like:

```
fan/id_00/{normal,abnormal}/*.wav
fan/id_02/...
```

`audio_ssl` includes a helper: `python -m audio_ssl.scripts.download_mimii`.
The BEATs checkpoint (for the BEATs-JEPA variant) is fetched with
`python -m audio_ssl.scripts.download_beats`.

> **Notebook convention:** launch Jupyter from the **repository root** so relative
> paths (`fan/`, `dino_asd` imports) resolve. Colab notebooks expect the data on
> Google Drive — adjust the `DATA_ROOT` at the top of the notebook to match.

### Running things

- **Notebooks** — open the files in [`notebooks/`](notebooks/) in order.
- **Autoencoder / JEPA** — see the [`audio_ssl` README](audio_ssl/README.md) for
  training, evaluation, sweeps, and experiment tracking.

## Suggested reading order

1. `notebooks/01_eda.ipynb` — what the data looks like.
2. `notebooks/02_features_stft_logmel.ipynb` — how we turn audio into features.
3. `notebooks/03` & `04` — supervised baselines to beat.
4. `notebooks/05_dino.ipynb` — the DINO SSL method.
5. [`audio_ssl` README](audio_ssl/README.md) — the autoencoder & JEPA experiments.
6. [`docs/JEPA_literature_review.md`](docs/JEPA_literature_review.md) — background.
