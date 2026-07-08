from __future__ import annotations

import argparse
import multiprocessing as mp
import os
from functools import partial
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from sklearn.manifold import TSNE  # noqa: E402

from audio_ssl.src.data.splits import find_target_dirs, make_baseline_split, parse_target_info
from audio_ssl.src.evaluation.jepa_embeddings import embed_spectrograms
from audio_ssl.src.features.logmel import file_to_vector_array
from audio_ssl.src.features.spectrogram import stack_logmels
from audio_ssl.src.lightning.autoencoder_module import LitAutoEncoder
from audio_ssl.src.lightning.jepa_module import LitJEPA
from audio_ssl.src.utils.config import load_config

BASE = "./dataset"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Side-by-side t-SNE of test-clip embeddings (AE bottleneck vs JEPA encoder). "
                    "Use --target-dir for one target or --machine to pool all its ids/SNRs."
    )
    parser.add_argument("--ae-run", required=True)
    parser.add_argument("--jepa-run", required=True)
    parser.add_argument("--target-dir", help="A single MIMII target dir.")
    parser.add_argument("--machine", help="Pool every target of this machine (fan/pump/slider/valve).")
    parser.add_argument("--max-clips", type=int, default=2400, help="Total clips after stratified subsampling.")
    parser.add_argument("--ae-config", default="audio_ssl/configs/autoencoder_baseline.yaml")
    parser.add_argument("--jepa-config", default="audio_ssl/configs/jepa_baseline.yaml")
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def gather_targets(machine: str | None, target_dir: str | None) -> list[Path]:
    if target_dir:
        return [Path(target_dir)]
    return [d for d in find_target_dirs(BASE) if parse_target_info(d, BASE).machine_type == machine]


def stratified_subsample(files: list, labels: np.ndarray, k: int, seed: int = 42):
    """Up to k clips, balanced normal/abnormal, deterministic."""
    rng = np.random.default_rng(seed)
    keep = []
    for value in (0, 1):
        idx = np.where(labels == value)[0]
        take = min(k // 2, len(idx))
        keep.append(rng.choice(idx, take, replace=False))
    sel = np.sort(np.concatenate(keep))
    return [files[i] for i in sel], labels[sel]


def _vectors_parallel(files: list, feat: dict) -> list[np.ndarray]:
    n_jobs = int(os.environ.get("SLURM_CPUS_PER_TASK") or min(16, os.cpu_count() or 8))
    extract = partial(file_to_vector_array, **feat)
    if len(files) < 32 or n_jobs <= 1:
        return [extract(f) for f in files]
    with mp.Pool(n_jobs) as pool:
        return list(pool.imap(extract, files, chunksize=8))


def ae_embeddings(pooled: dict, ae_run: str, feat: dict, device: torch.device) -> np.ndarray:
    """Per-clip AE embedding = mean over frames of the bottleneck activation, using each
    clip's OWN per-target AE (the AE has no shared representation across targets)."""
    out = []
    for key, (files, _labels) in pooled.items():
        model = LitAutoEncoder.load_from_checkpoint(
            str(Path(ae_run) / "checkpoints" / key / "last.ckpt"), map_location=device).eval().to(device)
        hidden = list(model.hparams.hidden_dims)
        encoder = torch.nn.Sequential(
            *list(model.model.net.children())[: 2 * int(np.argmin(hidden)) + 1]).to(device).eval()
        with torch.inference_mode():
            for vectors in _vectors_parallel(files, feat):
                if len(vectors) == 0:
                    out.append(np.zeros(int(min(hidden)), dtype=np.float32))
                else:
                    out.append(encoder(torch.from_numpy(vectors).float().to(device)).mean(dim=0).cpu().numpy())
    return np.stack(out)


def jepa_embeddings(all_files: list, jepa_run: str, feat: dict, device: torch.device) -> np.ndarray:
    module = LitJEPA.load_from_checkpoint(
        str(Path(jepa_run) / "checkpoints" / "jepa_global" / "last.ckpt"), map_location=device)
    specs = stack_logmels(all_files, msg="jepa spectrograms", **feat)
    return embed_spectrograms(module, torch.from_numpy(specs).unsqueeze(1), device=device, encoder="target")


def scatter(ax, xy: np.ndarray, labels: np.ndarray, title: str) -> None:
    for value, color, name in [(0, "#2b6cb0", "normal"), (1, "#e53e3e", "abnormal")]:
        mask = labels == value
        ax.scatter(xy[mask, 0], xy[mask, 1], s=10, c=color, alpha=0.6, edgecolors="none", label=name)
    ax.set_title(title, fontsize=12)
    ax.set_xticks([]); ax.set_yticks([])
    ax.legend(loc="best", fontsize=9)


def tsne(x: np.ndarray) -> np.ndarray:
    perplexity = min(40, max(5, (len(x) - 1) // 4))
    return TSNE(n_components=2, init="pca", perplexity=perplexity, random_state=42).fit_transform(x)


def main() -> None:
    args = parse_args()
    if bool(args.target_dir) == bool(args.machine):
        raise SystemExit("Provide exactly one of --target-dir or --machine")
    device = torch.device(args.device) if args.device else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")

    target_dirs = gather_targets(args.machine, args.target_dir)
    k_per = max(2, args.max_clips // len(target_dirs))
    pooled: dict[str, tuple[list, np.ndarray]] = {}
    all_files: list = []
    all_labels: list = []
    for target_dir in target_dirs:
        info = parse_target_info(target_dir, base_directory=BASE)
        split = make_baseline_split(target_dir)
        files, labels = stratified_subsample(split.eval_files, split.eval_labels, k_per)
        pooled[info.key] = (files, labels)
        all_files += files
        all_labels += list(labels)
    all_labels = np.array(all_labels)
    scope = args.machine or parse_target_info(target_dirs[0], BASE).key
    print(f"{scope}: {len(target_dirs)} target(s), {len(all_files)} pooled clips "
          f"({int((all_labels == 0).sum())} normal / {int((all_labels == 1).sum())} abnormal)", flush=True)

    ae_feat = {**load_config(args.ae_config)["feature"], "channel": 0}
    jepa_feat = {**load_config(args.jepa_config)["feature"], "channel": 0}
    ae_emb = ae_embeddings(pooled, args.ae_run, ae_feat, device)
    jepa_emb = jepa_embeddings(all_files, args.jepa_run, jepa_feat, device)
    print(f"AE emb {ae_emb.shape} | JEPA emb {jepa_emb.shape}", flush=True)

    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    scatter(axes[0], tsne(ae_emb), all_labels, f"AE bottleneck ({ae_emb.shape[1]}-d, per-target models)")
    scatter(axes[1], tsne(jepa_emb), all_labels, f"JEPA encoder ({jepa_emb.shape[1]}-d, one global model)")
    fig.suptitle(f"t-SNE of test-clip embeddings — {scope} (all ids/SNRs)", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(args.output)
    if args.output.endswith(".pdf"):
        fig.savefig(args.output[:-4] + ".png", dpi=90)
    print(f"wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
