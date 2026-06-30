from __future__ import annotations

try:  # import before torch so Comet can auto-instrument
    import comet_ml  # noqa: F401
except ImportError:
    comet_ml = None

import argparse
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

from audio_ssl.src.data.splits import discover_targets, parse_target_info
from audio_ssl.src.evaluation.eval_artifacts import compute_and_save_roc, log_results_to_comet
from audio_ssl.src.utils.config import load_config, merge_cli_overrides
from audio_ssl.src.utils.io import ensure_dir, write_yaml
from audio_ssl.src.utils.loggers import load_env
from audio_ssl.src.utils.runs import resolve_run_dir

RUN_KEY = "jepa_global"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ensemble per-target anomaly scorers by rank- or z-averaging their saved scores."
    )
    parser.add_argument("--config", default="audio_ssl/configs/jepa_baseline.yaml")
    parser.add_argument("--base-directory", help="Override data.base_directory.")
    parser.add_argument("--run-dir", help="Run folder to read/write (default: the latest run).")
    parser.add_argument(
        "--score-dir", action="append", default=None,
        help="ROC subdir(s) holding <key>.npz with scores+labels (default: roc roc_embedding).",
    )
    parser.add_argument("--combine", choices=["rank", "zscore"], default="rank")
    parser.add_argument("--result-name", default="result_ensemble.yaml")
    parser.add_argument("--roc-subdir", default="roc_ensemble")
    return parser.parse_args()


def combine(score_arrays: list[np.ndarray], method: str) -> np.ndarray:
    """Fuse per-method score vectors (all 'higher = more anomalous') into one. Rank- and
    z-averaging are scale-free, so a strong and a weak scorer combine sensibly."""
    if method == "rank":
        return np.mean([rankdata(s) for s in score_arrays], axis=0)
    stacked = [(s - s.mean()) / (s.std() + 1e-9) for s in score_arrays]
    return np.mean(stacked, axis=0)


def main() -> None:
    args = parse_args()
    config = merge_cli_overrides(load_config(args.config), args)
    load_env()
    base = config["data"]["base_directory"]
    score_dirs = args.score_dir or ["roc", "roc_embedding"]
    output_dir = ensure_dir(resolve_run_dir(config["output"]["directory"], args.run_dir))
    roc_dir = ensure_dir(output_dir / args.roc_subdir)
    max_fpr = float(config.get("evaluation", {}).get("pauc_max_fpr", 0.1))
    print(f"RUN DIR: {output_dir}  | ensemble({args.combine}) of {score_dirs}", flush=True)

    info_by_key = {parse_target_info(td, base).key: parse_target_info(td, base)
                   for td in discover_targets(config)}

    results = {}
    for key, info in info_by_key.items():
        paths = [output_dir / d / f"{key}.npz" for d in score_dirs]
        if not all(p.exists() for p in paths):
            print(f"  (skip {key}: missing {[str(p) for p in paths if not p.exists()]})", flush=True)
            continue
        loaded = [np.load(p) for p in paths]
        labels = loaded[0]["labels"]
        if not all(np.array_equal(z["labels"], labels) for z in loaded):
            raise RuntimeError(f"label mismatch across score dirs for {key}")
        scores = combine([z["scores"] for z in loaded], args.combine)

        results[key] = compute_and_save_roc(
            roc_dir, info, scores, labels, f"ensemble({args.combine}):{'+'.join(score_dirs)}",
            list(range(len(labels))), max_fpr)
        print(f"{key}: AUC={results[key]['AUC']:.6f}  pAUC={results[key]['pAUC']:.6f}", flush=True)

    result_path = output_dir / args.result_name
    write_yaml(result_path, results)
    print(f"wrote {result_path}", flush=True)
    log_results_to_comet(config, output_dir / "checkpoints" / RUN_KEY / "comet_experiment.txt",
                         results, prefix="test_ens")


if __name__ == "__main__":
    main()
