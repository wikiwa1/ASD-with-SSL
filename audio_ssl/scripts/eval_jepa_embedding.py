from __future__ import annotations

try:  # import before torch so Comet can auto-instrument
    import comet_ml  # noqa: F401
except ImportError:
    comet_ml = None

import argparse
from pathlib import Path

import torch

from audio_ssl.src.data.splits import discover_targets, make_baseline_split, parse_target_info
from audio_ssl.src.evaluation.embedding_scores import build_scorer
from audio_ssl.src.evaluation.eval_artifacts import compute_and_save_roc, log_results_to_comet
from audio_ssl.src.evaluation.jepa_embeddings import embed_spectrograms, fit_set_embeddings
from audio_ssl.src.features.frontends import stack_features
from audio_ssl.src.lightning.loader import load_ssl_module
from audio_ssl.src.utils.config import load_config, merge_cli_overrides
from audio_ssl.src.utils.io import ensure_dir, write_yaml
from audio_ssl.src.utils.loggers import load_env
from audio_ssl.src.utils.runs import feature_cache_root, resolve_run_dir

RUN_KEY = "jepa_global"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the global JEPA encoder by embedding-distance one-class scoring."
    )
    parser.add_argument("--config", default="audio_ssl/configs/jepa_baseline.yaml")
    parser.add_argument("--target-dir", help="Evaluate only one MIMII target directory.")
    parser.add_argument("--base-directory", help="Override data.base_directory.")
    parser.add_argument("--run-dir", help="Run folder to evaluate (default: the latest run).")
    parser.add_argument("--checkpoint", help="Explicit JEPA checkpoint (default: last.ckpt in the run).")
    parser.add_argument("--tag", default="", help="Suffix for outputs so a specific-checkpoint eval "
                        "does not clobber the final one, e.g. --tag epoch220.")
    parser.add_argument("--method", help="Override embedding.method (mahalanobis|knn|gmm|flow).")
    parser.add_argument("--pca-dim", type=int, help="Override embedding.pca_dim.")
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def find_checkpoint(checkpoint_root: Path) -> Path:
    target = checkpoint_root / RUN_KEY
    last = target / "last.ckpt"
    if last.exists():
        return last
    candidates = sorted(target.glob("*.ckpt"))
    if not candidates:
        raise FileNotFoundError(f"No JEPA checkpoint in {target}")
    return candidates[-1]


def main() -> None:
    args = parse_args()
    config_path = args.config
    if args.run_dir and (Path(args.run_dir) / "config.yaml").exists():
        config_path = str(Path(args.run_dir) / "config.yaml")  # self-describing run -> correct machines/cache
        print(f"[eval] using run config: {config_path}", flush=True)
    config = merge_cli_overrides(load_config(config_path), args)
    load_env()
    data_cfg = config["data"]
    feature_cfg = {**config["feature"], "channel": data_cfg.get("channel", 0)}
    emb_cfg = dict(config.get("embedding", {}))
    if args.method:
        emb_cfg["method"] = args.method
    if args.pca_dim is not None:
        emb_cfg["pca_dim"] = args.pca_dim
    encoder = emb_cfg.get("encoder", "target")
    method = emb_cfg.get("method", "mahalanobis")

    tag = f"_{args.tag}" if args.tag else ""
    base_output = config["output"]["directory"]
    output_dir = ensure_dir(resolve_run_dir(base_output, args.run_dir))
    checkpoint_root = output_dir / "checkpoints"
    roc_dir = ensure_dir(output_dir / f"roc_embedding{tag}")
    max_fpr = float(config.get("evaluation", {}).get("pauc_max_fpr", 0.1))
    batch_size = int(config["fit"].get("batch_size", 256))
    device = torch.device(args.device) if args.device else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")
    cache_path = feature_cache_root(base_output) / f"jepa_specs_{feature_cfg['n_mels']}m_{feature_cfg['target_frames']}t.npy"
    print(f"RUN DIR: {output_dir}  | scorer={method} encoder={encoder}", flush=True)

    checkpoint_path = Path(args.checkpoint) if args.checkpoint else find_checkpoint(checkpoint_root)
    module = load_ssl_module(config, checkpoint_path, device)

    target_dirs = [Path(args.target_dir)] if args.target_dir else discover_targets(config)

    # One-class fit set: per-target normal-train embeddings (frozen encoder).
    normal_emb = fit_set_embeddings(module, config, target_dirs, cache_path, feature_cfg, batch_size, device, encoder)

    results = {}
    for target_dir in target_dirs:
        info = parse_target_info(target_dir, base_directory=data_cfg["base_directory"])
        split = make_baseline_split(
            target_dir,
            normal_dir_name=data_cfg.get("normal_dir_name", "normal"),
            abnormal_dir_name=data_cfg.get("abnormal_dir_name", "abnormal"),
            ext=data_cfg.get("ext", "wav"),
        )
        eval_specs = stack_features(split.eval_files, msg=f"eval spectrograms {info.key}", **feature_cfg)
        eval_emb = embed_spectrograms(
            module, torch.from_numpy(eval_specs).unsqueeze(1), batch_size, device, encoder
        )
        scorer = build_scorer(emb_cfg).fit(normal_emb[info.key])
        scores = scorer.score(eval_emb)

        results[info.key] = compute_and_save_roc(
            roc_dir, info, scores, split.eval_labels, checkpoint_path, split.eval_files, max_fpr
        )
        print(f"{info.key}: AUC={results[info.key]['AUC']:.6f}  pAUC={results[info.key]['pAUC']:.6f}", flush=True)

    result_path = output_dir / f"result_embedding{tag}.yaml"
    write_yaml(result_path, results)
    print(f"wrote {result_path}", flush=True)

    # Log into the same global JEPA experiment under a distinct prefix.
    log_results_to_comet(config, checkpoint_root / RUN_KEY / "comet_experiment.txt", results, prefix=f"test_emb{tag}")


if __name__ == "__main__":
    main()
