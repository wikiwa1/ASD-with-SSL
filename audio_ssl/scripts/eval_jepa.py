from __future__ import annotations

try:  # import before torch so Comet can auto-instrument
    import comet_ml  # noqa: F401
except ImportError:
    comet_ml = None

import argparse
from pathlib import Path

import torch

from audio_ssl.src.data.splits import discover_targets, make_baseline_split, parse_target_info
from audio_ssl.src.evaluation.eval_artifacts import compute_and_save_roc, log_results_to_comet
from audio_ssl.src.evaluation.jepa_scores import score_spectrograms
from audio_ssl.src.features.spectrogram import stack_logmels
from audio_ssl.src.lightning.jepa_module import LitJEPA
from audio_ssl.src.utils.config import load_config, merge_cli_overrides
from audio_ssl.src.utils.io import ensure_dir, write_yaml
from audio_ssl.src.utils.loggers import load_env
from audio_ssl.src.utils.runs import resolve_run_dir

RUN_KEY = "jepa_global"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the global JEPA via latent prediction error.")
    parser.add_argument("--config", default="audio_ssl/configs/jepa_baseline.yaml")
    parser.add_argument("--target-dir", help="Evaluate only one MIMII target directory.")
    parser.add_argument("--base-directory", help="Override data.base_directory.")
    parser.add_argument("--run-dir", help="Run folder to evaluate (default: the latest run).")
    parser.add_argument("--checkpoint", help="Explicit JEPA checkpoint (default: last.ckpt in the run).")
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
    config = merge_cli_overrides(load_config(args.config), args)
    load_env()
    data_cfg = config["data"]
    feature_cfg = {**config["feature"], "channel": data_cfg.get("channel", 0)}
    output_dir = ensure_dir(resolve_run_dir(config["output"]["directory"], args.run_dir))
    checkpoint_root = output_dir / "checkpoints"
    roc_dir = ensure_dir(output_dir / "roc")
    max_fpr = float(config.get("evaluation", {}).get("pauc_max_fpr", 0.1))
    num_masks = int(config.get("scoring", {}).get("num_masks", 16))
    batch_size = int(config["fit"].get("batch_size", 256))
    device = torch.device(args.device) if args.device else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")
    print(f"RUN DIR: {output_dir}", flush=True)

    checkpoint_path = Path(args.checkpoint) if args.checkpoint else find_checkpoint(checkpoint_root)
    module = LitJEPA.load_from_checkpoint(str(checkpoint_path), map_location=device)

    if args.target_dir:
        target_dirs = [Path(args.target_dir)]
    else:
        target_dirs = discover_targets(config)

    results = {}
    for target_dir in target_dirs:
        info = parse_target_info(target_dir, base_directory=data_cfg["base_directory"])
        split = make_baseline_split(
            target_dir,
            normal_dir_name=data_cfg.get("normal_dir_name", "normal"),
            abnormal_dir_name=data_cfg.get("abnormal_dir_name", "abnormal"),
            ext=data_cfg.get("ext", "wav"),
        )
        specs = stack_logmels(split.eval_files, msg=f"spectrograms {info.key}", **feature_cfg)
        specs = torch.from_numpy(specs).unsqueeze(1)  # (N, 1, M, T)
        scores = score_spectrograms(module, specs, num_masks=num_masks, batch_size=batch_size, device=device)

        results[info.key] = compute_and_save_roc(
            roc_dir, info, scores, split.eval_labels, checkpoint_path, split.eval_files, max_fpr
        )
        print(f"{info.key}: AUC={results[info.key]['AUC']:.6f}  pAUC={results[info.key]['pAUC']:.6f}", flush=True)

    result_path = output_dir / config["output"].get("result_file", "result.yaml")
    write_yaml(result_path, results)
    print(f"wrote {result_path}", flush=True)

    # Log per-target test metrics into the single global JEPA experiment.
    log_results_to_comet(config, checkpoint_root / RUN_KEY / "comet_experiment.txt", results, prefix="test")


if __name__ == "__main__":
    main()
