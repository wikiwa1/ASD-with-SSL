from __future__ import annotations

try:  # import before torch so Comet can auto-instrument
    import comet_ml  # noqa: F401
except ImportError:
    comet_ml = None

import argparse
from pathlib import Path

import torch

from audio_ssl.src.data.splits import find_target_dirs, make_baseline_split, parse_target_info
from audio_ssl.src.evaluation.anomaly_scores import score_files
from audio_ssl.src.evaluation.auc import roc_curve_points
from audio_ssl.src.lightning.autoencoder_module import LitAutoEncoder
from audio_ssl.src.utils.config import load_config, merge_cli_overrides
from audio_ssl.src.utils.io import ensure_dir, save_npz, write_yaml
from audio_ssl.src.utils.loggers import end_experiments, load_env, resume_comet_experiment
from audio_ssl.src.utils.runs import resolve_run_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate trained autoencoder checkpoints.")
    parser.add_argument("--config", default="audio_ssl/configs/autoencoder_baseline.yaml")
    parser.add_argument("--target-dir", help="Evaluate only one MIMII target directory.")
    parser.add_argument("--base-directory", help="Override data.base_directory.")
    parser.add_argument("--run-dir", help="Run folder to evaluate (default: the latest run).")
    parser.add_argument("--checkpoint", help="Use one explicit checkpoint for --target-dir.")
    parser.add_argument(
        "--fragments",
        action="store_true",
        help="Write one results/<key>.yaml per target (for parallel eval) instead of a combined result.yaml.",
    )
    parser.add_argument("--device", default=None, help="cpu, cuda, or cuda:N. Default auto-selects.")
    return parser.parse_args()


def target_dirs_from_args(config: dict, args: argparse.Namespace) -> list[Path]:
    if args.target_dir:
        return [Path(args.target_dir)]
    return find_target_dirs(config["data"]["base_directory"])


def find_checkpoint(checkpoint_root: Path, key: str) -> Path:
    target = checkpoint_root / key
    last = target / "last.ckpt"
    if last.exists():
        return last
    candidates = sorted(target.glob("*.ckpt"))
    if not candidates:
        raise FileNotFoundError(f"No checkpoint found in {target}")
    return candidates[-1]


def main() -> None:
    args = parse_args()
    config = merge_cli_overrides(load_config(args.config), args)
    load_env()  # pull COMET_* credentials from .env if present
    data_cfg = config["data"]
    feature_cfg = {**config["feature"], "channel": data_cfg.get("channel", 0)}
    output_dir = ensure_dir(resolve_run_dir(config["output"]["directory"], args.run_dir))
    checkpoint_root = output_dir / "checkpoints"
    roc_dir = ensure_dir(output_dir / "roc")
    max_fpr = float(config.get("evaluation", {}).get("pauc_max_fpr", 0.1))
    print(f"RUN DIR: {output_dir}", flush=True)

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    results = {}
    for target_dir in target_dirs_from_args(config, args):
        info = parse_target_info(target_dir, base_directory=data_cfg["base_directory"])
        split = make_baseline_split(
            target_dir,
            normal_dir_name=data_cfg.get("normal_dir_name", "normal"),
            abnormal_dir_name=data_cfg.get("abnormal_dir_name", "abnormal"),
            ext=data_cfg.get("ext", "wav"),
        )
        checkpoint_path = Path(args.checkpoint) if args.checkpoint else find_checkpoint(checkpoint_root, info.key)
        module = LitAutoEncoder.load_from_checkpoint(str(checkpoint_path), map_location=device)
        scores = score_files(module.model, split.eval_files, feature_cfg, device=device)
        roc = roc_curve_points(split.eval_labels, scores, max_fpr=max_fpr)

        npz_path = roc_dir / f"{info.key}.npz"
        save_npz(
            npz_path,
            fpr=roc["fpr"],
            tpr=roc["tpr"],
            thresholds=roc["thresholds"],
            scores=scores,
            labels=split.eval_labels,
        )
        png_path = roc_dir / f"{info.key}.png"
        try:
            from audio_ssl.src.evaluation.plots import save_roc_plot

            save_roc_plot(
                png_path,
                roc["fpr"],
                roc["tpr"],
                roc["auc"],
                pauc=roc["pauc"],
                max_fpr=roc["max_fpr"],
                title=info.key,
            )
            plot_path = str(png_path)
        except Exception as exc:  # plotting is optional; never fail eval over a plot
            print(f"  (skipped ROC plot for {info.key}: {exc})", flush=True)
            plot_path = None

        results[info.key] = {
            "AUC": roc["auc"],
            "pAUC": roc["pauc"],
            "pauc_max_fpr": roc["max_fpr"],
            "machine_type": info.machine_type,
            "machine_id": info.machine_id,
            "db": info.db,
            "checkpoint": str(checkpoint_path),
            "num_eval_files": int(len(split.eval_files)),
            "roc_data": str(npz_path),
            "roc_plot": plot_path,
        }
        if args.fragments:  # one file per target so parallel workers don't clobber result.yaml
            frag_dir = ensure_dir(output_dir / "results")
            write_yaml(frag_dir / f"{info.key}.yaml", {info.key: results[info.key]})
        print(f"{info.key}: AUC={roc['auc']:.6f}  pAUC={roc['pauc']:.6f}")

        # Reattach test metrics + ROC to the same Comet experiment as the training run.
        key_file = checkpoint_root / info.key / "comet_experiment.txt"
        experiment_key = key_file.read_text().strip() if key_file.exists() else None
        comet_logger = resume_comet_experiment(config, experiment_key) if experiment_key else None
        if comet_logger is not None:
            try:
                comet_logger.log_metrics({"test_AUC": roc["auc"], "test_pAUC": roc["pauc"]})
                experiment = comet_logger.experiment
                experiment.log_curve(
                    f"ROC_{info.key}",
                    x=[float(v) for v in roc["fpr"]],
                    y=[float(v) for v in roc["tpr"]],
                )
                if plot_path:
                    experiment.log_image(plot_path, name=f"ROC_{info.key}")
            except Exception as exc:  # never fail eval over experiment-tracking
                print(f"  (comet eval logging skipped for {info.key}: {exc})", flush=True)
            finally:
                end_experiments([comet_logger])

    if args.fragments:
        print(f"wrote {len(results)} fragment(s) to {output_dir / 'results'}")
    else:
        result_path = output_dir / config["output"].get("result_file", "result.yaml")
        write_yaml(result_path, results)
        print(f"wrote {result_path}")


if __name__ == "__main__":
    main()
