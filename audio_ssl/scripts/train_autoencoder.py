from __future__ import annotations

try:  # import before torch so Comet can auto-instrument the run
    import comet_ml  # noqa: F401
except ImportError:
    comet_ml = None

import argparse
import os
import time
from pathlib import Path

import lightning.pytorch as pl
import numpy as np
from lightning.pytorch.callbacks import ModelCheckpoint

from audio_ssl.src.data.mimii_datamodule import MIMIIAutoEncoderDataModule
from audio_ssl.src.data.splits import find_target_dirs, make_baseline_split, parse_target_info
from audio_ssl.src.features.logmel import list_to_vector_array
from audio_ssl.src.lightning.autoencoder_module import LitAutoEncoder
from audio_ssl.src.utils.config import load_config, merge_cli_overrides
from audio_ssl.src.utils.io import ensure_dir, save_npz
from audio_ssl.src.utils.loggers import build_loggers, comet_experiment_key, end_experiments, load_env
from audio_ssl.src.utils.runs import create_run_dir, feature_cache_root
from audio_ssl.src.utils.seed import seed_everything


def global_rank_from_env() -> int:
    for name in ("RANK", "SLURM_PROCID", "LOCAL_RANK"):
        value = os.environ.get(name)
        if value is not None and value.isdigit():
            return int(value)
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the MIMII autoencoder baseline.")
    parser.add_argument("--config", default="audio_ssl/configs/autoencoder_baseline.yaml")
    parser.add_argument("--target-dir", help="Train only one MIMII target directory.")
    parser.add_argument("--base-directory", help="Override data.base_directory.")
    parser.add_argument("--run-dir", help="Reuse an existing run folder instead of creating a new one.")
    parser.add_argument("--max-epochs", type=int, help="Override fit.epochs.")
    parser.add_argument("--accelerator")
    parser.add_argument("--devices")
    parser.add_argument("--num-nodes", type=int)
    parser.add_argument("--strategy")
    parser.add_argument("--precision")
    return parser.parse_args()


def target_dirs_from_args(config: dict, args: argparse.Namespace) -> list[Path]:
    if args.target_dir:
        return [Path(args.target_dir)]
    return find_target_dirs(config["data"]["base_directory"])


def main() -> None:
    args = parse_args()
    config = merge_cli_overrides(load_config(args.config), args)
    load_env()  # pull COMET_* credentials from .env if present
    seed_everything(int(config.get("seed", 42)))

    data_cfg = config["data"]
    feature_cfg = {**config["feature"], "channel": data_cfg.get("channel", 0)}
    fit_cfg = config["fit"]
    trainer_cfg = config["trainer"]
    base_output = config["output"]["directory"]
    output_dir = ensure_dir(args.run_dir) if args.run_dir else create_run_dir(base_output)
    # Feature cache is shared across runs (logmel vectors depend on data+feature cfg,
    # not the run), so reruns don't recompute features.
    feature_cache_dir = ensure_dir(feature_cache_root(base_output))
    checkpoint_root = ensure_dir(output_dir / "checkpoints")
    log_root = ensure_dir(output_dir / "logs")
    print(f"RUN DIR: {output_dir}", flush=True)

    target_dirs = target_dirs_from_args(config, args)
    if not target_dirs:
        raise FileNotFoundError(f"No target dirs found under {data_cfg['base_directory']}")

    for target_dir in target_dirs:
        info = parse_target_info(target_dir, base_directory=data_cfg["base_directory"])
        split = make_baseline_split(
            target_dir,
            normal_dir_name=data_cfg.get("normal_dir_name", "normal"),
            abnormal_dir_name=data_cfg.get("abnormal_dir_name", "abnormal"),
            ext=data_cfg.get("ext", "wav"),
        )

        train_features = None
        cache_path = feature_cache_dir / f"train_{info.key}.npz"
        global_rank = global_rank_from_env()
        if data_cfg.get("cache_features", True) and cache_path.exists():
            with np.load(cache_path) as cached:
                train_features = cached["train_features"]
        elif data_cfg.get("cache_features", True):
            if global_rank == 0:
                train_features = list_to_vector_array(
                    split.train_files,
                    msg=f"generate train features {info.key}",
                    **feature_cfg,
                )
                save_npz(cache_path, train_features=train_features)
            else:
                while not cache_path.exists():
                    time.sleep(5)
                with np.load(cache_path) as cached:
                    train_features = cached["train_features"]

        datamodule = MIMIIAutoEncoderDataModule(
            train_files=split.train_files,
            feature_kwargs=feature_cfg,
            batch_size=int(fit_cfg["batch_size"]),
            validation_split=float(fit_cfg["validation_split"]),
            shuffle=bool(fit_cfg["shuffle"]),
            num_workers=int(fit_cfg.get("num_workers", 4)),
            seed=int(config.get("seed", 42)),
            train_features=train_features,
        )
        input_dim = int(config["feature"]["n_mels"]) * int(config["feature"]["frames"])
        module = LitAutoEncoder(
            input_dim=input_dim,
            hidden_dims=config["model"].get("hidden_dims", [64, 64, 8, 64, 64]),
            lr=float(fit_cfg["lr"]),
        )

        checkpoint_dir = ensure_dir(checkpoint_root / info.key)
        checkpoint = ModelCheckpoint(
            dirpath=checkpoint_dir,
            filename="{epoch:03d}-{val_loss:.6f}",
            monitor="val_loss",
            mode="min",
            save_last=True,
            auto_insert_metric_name=False,
        )
        loggers = build_loggers(
            run_name=info.key,
            csv_save_dir=log_root,
            config=config,
            extra_params={
                "run": output_dir.name,
                "machine_type": info.machine_type,
                "machine_id": info.machine_id,
                "db": info.db,
                "n_train_files": len(split.train_files),
                "n_eval_files": len(split.eval_files),
                "fit/epochs": int(fit_cfg["epochs"]),
                "fit/batch_size": int(fit_cfg["batch_size"]),
                "fit/lr": float(fit_cfg["lr"]),
                "fit/validation_split": float(fit_cfg["validation_split"]),
                **{f"feature/{name}": value for name, value in feature_cfg.items()},
            },
        )

        # Persist the Comet experiment key so eval can reattach test metrics + ROC.
        experiment_key = comet_experiment_key(loggers)
        if experiment_key and global_rank == 0:
            (checkpoint_dir / "comet_experiment.txt").write_text(experiment_key)

        trainer = pl.Trainer(
            max_epochs=int(fit_cfg["epochs"]),
            accelerator=trainer_cfg.get("accelerator", "auto"),
            devices=trainer_cfg.get("devices", "auto"),
            num_nodes=int(trainer_cfg.get("num_nodes", 1)),
            strategy=trainer_cfg.get("strategy", "auto"),
            precision=trainer_cfg.get("precision", "32-true"),
            log_every_n_steps=int(trainer_cfg.get("log_every_n_steps", 20)),
            callbacks=[checkpoint],
            logger=loggers,
        )
        trainer.fit(module, datamodule=datamodule)
        end_experiments(loggers)
        print(f"{info.key}: best checkpoint -> {checkpoint.best_model_path}")


if __name__ == "__main__":
    main()
