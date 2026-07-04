from __future__ import annotations

try:  # import before torch so Comet can auto-instrument
    import comet_ml  # noqa: F401
except ImportError:
    comet_ml = None

import argparse
import os
import time
from pathlib import Path

import lightning.pytorch as pl
import numpy as np
import torch
from lightning.pytorch.callbacks import ModelCheckpoint

from audio_ssl.src.data.jepa_datamodule import JEPASpectrogramDataModule
from audio_ssl.src.data.splits import discover_targets, make_baseline_split
from audio_ssl.src.features.frontends import stack_features
from audio_ssl.src.lightning.auc_monitor import PeriodicAUCMonitor
from audio_ssl.src.lightning.jepa_module import LitJEPA
from audio_ssl.src.utils.config import load_config, merge_cli_overrides
from audio_ssl.src.utils.io import ensure_dir, write_yaml
from audio_ssl.src.utils.loggers import build_loggers, comet_experiment_key, end_experiments, load_env
from audio_ssl.src.utils.runs import create_run_dir, feature_cache_root
from audio_ssl.src.utils.seed import seed_everything

RUN_KEY = "jepa_global"  # one global encoder -> one checkpoint dir / Comet experiment


def global_rank_from_env() -> int:
    for name in ("RANK", "SLURM_PROCID", "LOCAL_RANK"):
        value = os.environ.get(name)
        if value is not None and value.isdigit():
            return int(value)
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pretrain the global audio-JEPA encoder on normal MIMII.")
    parser.add_argument("--config", default="audio_ssl/configs/jepa_baseline.yaml")
    parser.add_argument("--base-directory", help="Override data.base_directory.")
    parser.add_argument("--run-dir", help="Reuse an existing run folder instead of creating a new one.")
    parser.add_argument("--max-epochs", type=int, help="Override fit.epochs.")
    parser.add_argument("--accelerator")
    parser.add_argument("--devices")
    parser.add_argument("--num-nodes", type=int)
    parser.add_argument("--strategy")
    parser.add_argument("--precision")
    return parser.parse_args()


def gather_normal_train_files(config: dict) -> list:
    data_cfg = config["data"]
    files = []
    for target_dir in discover_targets(config):
        split = make_baseline_split(
            target_dir,
            normal_dir_name=data_cfg.get("normal_dir_name", "normal"),
            abnormal_dir_name=data_cfg.get("abnormal_dir_name", "abnormal"),
            ext=data_cfg.get("ext", "wav"),
        )
        files.extend(split.train_files)
    return files


def main() -> None:
    args = parse_args()
    config = merge_cli_overrides(load_config(args.config), args)
    load_env()
    seed_everything(int(config.get("seed", 42)))

    data_cfg = config["data"]
    feature_cfg = {**config["feature"], "channel": data_cfg.get("channel", 0)}
    fit_cfg = config["fit"]
    trainer_cfg = config["trainer"]
    model_cfg = config["model"]

    base_output = config["output"]["directory"]
    output_dir = ensure_dir(args.run_dir) if args.run_dir else create_run_dir(base_output)
    checkpoint_dir = ensure_dir(output_dir / "checkpoints" / RUN_KEY)
    log_root = ensure_dir(output_dir / "logs")
    cache_path = feature_cache_root(base_output) / f"jepa_specs_{feature_cfg['n_mels']}m_{feature_cfg['target_frames']}t.npy"
    print(f"RUN DIR: {output_dir}", flush=True)
    if global_rank_from_env() == 0:
        # Make the run self-describing so eval uses the SAME machines/features/cache
        # instead of trusting whatever --config the user later passes.
        cfg_tmp = output_dir / f".config.{os.getpid()}.tmp"
        write_yaml(cfg_tmp, config)
        cfg_tmp.replace(output_dir / "config.yaml")

    train_files = gather_normal_train_files(config)
    print(f"pooled {len(train_files)} normal-train clips across all targets", flush=True)

    datamodule = JEPASpectrogramDataModule(
        train_files=train_files,
        feature_kwargs=feature_cfg,
        batch_size=int(fit_cfg["batch_size"]),
        num_workers=int(fit_cfg.get("num_workers", 8)),
        val_split=float(fit_cfg.get("val_split", 0.02)),
        seed=int(config.get("seed", 42)),
        cache_path=cache_path,
    )

    # Rank 0 extracts/caches spectrograms; other ranks wait for the atomic cache file.
    global_rank = global_rank_from_env()
    if global_rank == 0:
        datamodule.setup()
    else:
        while not cache_path.exists():
            time.sleep(10)
        datamodule.setup()

    module = LitJEPA(
        n_mels=int(feature_cfg["n_mels"]),
        target_frames=int(feature_cfg["target_frames"]),
        patch_mels=int(model_cfg["patch_mels"]),
        patch_frames=int(model_cfg["patch_frames"]),
        embed_dim=int(model_cfg["embed_dim"]),
        depth=int(model_cfg["depth"]),
        heads=int(model_cfg["heads"]),
        predictor_dim=int(model_cfg["predictor_dim"]),
        predictor_depth=int(model_cfg["predictor_depth"]),
        predictor_heads=int(model_cfg["predictor_heads"]),
        mlp_ratio=float(model_cfg.get("mlp_ratio", 4.0)),
        num_blocks=int(model_cfg["num_blocks"]),
        mask_scale=tuple(model_cfg["mask_scale"]),
        mask_aspect=tuple(model_cfg["mask_aspect"]),
        ema_start=float(model_cfg["ema_start"]),
        ema_end=float(model_cfg["ema_end"]),
        lr=float(fit_cfg["lr"]),
        weight_decay=float(fit_cfg.get("weight_decay", 0.05)),
        warmup_frac=float(fit_cfg.get("warmup_frac", 0.1)),
        lr_schedule=fit_cfg.get("lr_schedule", "cosine"),
        mel_mean=datamodule.mel_mean,
        mel_std=datamodule.mel_std,
    )

    loggers = build_loggers(
        run_name=output_dir.name,
        csv_save_dir=log_root,
        config=config,
        extra_params={
            "run": output_dir.name,
            "n_train_files": len(train_files),
            "embed_dim": int(model_cfg["embed_dim"]),
            "depth": int(model_cfg["depth"]),
            "predictor_dim": int(model_cfg["predictor_dim"]),
            "num_blocks": int(model_cfg["num_blocks"]),
            "fit/epochs": int(fit_cfg["epochs"]),
            "fit/batch_size": int(fit_cfg["batch_size"]),
            "fit/lr": float(fit_cfg["lr"]),
            **{f"feature/{k}": v for k, v in feature_cfg.items()},
        },
    )
    experiment_key = comet_experiment_key(loggers)
    if experiment_key and global_rank == 0:
        (checkpoint_dir / "comet_experiment.txt").write_text(experiment_key)

    # Periodic checkpoints (no val_loss monitor / no top-k eviction -> no DDP eviction
    # crash). save_top_k=-1 keeps every Nth-epoch checkpoint so AUC-vs-step can be plotted;
    # weights-only keeps them small. last.ckpt is what eval loads.
    ckpt_cfg = config.get("checkpoint", {})
    checkpoint = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename="epoch{epoch:03d}",
        save_last=True,
        save_top_k=-1,
        every_n_epochs=int(ckpt_cfg.get("every_n_epochs", 10)),
        save_weights_only=True,
    )
    callbacks = [checkpoint]

    monitor_cfg = config.get("monitor", {})
    if monitor_cfg.get("enabled", False):
        # Pre-extract the monitor target's spectrograms ONCE, BEFORE fit, and cache them, so
        # the callback never forks a librosa pool during training (a mid-training fork
        # inherits Lightning's CUDA SIGTERM handler and crashes). Cached -> reused on reruns.
        monitor_target = Path(monitor_cfg.get("target_dir", "dataset/0_dB/fan/id_00"))
        split = make_baseline_split(monitor_target)
        cache_key = monitor_target.as_posix().strip("./").replace("/", "_")
        monitor_cache = feature_cache_root(base_output) / f"monitor_{cache_key}_{feature_cfg['n_mels']}m_{feature_cfg['target_frames']}t.npz"
        if not monitor_cache.exists():
            monitor_cache.parent.mkdir(parents=True, exist_ok=True)
            normal = stack_features(split.train_files, msg="monitor fit specs", **feature_cfg)
            evaluation = stack_features(split.eval_files, msg="monitor eval specs", **feature_cfg)
            tmp = monitor_cache.with_suffix(".tmp.npz")
            np.savez(tmp, normal=normal, evaluation=evaluation, labels=split.eval_labels)
            tmp.replace(monitor_cache)
        with np.load(monitor_cache) as cached:
            normal_specs = torch.from_numpy(cached["normal"]).unsqueeze(1)
            eval_specs = torch.from_numpy(cached["evaluation"]).unsqueeze(1)
            labels = cached["labels"]
        callbacks.append(
            PeriodicAUCMonitor(
                normal_specs, eval_specs, labels,
                emb_cfg=config.get("embedding", {}),
                every_n_epochs=int(monitor_cfg.get("every_n_epochs", 1)),
                batch_size=int(fit_cfg["batch_size"]),
                metric_name=monitor_cfg.get("metric_name", "monitor_AUC"),
            )
        )

    trainer = pl.Trainer(
        max_epochs=int(fit_cfg["epochs"]),
        accelerator=trainer_cfg.get("accelerator", "auto"),
        devices=trainer_cfg.get("devices", "auto"),
        num_nodes=int(trainer_cfg.get("num_nodes", 1)),
        strategy=trainer_cfg.get("strategy", "auto"),
        precision=trainer_cfg.get("precision", "bf16-mixed"),
        gradient_clip_val=float(trainer_cfg.get("gradient_clip_val", 1.0)),
        log_every_n_steps=int(trainer_cfg.get("log_every_n_steps", 20)),
        callbacks=callbacks,
        logger=loggers,
    )
    trainer.fit(module, datamodule=datamodule)
    end_experiments(loggers)
    print(f"{RUN_KEY}: checkpoints -> {checkpoint_dir}", flush=True)


if __name__ == "__main__":
    main()
