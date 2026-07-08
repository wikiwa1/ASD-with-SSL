from __future__ import annotations

try:  # import before torch so Comet can auto-instrument
    import comet_ml  # noqa: F401
except ImportError:
    comet_ml = None

import argparse
import os
from pathlib import Path

import lightning.pytorch as pl
import numpy as np
import torch
from lightning.pytorch.callbacks import ModelCheckpoint

from audio_ssl.scripts.train_jepa import gather_normal_train_files, global_rank_from_env
from audio_ssl.src.data.jepa_datamodule import JEPASpectrogramDataModule
from audio_ssl.src.data.splits import make_baseline_split
from audio_ssl.src.features.frontends import stack_features
from audio_ssl.src.lightning.auc_monitor import PeriodicAUCMonitor
from audio_ssl.src.lightning.dino_module import LitDINO
from audio_ssl.src.utils.config import load_config, merge_cli_overrides
from audio_ssl.src.utils.io import ensure_dir, write_yaml
from audio_ssl.src.utils.loggers import build_loggers, comet_experiment_key, end_experiments, load_env
from audio_ssl.src.utils.runs import create_run_dir, feature_cache_root
from audio_ssl.src.utils.seed import seed_everything
from audio_ssl.src.utils.slurm import slurm_ddp_plugins

RUN_KEY = "jepa_global"  # shared checkpoint subdir so the embedding eval finds it


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DINO pretraining (frequency-band multi-crop) on normal MIMII.")
    parser.add_argument("--config", default="audio_ssl/configs/dino_baseline.yaml")
    parser.add_argument("--base-directory")
    parser.add_argument("--run-dir")
    parser.add_argument("--max-epochs", type=int)
    parser.add_argument("--accelerator")
    parser.add_argument("--devices")
    parser.add_argument("--num-nodes", type=int)
    parser.add_argument("--strategy")
    parser.add_argument("--precision")
    return parser.parse_args()


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
    global_rank = global_rank_from_env()
    if global_rank == 0:
        cfg_tmp = output_dir / f".config.{os.getpid()}.tmp"
        write_yaml(cfg_tmp, config)
        cfg_tmp.replace(output_dir / "config.yaml")

    train_files = gather_normal_train_files(config)
    print(f"pooled {len(train_files)} normal-train clips", flush=True)

    datamodule = JEPASpectrogramDataModule(
        train_files=train_files, feature_kwargs=feature_cfg,
        batch_size=int(fit_cfg["batch_size"]), num_workers=int(fit_cfg.get("num_workers", 8)),
        val_split=float(fit_cfg.get("val_split", 0.02)), seed=int(config.get("seed", 42)),
        cache_path=cache_path,
    )
    if global_rank == 0:
        datamodule.setup()
    else:
        import time
        while not cache_path.exists():
            time.sleep(10)
        datamodule.setup()

    backbone = model_cfg.get("backbone", "resnet18")
    backbone_cfg = None
    if backbone == "beats":
        from audio_ssl.src.models.beats_jepa.encoder import BEATsEncoder
        backbone_cfg = {
            "beats_cfg": BEATsEncoder.load_pretrained_cfg(model_cfg["beats_checkpoint"]),
            "beats_checkpoint": model_cfg["beats_checkpoint"],
            "finetune_last_n": int(model_cfg.get("finetune_last_n", 2)),
            "target_frames": int(feature_cfg["target_frames"]),
        }

    module = LitDINO(
        backbone=backbone,
        backbone_cfg=backbone_cfg,
        crop_axis=model_cfg.get("crop_axis", "freq"),
        input_norm=model_cfg.get("input_norm", "instance"),
        out_dim=int(model_cfg.get("out_dim", 1024)),
        head_hidden=int(model_cfg.get("head_hidden", 512)),
        head_bottleneck=int(model_cfg.get("head_bottleneck", 64)),
        n_global=int(model_cfg.get("n_global", 2)), n_local=int(model_cfg.get("n_local", 6)),
        global_frac=float(model_cfg.get("global_frac", 0.6)),
        local_frac=float(model_cfg.get("local_frac", 0.25)),
        crop_mels=int(model_cfg.get("crop_mels", 128)),
        crop_frames=int(model_cfg.get("crop_frames", 128)),
        teacher_temp=float(model_cfg.get("teacher_temp", 0.04)),
        student_temp=float(model_cfg.get("student_temp", 0.1)),
        center_momentum=float(model_cfg.get("center_momentum", 0.9)),
        ema_momentum=float(model_cfg.get("ema_momentum", 0.996)),
        augment=config.get("augment", {}),
        lr=float(fit_cfg["lr"]), weight_decay=float(fit_cfg.get("weight_decay", 0.04)),
        warmup_frac=float(fit_cfg.get("warmup_frac", 0.0)),
        lr_schedule=fit_cfg.get("lr_schedule", "constant"),
    )

    loggers = build_loggers(
        run_name=output_dir.name, csv_save_dir=log_root, config=config,
        extra_params={"run": output_dir.name, "arch": "dino", "n_train_files": len(train_files),
                      "backbone": model_cfg.get("backbone", "resnet18"),
                      "out_dim": int(model_cfg.get("out_dim", 1024)),
                      "n_global": int(model_cfg.get("n_global", 2)),
                      "n_local": int(model_cfg.get("n_local", 6)),
                      "fit/epochs": int(fit_cfg["epochs"]), "fit/lr": float(fit_cfg["lr"]),
                      "fit/batch_size": int(fit_cfg["batch_size"]),
                      **{f"feature/{k}": v for k, v in feature_cfg.items()}},
    )
    experiment_key = comet_experiment_key(loggers)
    if experiment_key and global_rank == 0:
        (checkpoint_dir / "comet_experiment.txt").write_text(experiment_key)

    ckpt_cfg = config.get("checkpoint", {})
    checkpoint = ModelCheckpoint(dirpath=checkpoint_dir, filename="{epoch:03d}", save_last=True,
                                 save_top_k=-1, every_n_epochs=int(ckpt_cfg.get("every_n_epochs", 10)),
                                 save_weights_only=True, auto_insert_metric_name=False)
    callbacks = [checkpoint]

    monitor_cfg = config.get("monitor", {})
    if monitor_cfg.get("enabled", False):
        target = Path(monitor_cfg.get("target_dir", "dataset/0_dB/fan/id_00"))
        split = make_baseline_split(target)
        mkey = target.as_posix().strip("./").replace("/", "_")
        mcache = feature_cache_root(base_output) / f"monitor_{mkey}_{feature_cfg['n_mels']}m_{feature_cfg['target_frames']}t.npz"
        if not mcache.exists():
            mcache.parent.mkdir(parents=True, exist_ok=True)
            normal = stack_features(split.train_files, msg="monitor fit specs", **feature_cfg)
            evaluation = stack_features(split.eval_files, msg="monitor eval specs", **feature_cfg)
            tmp = mcache.with_suffix(".tmp.npz")
            np.savez(tmp, normal=normal, evaluation=evaluation, labels=split.eval_labels)
            tmp.replace(mcache)
        with np.load(mcache) as cached:
            callbacks.append(PeriodicAUCMonitor(
                torch.from_numpy(cached["normal"]).unsqueeze(1),
                torch.from_numpy(cached["evaluation"]).unsqueeze(1), cached["labels"],
                emb_cfg=config.get("embedding", {}),
                every_n_epochs=int(monitor_cfg.get("every_n_epochs", 1)),
                batch_size=int(fit_cfg["batch_size"]),
                metric_name=monitor_cfg.get("metric_name", "monitor_AUC")))

    trainer = pl.Trainer(
        max_epochs=int(fit_cfg["epochs"]), accelerator=trainer_cfg.get("accelerator", "auto"),
        devices=trainer_cfg.get("devices", "auto"), num_nodes=int(trainer_cfg.get("num_nodes", 1)),
        strategy=trainer_cfg.get("strategy", "auto"), precision=trainer_cfg.get("precision", "bf16-mixed"),
        gradient_clip_val=float(trainer_cfg.get("gradient_clip_val", 1.0)),
        log_every_n_steps=int(trainer_cfg.get("log_every_n_steps", 20)),
        callbacks=callbacks, logger=loggers,
        plugins=slurm_ddp_plugins(),
    )
    trainer.fit(module, datamodule=datamodule)
    end_experiments(loggers)
    print(f"{RUN_KEY}: checkpoints -> {checkpoint_dir}", flush=True)


if __name__ == "__main__":
    main()
