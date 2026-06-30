from __future__ import annotations

import random
from datetime import datetime
from pathlib import Path

# Comet-style adjective_noun slug, generated locally so the run folder is unique even
# though each MIMII target spawns its own Comet experiment. SystemRandom is used so the
# name is NOT affected by seed_everything() (which would otherwise make every run identical).
_ADJECTIVES = (
    "amber brisk calm clever cosmic crimson dapper eager fuzzy golden hidden jolly keen "
    "lucid mellow nimble olive plucky quiet rapid silver tidal vivid witty zesty"
).split()
_NOUNS = (
    "otter falcon cedar comet quartz maple raven lynx willow ember pine heron koi sparrow "
    "basil cobra delta fjord glacier harbor ivy jasper kelp lotus mesa"
).split()


def _bigram(rng: random.Random) -> str:
    return f"{rng.choice(_ADJECTIVES)}_{rng.choice(_NOUNS)}"


def generate_run_name(prefix: str, now: datetime | None = None) -> str:
    """e.g. autoencoder_baseline_20260629_amber_otter_0423"""
    rng = random.SystemRandom()
    date = (now or datetime.now()).strftime("%Y%m%d")
    return f"{prefix}_{date}_{_bigram(rng)}_{rng.randint(0, 9999):04d}"


def create_run_dir(base_directory: str | Path) -> Path:
    """Create a fresh, uniquely named run folder next to `base_directory`.

    base_directory's name is the prefix and its parent is the runs root, so
    `audio_ssl/outputs/autoencoder_baseline` ->
    `audio_ssl/outputs/autoencoder_baseline_<date>_<bigram>_<nnnn>`.
    Also updates a `<prefix>_latest` symlink for convenient eval defaults.
    """
    base = Path(base_directory)
    runs_root = base.parent
    prefix = base.name
    run_dir = runs_root / generate_run_name(prefix)
    run_dir.mkdir(parents=True, exist_ok=True)

    latest = runs_root / f"{prefix}_latest"
    try:
        if latest.is_symlink() or latest.exists():
            latest.unlink()
        latest.symlink_to(run_dir.name)  # relative symlink within runs_root
    except OSError:
        pass  # symlinks are a convenience, not required
    return run_dir


def feature_cache_root(base_directory: str | Path) -> Path:
    """Shared (run-independent) logmel feature cache so reruns don't recompute features."""
    base = Path(base_directory)
    return base.parent / f"{base.name}_feature_cache"


def resolve_run_dir(base_directory: str | Path, run_dir: str | Path | None = None) -> Path:
    """Pick the run folder to read for eval: explicit `run_dir`, else `<prefix>_latest`,
    else the legacy base directory itself."""
    if run_dir:
        return Path(run_dir)
    base = Path(base_directory)
    latest = base.parent / f"{base.name}_latest"
    if latest.exists():
        return latest.resolve()
    return base
