from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from lightning.pytorch.loggers import Logger

try:  # imported before torch elsewhere so Comet can auto-instrument
    import comet_ml  # noqa: F401
    _COMET_INSTALLED = True
except ImportError:
    _COMET_INSTALLED = False

try:
    from dotenv import find_dotenv, load_dotenv
    _DOTENV_INSTALLED = True
except ImportError:
    _DOTENV_INSTALLED = False


def _env_rank() -> int:
    for name in ("RANK", "SLURM_PROCID", "LOCAL_RANK"):
        value = os.environ.get(name)
        if value and value.isdigit():
            return int(value)
    return 0


def load_env(explicit_path: str | Path | None = None) -> None:
    """Load the repo `.env` so COMET_* credentials reach the process environment.

    No-op when python-dotenv is missing. Existing environment variables win
    (override=False) so an exported COMET_API_KEY is never clobbered.
    """
    if not _DOTENV_INSTALLED:
        return
    candidates: list[Path] = []
    if explicit_path is not None:
        candidates.append(Path(explicit_path))
    found = find_dotenv(usecwd=True)
    if found:
        candidates.append(Path(found))
    # Reliable fallback: repo root relative to this file (…/audio_ssl/src/utils/loggers.py)
    candidates.append(Path(__file__).resolve().parents[3] / ".env")
    for candidate in candidates:
        if candidate and candidate.is_file():
            load_dotenv(candidate, override=False)
            return


def build_loggers(
    *,
    run_name: str,
    csv_save_dir: str | Path,
    config: dict,
    extra_params: dict[str, Any] | None = None,
) -> list[Logger]:
    """Build Lightning loggers for one training run.

    Returns a single CometLogger on global rank 0 when `logging.comet.enabled` is true,
    comet_ml is installed, and a COMET_API_KEY is available; otherwise returns NO logger.
    We deliberately do NOT use Lightning's CSVLogger — its header-rewrite crashes under
    DDP ("dict contains fields not in fieldnames") with our train/val logging cadence, and
    Comet already captures the curves. `csv_save_dir` is accepted for signature stability.

    Comet config (under `logging.comet` in the YAML):
      enabled: bool             turn Comet on/off
      online: bool              False -> offline archives (use on nodes w/o internet)
      workspace / project: str  null -> COMET_WORKSPACE / COMET_PROJECT_NAME from .env
      offline_directory: str    null -> <output.directory>/comet_offline (offline only)
      tags: list[str]           applied to every experiment
    """
    no_loggers: list[Logger] = []

    comet_cfg = (config.get("logging") or {}).get("comet") or {}
    if not comet_cfg.get("enabled", False):
        return no_loggers
    if _env_rank() != 0:
        return no_loggers  # DDP-safe: only global rank 0 creates a Comet experiment
    if not _COMET_INSTALLED:
        print("[loggers] logging.comet.enabled but comet_ml is not installed; no logger.", flush=True)
        return no_loggers

    api_key = os.environ.get("COMET_API_KEY")
    if not api_key:
        print("[loggers] COMET_API_KEY not set (.env not loaded?); no logger.", flush=True)
        return no_loggers

    online = bool(comet_cfg.get("online", True))
    kwargs: dict[str, Any] = {"name": run_name}
    tags = comet_cfg.get("tags")
    if tags:
        kwargs["tags"] = list(tags)
    if not online:
        offline_dir = comet_cfg.get("offline_directory") or (
            Path(config["output"]["directory"]) / "comet_offline"
        )
        Path(offline_dir).mkdir(parents=True, exist_ok=True)
        kwargs["offline_directory"] = str(offline_dir)

    # Comet's git/conda/system/CO2 auto-collection scans the filesystem and makes
    # network calls at experiment creation — slow and hang-prone on HPC (NERSC home
    # GPFS). Disable by default; override individually via logging.comet.options.
    experiment_options = {
        "log_code": False,
        "log_graph": False,
        "log_git_metadata": False,
        "log_git_patch": False,
        "log_env_details": False,
        "auto_log_co2": False,
    }
    experiment_options.update(comet_cfg.get("options") or {})

    from lightning.pytorch.loggers import CometLogger

    comet_logger = CometLogger(
        api_key=api_key,
        workspace=comet_cfg.get("workspace") or os.environ.get("COMET_WORKSPACE"),
        project=comet_cfg.get("project") or os.environ.get("COMET_PROJECT_NAME"),
        online=online,
        **experiment_options,
        **kwargs,
    )
    if extra_params:
        comet_logger.log_hyperparams(extra_params)
    print("[loggers] logging to Comet (no CSVLogger).", flush=True)
    return [comet_logger]


def end_experiments(loggers: list[Logger]) -> None:
    """Explicitly end Comet experiments after a run.

    Lightning's CometLogger.finalize() only flushes, never ends. In a per-target
    loop we want each experiment closed before the next target starts; for offline
    mode end() is also what writes the archive to disk. No-op for other loggers and
    for ranks where no experiment was created.
    """
    for logger in loggers:
        experiment = getattr(logger, "_experiment", None)
        if experiment is not None and hasattr(experiment, "end"):
            experiment.end()


def comet_experiment_key(loggers: list[Logger]) -> str | None:
    """Return the Comet experiment key from a built logger list, or None.

    Used to persist the key at train time so evaluation can reattach test metrics
    to the same experiment. Only rank 0 has a live experiment.
    """
    for logger in loggers:
        experiment = getattr(logger, "_experiment", None)
        if experiment is not None and hasattr(experiment, "get_key"):
            try:
                return experiment.get_key()
            except Exception:
                pass
        key = getattr(logger, "_experiment_key", None)
        if key:
            return key
    return None


def resume_comet_experiment(config: dict, experiment_key: str):
    """Reattach to an existing Comet experiment for logging (e.g. eval metrics).

    Returns a Lightning CometLogger bound to `experiment_key` (mode="get"), or None
    when Comet is disabled/unavailable, no API key is set, or no key was given.
    Requires online mode — resuming an offline archive is not supported.
    """
    comet_cfg = (config.get("logging") or {}).get("comet") or {}
    if not comet_cfg.get("enabled", False) or not _COMET_INSTALLED or not experiment_key:
        return None
    api_key = os.environ.get("COMET_API_KEY")
    if not api_key:
        print("[loggers] COMET_API_KEY not set; cannot resume experiment.", flush=True)
        return None
    if not bool(comet_cfg.get("online", True)):
        print("[loggers] online=false; skipping eval logging to existing experiment.", flush=True)
        return None

    from lightning.pytorch.loggers import CometLogger

    return CometLogger(
        api_key=api_key,
        workspace=comet_cfg.get("workspace") or os.environ.get("COMET_WORKSPACE"),
        project=comet_cfg.get("project") or os.environ.get("COMET_PROJECT_NAME"),
        online=True,
        experiment_key=experiment_key,
        mode="get",
        log_code=False,
        log_graph=False,
        log_git_metadata=False,
        log_git_patch=False,
        log_env_details=False,
        auto_log_co2=False,
    )
