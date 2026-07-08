from __future__ import annotations

"""SLURM/Lightning glue.

Lightning deliberately refuses to auto-detect SLURM inside interactive allocations
(SLURM_JOB_NAME == "interactive" or "bash", see lightning.fabric...slurm._is_srun_used),
so under our `salloc -q interactive` launchers each srun task silently built its OWN
DDP world by re-spawning itself once per GPU: N duplicate trainings sharing the GPUs,
N rank-0s racing to write checkpoints (the -v1..-v3 dupes) and N interleaved monitor
curves in one log. Passing SLURMEnvironment explicitly restores a single world with
rank = SLURM_PROCID. auto_requeue=False because interactive jobs are never requeued.
"""

import os

from lightning.fabric.plugins.environments.slurm import SLURMEnvironment


def slurm_ddp_plugins() -> list:
    """Explicit SLURMEnvironment when srun launched multiple tasks; else default detection
    (single-task srun, login-node CPU smoke tests) is fine and returns []."""
    if int(os.environ.get("SLURM_NTASKS", "1")) > 1:
        return [SLURMEnvironment(auto_requeue=False)]
    return []
