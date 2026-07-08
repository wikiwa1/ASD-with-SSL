from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

SNRS = ["6_dB", "0_dB", "-6_dB"]
IDS = ["id_00", "id_02", "id_04", "id_06"]
COLORS = ["#9aa6b2", "#0b8f8a", "#e07b39", "#7b5ea7"]
_INTEGRATE = getattr(np, "trapezoid", np.trapz)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Per machine-ID ROC curves, vertically averaged over SNRs, one curve per method."
    )
    parser.add_argument("--roc", action="append", required=True, metavar="LABEL=DIR",
                        help="Directory of <machine>_<id>_<snr>.npz ROC files with a label (repeatable).")
    parser.add_argument("--machines", default="fan", help="Comma-separated machine types (e.g. fan).")
    parser.add_argument("--output", required=True)
    parser.add_argument("--title", default="ROC (averaged over SNRs)")
    return parser.parse_args()


def snr_averaged(roc_dir: Path, machine: str, idn: str, common_fpr: np.ndarray):
    """Vertical-average TPR over the 3 SNRs (avoids mixing per-target score scales) and
    return (mean_tpr, mean_AUC-over-SNRs). None if any SNR file is missing."""
    tprs, aucs = [], []
    for snr in SNRS:
        path = roc_dir / f"{machine}_{idn}_{snr}.npz"
        if not path.exists():
            return None
        with np.load(path) as data:
            fpr, tpr = data["fpr"], data["tpr"]
        tprs.append(np.interp(common_fpr, fpr, tpr))
        aucs.append(float(_INTEGRATE(tpr, fpr)))
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[0] = 0.0
    return mean_tpr, float(np.mean(aucs))


def main() -> None:
    args = parse_args()
    methods = [(spec.partition("=")[0], Path(spec.partition("=")[2])) for spec in args.roc]
    machines = [m.strip() for m in args.machines.split(",") if m.strip()]
    common_fpr = np.linspace(0.0, 1.0, 300)

    fig, axes = plt.subplots(len(machines), len(IDS), figsize=(15, 3.6 * len(machines)),
                             sharex=True, sharey=True, squeeze=False)
    for r, machine in enumerate(machines):
        for c, idn in enumerate(IDS):
            ax = axes[r][c]
            ax.plot([0, 1], [0, 1], ls="--", lw=0.8, color="gray")
            for i, (label, roc_dir) in enumerate(methods):
                curve = snr_averaged(roc_dir, machine, idn, common_fpr)
                if curve is None:
                    continue
                mean_tpr, auc = curve
                ax.plot(common_fpr, mean_tpr, lw=1.8, color=COLORS[i % len(COLORS)],
                        label=f"{label} ({auc:.3f})")
            ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
            ax.legend(loc="lower right", fontsize=8)
            if r == 0:
                ax.set_title(idn.replace("id_", "id "), fontsize=12)
            if c == 0:
                ax.set_ylabel(f"{machine}\nTPR", fontsize=11, fontweight="bold")
            if r == len(machines) - 1:
                ax.set_xlabel("FPR", fontsize=10)

    fig.suptitle(f"{args.title}  (mean AUC over -6/0/6 dB in legend)", fontsize=14, y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(args.output)
    if args.output.endswith(".pdf"):
        fig.savefig(args.output[:-4] + ".png", dpi=90)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
