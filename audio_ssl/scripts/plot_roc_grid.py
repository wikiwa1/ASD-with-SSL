from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402

MACHINES = ["fan", "pump", "slider", "valve"]
SNRS = ["6_dB", "0_dB", "-6_dB"]  # easy -> hard
IDS = ["id_00", "id_02", "id_04", "id_06"]
COLORS = ["#9aa6b2", "#0b8f8a", "#e07b39", "#7b5ea7"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Per-target ROC curves overlaying methods, one PDF page per machine."
    )
    parser.add_argument(
        "--roc", action="append", required=True, metavar="LABEL=DIR",
        help="Directory of <key>.npz ROC files with a label, e.g. AE=run/roc (repeatable).",
    )
    parser.add_argument("--output", required=True, help="Output PDF path.")
    parser.add_argument("--title", default="MIMII ROC curves")
    return parser.parse_args()


def load_curve(roc_dir: Path, key: str) -> tuple[np.ndarray, np.ndarray, float] | None:
    path = roc_dir / f"{key}.npz"
    if not path.exists():
        return None
    with np.load(path) as data:
        fpr, tpr = data["fpr"], data["tpr"]
    integrate = getattr(np, "trapezoid", np.trapz)
    return fpr, tpr, float(integrate(tpr, fpr))


def main() -> None:
    args = parse_args()
    methods = [(spec.partition("=")[0], Path(spec.partition("=")[2])) for spec in args.roc]

    with PdfPages(args.output) as pdf:
        for machine in MACHINES:
            fig, axes = plt.subplots(len(SNRS), len(IDS), figsize=(13, 10),
                                     sharex=True, sharey=True)
            for r, snr in enumerate(SNRS):
                for c, idn in enumerate(IDS):
                    ax = axes[r, c]
                    key = f"{machine}_{idn}_{snr}"
                    ax.plot([0, 1], [0, 1], ls="--", lw=0.8, color="gray")
                    for i, (label, roc_dir) in enumerate(methods):
                        curve = load_curve(roc_dir, key)
                        if curve is None:
                            continue
                        fpr, tpr, auc = curve
                        ax.plot(fpr, tpr, lw=1.6, color=COLORS[i % len(COLORS)],
                                label=f"{label} ({auc:.3f})")
                    ax.set_xlim(0, 1)
                    ax.set_ylim(0, 1.02)
                    ax.legend(loc="lower right", fontsize=7)
                    if r == 0:
                        ax.set_title(idn.replace("id_", "id "), fontsize=11)
                    if c == 0:
                        ax.set_ylabel(f"{snr.replace('_dB', ' dB')}\nTPR", fontsize=10)
                    if r == len(SNRS) - 1:
                        ax.set_xlabel("FPR", fontsize=9)
            fig.suptitle(f"{args.title} — {machine}", fontsize=14, y=0.98)
            fig.tight_layout(rect=[0, 0, 1, 0.96])
            pdf.savefig(fig)
            plt.close(fig)
    print(f"wrote {args.output} ({len(MACHINES)} pages)")


if __name__ == "__main__":
    main()
