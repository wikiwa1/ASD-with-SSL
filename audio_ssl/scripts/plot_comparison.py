from __future__ import annotations

import argparse

import matplotlib

matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from audio_ssl.src.utils.io import read_yaml  # noqa: E402

MACHINES = ["fan", "pump", "slider", "valve"]
SNRS = ["6_dB", "0_dB", "-6_dB"]  # easy -> hard
IDS = ["id_00", "id_02", "id_04", "id_06"]
COLORS = ["#9aa6b2", "#0b8f8a", "#e07b39", "#7b5ea7"]  # method colors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Grid plot comparing per-target AUC across methods into a single PDF."
    )
    parser.add_argument(
        "--result", action="append", required=True, metavar="LABEL=PATH",
        help="A result yaml with a label, e.g. AE=.../result.yaml (repeatable).",
    )
    parser.add_argument("--metric", default="AUC", help="Metric field to plot (AUC or pAUC).")
    parser.add_argument("--output", required=True, help="Output PDF path.")
    parser.add_argument("--title", default="MIMII anomaly detection")
    return parser.parse_args()


def load(path: str, metric: str) -> dict[tuple[str, str, str], float]:
    data = read_yaml(path)
    return {
        (v["machine_type"], v["machine_id"], v["db"]): float(v[metric])
        for v in data.values()
    }


def main() -> None:
    args = parse_args()
    methods = []
    for spec in args.result:
        label, _, path = spec.partition("=")
        methods.append((label, load(path, args.metric)))

    fig, axes = plt.subplots(len(MACHINES), len(SNRS), figsize=(13, 12), sharey=True)
    width = 0.8 / len(methods)
    x = np.arange(len(IDS))

    for r, machine in enumerate(MACHINES):
        for c, snr in enumerate(SNRS):
            ax = axes[r, c]
            for i, (label, data) in enumerate(methods):
                vals = [data.get((machine, idn, snr), np.nan) for idn in IDS]
                bars = ax.bar(x + i * width, vals, width, label=label, color=COLORS[i % len(COLORS)])
                for rect, v in zip(bars, vals):
                    if not np.isnan(v):
                        ax.text(rect.get_x() + rect.get_width() / 2, v + 0.01, f"{v:.2f}",
                                ha="center", va="bottom", fontsize=6, rotation=90)
            ax.axhline(0.5, ls="--", color="gray", lw=0.8)
            ax.set_ylim(0, 1.12)
            ax.set_xticks(x + width * (len(methods) - 1) / 2)
            ax.set_xticklabels([idn.replace("id_", "") for idn in IDS], fontsize=8)
            if c == 0:
                ax.set_ylabel(machine, fontsize=12, fontweight="bold")
            if r == 0:
                ax.set_title(snr.replace("_dB", " dB"), fontsize=12)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(methods), bbox_to_anchor=(0.5, 0.965), fontsize=11)
    means = "    ".join(f"{label} mean={np.nanmean(list(data.values())):.3f}" for label, data in methods)
    fig.suptitle(f"{args.title}  —  {args.metric} (rows: machine, cols: SNR, x: model id)\n{means}",
                 fontsize=13, y=0.995)
    fig.text(0.5, 0.015, "model id", ha="center", fontsize=11)
    fig.tight_layout(rect=[0, 0.025, 1, 0.93])
    fig.savefig(args.output)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
