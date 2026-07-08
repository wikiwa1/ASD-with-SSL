from __future__ import annotations

import argparse
from pathlib import Path
from statistics import mean

from audio_ssl.src.utils.config import load_config
from audio_ssl.src.utils.io import read_yaml, write_yaml
from audio_ssl.src.utils.runs import resolve_run_dir

# MIMII machine types and SNR conditions, in the order the paper reports them.
MACHINES = ["fan", "pump", "slider", "valve"]
SNRS = ["6_dB", "0_dB", "-6_dB"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge per-target eval fragments and print the MIMII machine x SNR AUC table."
    )
    parser.add_argument("--config", default="audio_ssl/configs/autoencoder_baseline.yaml")
    parser.add_argument("--run-dir", help="Run folder to aggregate (default: the latest run).")
    parser.add_argument("--result-file", default="result.yaml",
                        help="Result file to aggregate (e.g. result_embedding.yaml).")
    return parser.parse_args()


def collect_results(run_dir: Path, result_file: str = "result.yaml") -> dict:
    """Read the named result file; for the default, fall back to per-target fragments
    (parallel eval) when the combined file is absent."""
    combined = run_dir / result_file
    if combined.exists():
        return read_yaml(combined)
    if result_file == "result.yaml":
        frag_dir = run_dir / "results"
        if frag_dir.is_dir():
            results: dict = {}
            for frag in sorted(frag_dir.glob("*.yaml")):
                results.update(read_yaml(frag))
            return results
    return {}


def table(results: dict, metric: str) -> dict:
    """machine -> snr -> mean over model IDs (+ per-machine and overall means)."""
    grid: dict = {m: {s: [] for s in SNRS} for m in MACHINES}
    for entry in results.values():
        m, s = entry.get("machine_type"), entry.get("db")
        if m in grid and s in grid[m] and entry.get(metric) is not None:
            grid[m][s].append(float(entry[metric]))
    out: dict = {}
    all_vals: list[float] = []
    for m in MACHINES:
        row = {}
        present = []
        for s in SNRS:
            vals = grid[m][s]
            if vals:
                row[s] = round(mean(vals), 4)
                present.extend(vals)
        if present:
            row["mean"] = round(mean(present), 4)
            all_vals.extend(present)
            out[m] = row
    if all_vals:
        out["__overall__"] = round(mean(all_vals), 4)
    return out


def render(title: str, tbl: dict) -> str:
    lines = [title, f"  {'machine':8s}" + "".join(f"{s:>9s}" for s in SNRS) + f"{'mean':>9s}"]
    for m in MACHINES:
        if m not in tbl:
            continue
        row = tbl[m]
        cells = "".join(f"{row[s]:>9.4f}" if s in row else f"{'-':>9s}" for s in SNRS)
        lines.append(f"  {m:8s}{cells}{row.get('mean', float('nan')):>9.4f}")
    if "__overall__" in tbl:
        lines.append(f"  {'overall':8s}{'':>27s}{tbl['__overall__']:>9.4f}")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    run_dir = resolve_run_dir(config["output"]["directory"], args.run_dir)
    results = collect_results(run_dir, args.result_file)
    if not results:
        raise SystemExit(f"No eval results ({args.result_file}) found under {run_dir}")

    auc_tbl = table(results, "AUC")
    pauc_tbl = table(results, "pAUC")

    print(f"\nRUN DIR: {run_dir}   ({len(results)} targets)\n")
    print(render("AUC  (mean over model IDs)", auc_tbl))
    print()
    print(render("pAUC (mean over model IDs)", pauc_tbl))
    print()

    # Name the summary after the result file so different scorers don't clobber each
    # other (result.yaml->summary.yaml, result_embedding.yaml->summary_embedding.yaml).
    summary_file = (
        "summary.yaml" if args.result_file == "result.yaml"
        else args.result_file.replace("result", "summary")
    )
    if args.result_file == "result.yaml":
        write_yaml(run_dir / "result.yaml", results)  # consolidate parallel-eval fragments
    write_yaml(run_dir / summary_file, {"AUC": auc_tbl, "pAUC": pauc_tbl, "num_targets": len(results)})
    print(f"wrote {run_dir / summary_file}")


if __name__ == "__main__":
    main()
