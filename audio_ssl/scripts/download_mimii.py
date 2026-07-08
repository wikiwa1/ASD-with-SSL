from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import urllib.request
from pathlib import Path


ZENODO_API = "https://zenodo.org/api/records/3384388"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download and extract the MIMII Zenodo dataset.")
    parser.add_argument("--output", default="dataset", help="Dataset output directory.")
    parser.add_argument(
        "--machines",
        nargs="*",
        default=["fan", "pump", "slider", "valve"],
        choices=["fan", "pump", "slider", "valve"],
        help="Machine archives to download.",
    )
    parser.add_argument(
        "--snr",
        nargs="*",
        default=["-6_dB", "0_dB", "6_dB"],
        choices=["-6_dB", "0_dB", "6_dB"],
        help="SNR archives to download.",
    )
    parser.add_argument("--no-extract", action="store_true", help="Only download zip archives.")
    return parser.parse_args()


def fetch_record() -> dict:
    with urllib.request.urlopen(ZENODO_API, timeout=60) as response:
        return json.load(response)


def wanted_files(record: dict, machines: set[str], snrs: set[str]) -> list[dict]:
    selected = []
    for file_info in record["files"]:
        key = file_info["key"]
        if not key.endswith(".zip"):
            continue
        stem = key.removesuffix(".zip")
        snr, machine = stem.rsplit("_", 1)
        if snr in snrs and machine in machines:
            selected.append(file_info)
    return sorted(selected, key=lambda item: item["key"])


def run(cmd: list[str]) -> None:
    print(" ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def download(file_info: dict, archive_dir: Path) -> Path:
    archive_dir.mkdir(parents=True, exist_ok=True)
    target = archive_dir / file_info["key"]
    url = file_info["links"]["self"]
    if target.exists() and target.stat().st_size == file_info["size"]:
        print(f"already downloaded {target}", flush=True)
        return target
    run(["wget", "-nv", "-c", "-O", str(target), url])
    return target


def archive_snr(archive: Path) -> str:
    stem = archive.name.removesuffix(".zip")
    snr, _machine = stem.rsplit("_", 1)
    return snr


def extract(archive: Path, output_dir: Path) -> None:
    extract_dir = output_dir / archive_snr(archive)
    if shutil.which("unzip"):
        run(["unzip", "-q", "-n", str(archive), "-d", str(extract_dir)])
    elif shutil.which("7z"):
        run(["7z", "x", "-aos", str(archive), f"-o{extract_dir}"])
    else:
        raise RuntimeError("Need either unzip or 7z to extract MIMII archives")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output)
    archive_dir = output_dir / "archives"
    record = fetch_record()
    files = wanted_files(record, machines=set(args.machines), snrs=set(args.snr))
    if not files:
        raise RuntimeError("No matching MIMII archives found")

    total = sum(file_info["size"] for file_info in files)
    print(f"selected {len(files)} archives, {total / 1024**3:.1f} GiB zipped", flush=True)
    for file_info in files:
        archive = download(file_info, archive_dir)
        if not args.no_extract:
            extract(archive, output_dir)


if __name__ == "__main__":
    main()
