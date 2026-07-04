from __future__ import annotations

"""Fetch the pretrained BEATs checkpoint into audio_ssl/pretrained/.

The official links (github.com/microsoft/unilm/tree/master/beats) are OneDrive shares
that now require interactive auth (the api.onedrive.com direct-download trick 401s), so
we default to a HuggingFace mirror of the same file. Verified 2026-07-03: 361 MB,
cfg matches the paper (12 layers, 768-d, patch 16), 90.4M params.
"""

import argparse
import urllib.request
from pathlib import Path

MIRRORS = {
    "BEATs_iter3_plus_AS2M.pt": [
        "https://huggingface.co/datasets/Bencr/beats-checkpoints/resolve/main/BEATs_iter3_plus_AS2M.pt",
    ],
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Download a pretrained BEATs checkpoint.")
    parser.add_argument("--name", default="BEATs_iter3_plus_AS2M.pt", choices=sorted(MIRRORS))
    parser.add_argument("--dest", default="audio_ssl/pretrained")
    args = parser.parse_args()

    dest = Path(args.dest) / args.name
    if dest.exists():
        print(f"{dest} already exists ({dest.stat().st_size / 1e6:.0f} MB); nothing to do.")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    for url in MIRRORS[args.name]:
        try:
            print(f"downloading {url}")
            urllib.request.urlretrieve(url, dest)
            print(f"-> {dest} ({dest.stat().st_size / 1e6:.0f} MB)")
            return
        except Exception as exc:  # try the next mirror
            print(f"   failed: {exc}")
            dest.unlink(missing_ok=True)
    raise SystemExit(
        f"All mirrors failed. Download '{args.name}' manually from the links in "
        f"github.com/microsoft/unilm/tree/master/beats and place it at {dest}")


if __name__ == "__main__":
    main()
