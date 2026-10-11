"""Download the trained kmrl-copilot model from the project's GitHub release.

    uv run python scripts/get_model.py            # from backend/

The model (~1 GB) is too large for the git repository itself, so it is published as a
release asset. Run once; afterwards the app works offline. Set KMRL_MODEL_URL to use a
different source (a URL or a local .zip path).
"""
from __future__ import annotations

import hashlib
import os
import shutil
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1]
DEST = BACKEND / "models" / "kmrl-copilot"
URL = os.getenv("KMRL_MODEL_URL", "https://github.com/MeezaUni/METECH-Local/releases/latest/download/kmrl-copilot.zip")


def download(url: str, to: Path) -> None:
    req = urllib.request.Request(url, headers={"User-Agent": "metech-local"})
    with urllib.request.urlopen(req, timeout=60) as r, open(to, "wb") as f:
        total = int(r.headers.get("Content-Length") or 0)
        done = 0
        while chunk := r.read(1 << 20):
            f.write(chunk)
            done += len(chunk)
            if total:
                print(f"\r  {done / 2**20:,.0f} / {total / 2**20:,.0f} MB ({done / total:.0%})", end="", flush=True)
    print()


def main() -> int:
    if (DEST / "config.json").exists():
        print(f"Model already present at {DEST}")
        return 0
    print(f"Downloading the trained model from {URL}")
    with tempfile.TemporaryDirectory() as tmp:
        archive = Path(tmp) / "kmrl-copilot.zip"
        try:
            if Path(URL).is_file():
                shutil.copy(URL, archive)
            else:
                download(URL, archive)
            with zipfile.ZipFile(archive) as z:
                bad = z.testzip()
                if bad:
                    raise zipfile.BadZipFile(f"corrupt member {bad}")
                staging = Path(tmp) / "unpacked"
                z.extractall(staging)
        except Exception as exc:   # network, 404 (no release yet), corrupt archive
            print(f"Could not get the model: {exc}", file=sys.stderr)
            print("The app still runs with its rule engine. To train the model yourself, run train.bat "
                  "(about 15 minutes on an NVIDIA GPU).", file=sys.stderr)
            return 1
        # Accept archives with or without a top-level kmrl-copilot/ folder.
        root = next((p.parent for p in staging.rglob("config.json")), None)
        if root is None:
            print("The downloaded archive does not contain a model (config.json missing).", file=sys.stderr)
            return 1
        DEST.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(root, DEST)
    digest = hashlib.sha256((DEST / "model.safetensors").read_bytes()).hexdigest()[:12] if (DEST / "model.safetensors").exists() else "?"
    print(f"Model ready at {DEST} (weights sha256 {digest}...)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
