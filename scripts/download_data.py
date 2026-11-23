"""Download the raw datasets into data/raw/.

Usage:
    python scripts/download_data.py            # both
    python scripts/download_data.py secom      # SECOM only (UCI, no account needed)
    python scripts/download_data.py wm811k     # WM-811K only (needs a Kaggle API token)

WM-811K needs a Kaggle account: create a token at https://www.kaggle.com/settings
("Create New Token") and save it to ~/.kaggle/kaggle.json.
"""

from __future__ import annotations

import io
import sys
import zipfile
from pathlib import Path

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"

SECOM_URL = "https://archive.ics.uci.edu/static/public/179/secom.zip"
WM811K_KAGGLE = "qingyi/wm811k-wafer-map"


def download_secom() -> None:
    import requests

    if (RAW_DIR / "secom.data").exists():
        print("SECOM already present, skipping.")
        return
    print(f"Downloading SECOM from {SECOM_URL} ...")
    resp = requests.get(SECOM_URL, timeout=120)
    resp.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        for name in zf.namelist():
            if Path(name).name in {"secom.data", "secom_labels.data", "secom.names"}:
                (RAW_DIR / Path(name).name).write_bytes(zf.read(name))
    print("SECOM saved to", RAW_DIR)


def download_wm811k() -> None:
    if (RAW_DIR / "LSWMD.pkl").exists():
        print("WM-811K already present, skipping.")
        return
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError:
        sys.exit("Install the Kaggle client first: pip install kaggle")

    api = KaggleApi()
    api.authenticate()
    print(f"Downloading WM-811K ({WM811K_KAGGLE}, ~2 GB unzipped) ...")
    api.dataset_download_files(WM811K_KAGGLE, path=str(RAW_DIR), unzip=True, quiet=False)

    # The archive may nest the file in a sub-folder; move it to data/raw/LSWMD.pkl.
    found = next(RAW_DIR.rglob("LSWMD.pkl"), None)
    if found is None:
        sys.exit("Download finished but LSWMD.pkl was not found under data/raw/.")
    if found != RAW_DIR / "LSWMD.pkl":
        found.rename(RAW_DIR / "LSWMD.pkl")
    print("WM-811K saved to", RAW_DIR / "LSWMD.pkl")


def main(argv: list[str]) -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    targets = argv or ["secom", "wm811k"]
    for t in targets:
        {"secom": download_secom, "wm811k": download_wm811k}[t]()


if __name__ == "__main__":
    main(sys.argv[1:])
