"""Check that your processed WM-811K dataset matches the one behind the reported results.

    python scripts/prepare_wm811k.py   # build it from the Kaggle download
    python scripts/verify_dataset.py   # compare with reports/data_checksums.json

The processed file itself is not in the repository: WM-811K is redistributed on Kaggle
without a clear licence for re-sharing, so anyone can rebuild it from the original download
and prove it is identical instead. Checksums are taken of the array *contents*, not of the
.npz file, so they do not depend on zip timestamps or compression settings.

    python scripts/verify_dataset.py --write   # (maintainer) refresh the reference checksums
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

from wafer.preprocess import DATASET_PATH, load_dataset

REFERENCE = Path(__file__).resolve().parents[1] / "reports" / "data_checksums.json"


def array_sha256(a: np.ndarray) -> str:
    h = hashlib.sha256()
    if a.dtype == object:
        h.update("\n".join(map(str, a.ravel())).encode("utf-8"))
    else:
        h.update(np.ascontiguousarray(a).tobytes())
    h.update(f"|{a.dtype.str if a.dtype != object else 'object'}|{a.shape}".encode())
    return h.hexdigest()


def checksums(ds: dict) -> dict:
    return {
        k: {"shape": list(v.shape), "dtype": str(v.dtype), "sha256": array_sha256(v)}
        for k, v in sorted(ds.items())
    }


def main() -> int:
    if not DATASET_PATH.exists():
        print(f"{DATASET_PATH} not found: run scripts/prepare_wm811k.py first")
        return 1
    ds = load_dataset()
    current = checksums(ds)
    if "--write" in sys.argv:
        REFERENCE.write_text(json.dumps(current, indent=2) + "\n")
        print("wrote", REFERENCE)
        return 0

    reference = json.loads(REFERENCE.read_text())
    ok = True
    for name, ref in reference.items():
        got = current.get(name)
        match = got is not None and got["sha256"] == ref["sha256"]
        ok &= match
        print(f"{'OK  ' if match else 'DIFF'} {name:11s} {ref['dtype']:7s} {tuple(ref['shape'])}")
    print(
        "\nDataset matches the one used for the reported results."
        if ok
        else "\nDataset differs: results may not reproduce exactly."
    )
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
