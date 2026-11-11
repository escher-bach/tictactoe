"""Texts used in the experiments."""
import os
import zipfile

import numpy as np

DATA = os.environ.get("PM_DATA", "/tmp/claude-0/data")


def text8(lo, hi):
    """text8[lo:hi] as sign ids: a..z = 0..25, space = 26."""
    path = os.path.join(DATA, "text8")
    if not os.path.exists(path):
        with zipfile.ZipFile(path + ".zip") as z:
            z.extract("text8", DATA)
    with open(path, "rb") as f:
        f.seek(lo)
        raw = np.frombuffer(f.read(hi - lo), np.uint8)
    return np.where(raw == 32, 26, raw - 97).astype(np.int64)


VALID = (90_000_000, 90_200_000)
TEST = (95_000_000, 95_500_000)
