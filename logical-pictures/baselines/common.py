"""Shared data loading / result logging for the char-LM baselines.

Splits are fixed byte offsets into text8 (100M chars, 27-symbol alphabet):
  train = text8[0:N]
  valid = text8[90_000_000:90_200_000]
  test  = text8[95_000_000:95_500_000]
Symbols are mapped to ids 0..26: space -> 0, 'a'..'z' -> 1..26.
"""
import json
import os

import numpy as np

DATA_PATH = os.environ.get("TEXT8_PATH", "/tmp/claude-0/data/text8")
HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_PATH = os.path.join(HERE, "results.jsonl")

V = 27
VALID_LO, VALID_HI = 90_000_000, 90_200_000
TEST_LO, TEST_HI = 95_000_000, 95_500_000


def _read(lo, hi):
    with open(DATA_PATH, "rb") as f:
        f.seek(lo)
        raw = np.frombuffer(f.read(hi - lo), dtype=np.uint8)
    assert raw.size == hi - lo, "text8 is shorter than expected"
    ids = np.where(raw == 32, 0, raw.astype(np.int16) - 96)
    assert ids.min() >= 0 and ids.max() < V, "unexpected byte in text8"
    return ids.astype(np.uint8)


def load_splits(ntrain):
    """Returns (train, valid, test) as uint8 arrays of symbol ids."""
    assert 0 < ntrain <= VALID_LO, "train must not overlap valid/test"
    return _read(0, ntrain), _read(VALID_LO, VALID_HI), _read(TEST_LO, TEST_HI)


def append_result(rec, path=RESULTS_PATH):
    with open(path, "a") as f:
        f.write(json.dumps(rec) + "\n")
    print("RESULT " + json.dumps(rec), flush=True)
