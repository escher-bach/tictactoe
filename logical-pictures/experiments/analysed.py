"""How analysed is the learned picture?  (SOLUTIONS 9.3)

In a fully analysed form a two-valued fact is either repeated (probability near 1 for
what occurs) or new and uniform (probability 1/2).  This prints, for the held-out
facts, the distribution of the probability given to the fact that occurred."""
import argparse
import json
import sys

import numpy as np

sys.path.insert(0, __file__.rsplit("/", 2)[0])
from pm.data import TEST, text8  # noqa: E402
from pm.machine import load_form, run_form  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--form", default="runs/form_all.pkl")
ap.add_argument("--n", type=int, default=1_000_000)
a = ap.parse_args()
form = load_form(a.form)
_, st, pic = run_form(form, text8(0, a.n), text8(*TEST), static=True)
z = np.where(pic.fbit == 1, st, -st)[pic.Mtrain:].astype(np.float64)
p = 1 / (1 + np.exp(-z))
bits = -np.log2(p)
edges = [0, 0.05, 0.2, 0.4, 0.6, 0.8, 0.95, 0.99, 1.0000001]
rows = []
print(f"facts per sign: {len(p) / (pic.n - pic.ntrain):.2f}   bits per sign: {bits.sum() / (pic.n - pic.ntrain):.4f}")
print("probability given to what occurred | share of facts | share of the bits")
for lo, hi in zip(edges[:-1], edges[1:]):
    m = (p >= lo) & (p < hi)
    rows.append(dict(lo=lo, hi=min(hi, 1.0), facts=float(m.mean()), bits=float(bits[m].sum() / bits.sum())))
    print(f"  [{lo:.2f}, {min(hi, 1):.2f})   {m.mean():7.1%}   {bits[m].sum() / bits.sum():7.1%}")
json.dump(rows, open("runs/analysed.json", "w"), indent=1)
