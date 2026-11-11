"""Learn a form with the second level of the tower: objects, their kinds, and facts about them."""
import argparse
import sys
import time

import numpy as np

sys.path.insert(0, __file__.rsplit("/", 2)[0])
from pm.data import VALID, text8  # noqa: E402
from pm.machine import Learner, save_form  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=1_000_000)
ap.add_argument("--max-sorts", type=int, default=40)
ap.add_argument("--min-gain", type=float, default=0.001)
ap.add_argument("--threads", type=int, default=3)
ap.add_argument("--rounds", type=int, default=2)
ap.add_argument("--use", default="S,C,W,R,EQ,PTR")
ap.add_argument("--out", default="runs/form_tower.pkl")
a = ap.parse_args()
x = np.concatenate([text8(0, a.n), text8(*VALID)])
log = lambda s: print(s, flush=True)
t0 = time.time()
# the boundary of objects is the one the rule itself chooses at the level of signs
L0 = Learner(x, 27, a.n, threads=a.threads, use=("S", "W"), log=lambda s: None)
L0.run(max_sorts=2)
first = L0.sorts[1][0][0]
assert first[0] == "W", "the first identity admitted is not a run; no objects to build on"
boundary = sorted(L0.bounds[first[1]])
log(f"objects are runs bounded by signs {boundary}")
L = Learner(x, 27, a.n, threads=a.threads, min_gain=a.min_gain, use=tuple(a.use.split(",")), log=log,
            tower=dict(boundary=boundary, rounds=a.rounds))
form = L.run(max_sorts=a.max_sorts)
form["search_seconds"] = time.time() - t0
save_form(form, a.out)
log(f"done: {len(form['sorts'])} sorts, {time.time() - t0:.0f}s, valid {form['history'][-1]:.4f} bpc")
