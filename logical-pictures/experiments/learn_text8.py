"""Learn a form on text8.  Records are counted on text8[0:N]; identities are admitted
on their effect on a distant slice (text8[90.0M:90.2M]), so nothing is admitted for
fitting the text it was counted on."""
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
ap.add_argument("--use", default="S,C,W,R,EQ,CLS,PTR")
ap.add_argument("--code-eps", type=float, default=0.1)
ap.add_argument("--max-key", type=int, default=6)
ap.add_argument("--out", default="runs/form.pkl")
a = ap.parse_args()

x = np.concatenate([text8(0, a.n), text8(*VALID)])
t0 = time.time()
L = Learner(x, 27, a.n, threads=a.threads, min_gain=a.min_gain, use=tuple(a.use.split(",")),
            code_eps=a.code_eps, max_key=a.max_key, log=lambda s: print(s, flush=True))
form = L.run(max_sorts=a.max_sorts)
form["search_seconds"] = time.time() - t0
form["args"] = vars(a)
save_form(form, a.out)
print(f"done: {len(form['sorts'])} sorts, {time.time() - t0:.0f}s, valid {form['history'][-1]:.4f} bpc")
