"""The picture machine must not see the future.

Two checks on a learned form:
 1. Causality: change the test text after position T; the description of every
    fact before T must be bit-for-bit unchanged.
 2. Normalisation: at random positions, the probabilities of all signs (products
    along the code tree) sum to 1 -- by construction, verified numerically by
    scoring every alternative sign at that position.
Run: python3 checks/causality.py
"""
import sys

import numpy as np

sys.path.insert(0, __file__.rsplit("/", 2)[0])
from pm.data import text8  # noqa: E402
from pm.machine import Learner, run_form  # noqa: E402

x = text8(0, 130_000)
L = Learner(x, 27, 100_000, threads=2, log=lambda s: None)
form = L.run(max_sorts=14)
print("sorts:", len(form["sorts"]))
train, test = text8(0, 100_000), text8(200_000, 220_000)
for static in (False, True):
    _, st_a, pa = run_form(form, train, test, static=static)
    T = 12_000
    rng = np.random.default_rng(0)
    test_b = test.copy()
    test_b[T:] = rng.integers(0, 27, len(test) - T)
    _, st_b, pb = run_form(form, train, test_b, static=static)
    lo, hi = pa.Mtrain, int(pa.off[len(train) + T])
    same = np.array_equal(st_a[lo:hi], st_b[lo:hi])
    print(f"[{'PASS' if same else 'FAIL'}] causality (static={static}): {hi - lo} facts before the change identical")

    # normalisation: replace one test sign by each alternative and read its probability
    tot = []
    for t in rng.integers(100, T - 1, 12):
        ps = []
        for s in range(27):
            test_c = test.copy()
            test_c[t] = s
            _, st_c, pc = run_form(form, train, test_c, static=static)
            a, b = int(pc.off[len(train) + t]), int(pc.off[len(train) + t + 1])
            z = np.where(pc.fbit[a:b] == 1, st_c[a:b], -st_c[a:b]).astype(np.float64)
            ps.append(float(np.exp(-np.log1p(np.exp(-z)).sum())))
        tot.append(sum(ps))
    ok = max(abs(v - 1) for v in tot) < 1e-4
    print(f"[{'PASS' if ok else 'FAIL'}] normalisation (static={static}): sums over 27 signs in "
          f"[{min(tot):.6f}, {max(tot):.6f}] at 12 positions")
