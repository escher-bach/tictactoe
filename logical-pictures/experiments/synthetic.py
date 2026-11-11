"""Does the rule find the non-classical objects where they are the whole story?"""
import json
import sys

import numpy as np

sys.path.insert(0, __file__.rsplit("/", 2)[0])
from pm import kernels as K  # noqa: E402
from pm.machine import Learner, fmt_sort, run_form  # noqa: E402
from pm.worlds import analogy, respelled  # noqa: E402


def cost_at(st, pic, lo_pos, mask):
    """Mean bits per masked position of the test text."""
    z = np.where(pic.fbit == 1, st, -st).astype(np.float64)
    fb = np.log1p(np.exp(-z)) / np.log(2)
    per_pos = np.add.reduceat(fb, pic.off[:-1])
    test = per_pos[lo_pos:]
    return float(test[mask].mean()), float(test.mean())


def report(name, form, train, test, mask, results):
    print(f"== {name}: the form")
    for s in form["sorts"]:
        print("   ", fmt_sort(s))
    for static in (False, True):
        ladder, st, pic = run_form(form, train, test, static=static)
        at, overall = cost_at(st, pic, len(train), mask)
        mode = "static" if static else "dynamic"
        print(f"   {mode:7s}: {at:.3f} bits at the determined positions, {overall:.3f} bits/sign overall")
        results.append(dict(world=name, model="picture machine", mode=mode, bits_at_determined=at,
                            bits_per_sign=overall))


results = []

# -- respelled recall ------------------------------------------------------
x, ans, V = respelled(26_000, 1)
ntr, nva = 9 * 20_000, 9 * 3_000
L = Learner(x[:ntr + nva], V, ntr, threads=1, min_gain=0.002, log=lambda s: None)
form = L.run(max_sorts=12)
print('bounds', L.bounds)
report("respelled recall", form, x[:ntr], x[ntr + nva:], ans[ntr + nva:], results)
np.save("runs/respelled.npy", x)
np.save("runs/respelled_mask.npy", ans)

# -- analogy: every text brings its own names -------------------------------------
# Records are counted on signs 1..200.  Identities are admitted on a slice written in
# signs 201..300, and tested on signs 301..400: in this world a new text has new names.
V = 401
xtr, _ = analogy(24_000, 2, np.arange(1, 201))
xva, _ = analogy(3_000, 3, np.arange(201, 301))
x_in, m_in = analogy(3_000, 4, np.arange(1, 201))
x_out, m_out = analogy(3_000, 5, np.arange(301, 401))
L = Learner(np.concatenate([xtr, xva]), V, len(xtr), threads=1, min_gain=0.002, log=lambda s: None)
form = L.run(max_sorts=12)
report("analogy, signs of the corpus", form, xtr, x_in, m_in, results)
report("analogy, signs never seen", form, xtr, x_out, m_out, results)
np.save("runs/analogy_train.npy", xtr)
np.save("runs/analogy_valid.npy", xva)
np.save("runs/analogy_in.npy", x_in)
np.save("runs/analogy_out.npy", x_out)

with open("runs/synthetic_pm.json", "w") as f:
    json.dump(results, f, indent=1)
