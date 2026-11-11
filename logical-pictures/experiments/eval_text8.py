"""Picture a held-out text in a learned form.  Train text8[0:N], test text8[95.0M:95.5M]."""
import argparse
import json
import sys
import time

sys.path.insert(0, __file__.rsplit("/", 2)[0])
from pm.data import TEST, text8  # noqa: E402
from pm.machine import fmt_sort, load_form, run_form  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--form", default="runs/form.pkl")
ap.add_argument("--n", type=int, default=1_000_000)
ap.add_argument("--out", default="runs/results.jsonl")
ap.add_argument("--tag", default="")
ap.add_argument("--modes", default="static,dynamic")
ap.add_argument("--test-lo", type=int, default=TEST[0], help="start of the 500k test slice")
a = ap.parse_args()
form = load_form(a.form)
train, test = text8(0, a.n), text8(a.test_lo, a.test_lo + 500_000)
for static in [m == "static" for m in a.modes.split(",")]:
    t0 = time.time()
    print(f"-- N={a.n}  {'static' if static else 'dynamic'}")
    ladder = run_form(form, train, test, static=static, log=print)[0]   # drop the picture: it is large
    rec = dict(model="picture machine", tag=a.tag, test_lo=a.test_lo, form=a.form, ntrain=a.n,
               mode="static" if static else "dynamic", test_bpc=ladder[-1], ladder=ladder,
               sorts=[fmt_sort(s, form["bounds"]) for s in form["sorts"]],
               seconds=time.time() - t0)
    with open(a.out, "a") as f:
        f.write(json.dumps(rec) + "\n")
    print(f"   test {ladder[-1]:.4f} bpc   ({time.time() - t0:.0f}s)")
