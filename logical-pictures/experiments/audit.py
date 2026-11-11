"""The collapse audit (ARCHITECTURE 6): which row of THEORY 3.3 does each admitted sort
belong to, and how much of the residue did each row remove?"""
import argparse
import json
import sys

sys.path.insert(0, __file__.rsplit("/", 2)[0])
from pm.machine import fmt_sort, load_form  # noqa: E402

ROWS = ["prior", "literal window", "class window", "literal run", "plain cache",
        "gated repeat", "key over constructed objects"]
CLASSICAL = {"literal window": "n-gram", "class window": "class n-gram", "literal run": "n-gram / PPM with a learned boundary",
             "plain cache": "PPM / LZ / cache"}


def constructed(feats):
    """Does a key mention a recency object, a learned class of objects, or sameness?"""
    return any(f[0] in ("RF", "RF2", "RA", "CLS", "XC", "EQ") for f in feats)


def row(sort):
    key, ptr = sort
    kinds = {f[0] for f in key}
    if not key and ptr is None:
        return "prior"
    if ptr is None:
        if constructed(key):
            return "key over constructed objects"
        if kinds <= {"S"}:
            return "literal window"
        if kinds <= {"S", "C"}:
            return "class window"
        return "literal run"
    # a plain cache repeats the last occurrence of a literal key, with no condition on it
    literal_target = ptr[0] == "S" or (ptr[0] == "RF" and not constructed(ptr[1]) and "C" not in {f[0] for f in ptr[1]})
    if literal_target and not key:
        return "plain cache"
    if literal_target and not constructed(key) and "C" not in kinds:
        return "plain cache"
    return "gated repeat" if literal_target else "key over constructed objects"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--form", default="runs/form_fine.pkl")
    ap.add_argument("--results", default="runs/results.jsonl")
    a = ap.parse_args()
    form = load_form(a.form)
    out = []
    for line in open(a.results):
        r = json.loads(line)
        if r["form"] != a.form:
            continue
        lad = [r["ladder"][0] + 0] + r["ladder"]
        drops = {k: 0.0 for k in ROWS}
        for i, s in enumerate(form["sorts"]):
            if i:
                drops[row(s)] += lad[i] - lad[i + 1]
        tot = sum(drops.values())
        print(f"N={r['ntrain']:>9}  {r['mode']:7s}  from {r['ladder'][0]:.3f} to {r['ladder'][-1]:.4f} bpc:  " +
              "  ".join(f"{k}: {v:.3f} ({v / tot:.0%})" for k, v in drops.items() if k != "prior"))
        out.append(dict(ntrain=r["ntrain"], mode=r["mode"], drops=drops, first=r["ladder"][0], last=r["ladder"][-1]))
    print()
    for s in form["sorts"]:
        print(f"  {row(s):18s} {fmt_sort(s, form['bounds'])}")
    json.dump(out, open("runs/audit.json", "w"), indent=1)
