"""With facts about objects, the picture states an object's kind before its signs.
That is only a code if a reader who has the text up to a point can compute everything
the machine uses at that point.  Two checks on a form learned with the tower:

 1. Object facts.  Replace the text from the start of an object onward.  Every fact
    before it must be unchanged, and so must the description of the object's own
    facts up to and including the first one whose answer differs.
 2. Signs inside an object.  Replace an object by another object of the same class that
    shares a prefix with it.  The description of every fact up to and including the
    first sign that differs must be unchanged.
Run: python3 checks/decodable.py runs/form_tower.pkl
"""
import sys

import numpy as np

sys.path.insert(0, __file__.rsplit("/", 2)[0])
from pm.data import text8  # noqa: E402
from pm.machine import load_form, run_form  # noqa: E402

form = load_form(sys.argv[1] if len(sys.argv) > 1 else "runs/form_tower.pkl")
assert form.get("tower"), "this check is for forms with object facts"
train, test = text8(0, 200_000), text8(300_000, 330_000)
rng = np.random.default_rng(1)
for static in (False, True):
    _, st_a, pa = run_form(form, train, test, static=static)
    tw = pa.tower
    n0 = len(train)
    starts = tw.start[(tw.start > n0 + 2000) & (tw.start < n0 + 25000)]

    # 1. object facts
    bad = n = 0
    for s in rng.choice(starts, 6, replace=False):
        tb = test.copy()
        tb[s - n0:] = text8(5_000_000 + int(s), 5_000_000 + int(s) + len(test) - (s - n0))
        _, st_b, pb = run_form(form, train, tb, static=static)
        a0, b0 = int(pa.off[s]), int(pb.off[s])
        assert a0 == b0
        ok = np.array_equal(st_a[pa.Mtrain:a0], st_b[pb.Mtrain:b0])
        j = 0                                            # shared part of the object's path
        while pa.fobj[a0 + j] and pb.fobj[b0 + j] and pa.fnode[a0 + j] == pb.fnode[b0 + j]:
            ok &= st_a[a0 + j] == st_b[b0 + j]
            if pa.fbit[a0 + j] != pb.fbit[b0 + j]:
                break
            j += 1
        n += 1
        bad += not ok
    print(f"[{'PASS' if not bad else 'FAIL'}] object facts (static={static}): {n} objects replaced, "
          f"{bad} with a description that depended on the object itself")

    # 2. signs inside an object: swap in another object of the same class with a shared prefix
    bad = n = 0
    toks = np.flatnonzero((tw.start > n0 + 2000) & (tw.start < n0 + 25000))
    rng.shuffle(toks)
    for k in toks:
        if n >= 6:
            break
        ty = tw.type_of[k]
        w = test[tw.start[k] - n0:tw.end[k] - n0]
        mates = np.flatnonzero(tw.leaf == tw.leaf[ty])
        cand = None
        for m in mates[:400]:
            if m == ty:
                continue
            r = tw.rep[m]
            v = tw.x[tw.start[r]:tw.end[r]]
            c = 0
            while c < min(len(v), len(w)) and v[c] == w[c]:
                c += 1
            if c >= 1 and c < min(len(v), len(w)):
                cand = (v, c)
                break
        if cand is None:
            continue
        v, c = cand
        s = int(tw.start[k]) - n0
        tb = np.concatenate([test[:s], v, test[s + len(w):]])
        _, st_b, pb = run_form(form, train, tb, static=static)
        if pb.tower.leaf[pb.tower.type_of[np.searchsorted(pb.tower.start, n0 + s)]] != tw.leaf[ty] and \
                not np.array_equal(pb.tower.pbit[pb.tower.type_of[np.searchsorted(pb.tower.start, n0 + s)]], tw.pbit[ty]):
            continue                                     # not the same class after all
        div = n0 + s + c                                 # first position whose sign differs
        a1, b1 = int(pa.off[div]), int(pb.off[div])
        ok = a1 == b1 and np.array_equal(st_a[pa.Mtrain:a1], st_b[pb.Mtrain:b1])
        j = 0
        while a1 + j < pa.off[div + 1] and b1 + j < pb.off[div + 1] and pa.fnode[a1 + j] == pb.fnode[b1 + j]:
            ok &= st_a[a1 + j] == st_b[b1 + j]
            if pa.fbit[a1 + j] != pb.fbit[b1 + j]:
                break
            j += 1
        n += 1
        bad += not ok
    print(f"[{'PASS' if not bad else 'FAIL'}] signs inside an object (static={static}): {n} objects swapped for a "
          f"classmate with a shared prefix, {bad} with a description that depended on later signs")
