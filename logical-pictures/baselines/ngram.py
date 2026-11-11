#!/usr/bin/env python3
"""Interpolated modified Kneser-Ney character n-gram LM (Chen & Goodman 1998).

Static model: counts come from train only, nothing is updated on valid/test.

For a model of maximum order n (n-gram = n-1 context chars + 1 predicted char):

  level n (highest) uses raw counts c(u w);
  levels k < n use continuation counts  N1+(. u w) = #distinct left extensions;
  p_k(w|u) = (c_k(u w) - D_k(c_k(u w))) / c_k(u .)
             + [D1 N1(u.) + D2 N2(u.) + D3+ N3+(u.)] / c_k(u .) * p_{k-1}(w|u')
  p_0(w)   = 1/27                      (uniform floor below the unigram level)
  if c_k(u .) == 0 (unseen context):   p_k(w|u) = p_{k-1}(w|u')

D_k(1), D_k(2), D_k(3+) are the closed-form Chen-Goodman estimates from the
count-of-counts n1..n4 of the level's own (raw or continuation) count table.
If the closed form is degenerate (some n_i == 0, or a discount falls outside
[0, i]) -- which happens for the saturated lowest levels of a 27-symbol
alphabet -- the level falls back to D = (0.5, 1.0, 1.5) (same convention as
KenLM's --discount_fallback).  Which levels fell back is recorded in the output.

Usage: python3 ngram.py --ntrain 1000000
       python3 ngram.py --selftest      (compares against a slow dict reference)
"""
import os

os.environ.setdefault("NUMBA_NUM_THREADS", "2")
os.environ.setdefault("OMP_NUM_THREADS", "2")

import argparse
import time

import numba as nb
import numpy as np

from common import V, append_result, load_splits

ORDERS = (3, 4, 5, 6, 7, 8)
MAXN = max(ORDERS)


# ----------------------------------------------------------------------------
# counting (numpy, sort based)
# ----------------------------------------------------------------------------
def gram_codes(x, k):
    """Base-27 big-endian code of every k-gram x[t-k+1..t], t = k-1..len-1."""
    n = x.size - k + 1
    code = np.zeros(n, dtype=np.int64)
    for j in range(k):
        code = code * V + x[j:j + n]
    return code


def discounts(counts):
    """Chen-Goodman closed-form modified-KN discounts from count-of-counts."""
    n = [int(np.count_nonzero(counts == i)) for i in (1, 2, 3, 4)]
    fallback = False
    D = None
    if min(n) > 0:
        Y = n[0] / (n[0] + 2.0 * n[1])
        D = (1 - 2 * Y * n[1] / n[0], 2 - 3 * Y * n[2] / n[1], 3 - 4 * Y * n[3] / n[2])
        if not all(0.0 <= D[i] <= i + 1 for i in range(3)):
            D = None
    if D is None:
        fallback = True
        D = (0.5, 1.0, 1.5)
    return np.array(D), n, fallback


def build_table(grams, counts):
    """grams: sorted unique k-gram codes; counts: their (raw or continuation) counts.

    Returns a dict of flat arrays describing the level:
      ctx_keys (sorted unique context codes), ctx_start (CSR offsets into grams),
      ctx_total, ctx_gam (= D1 N1 + D2 N2 + D3 N3+), gsym, gdisc (= c - D(c)).
    """
    D, n14, fb = discounts(counts)
    ctx = grams // V
    ctx_keys, ctx_first = np.unique(ctx, return_index=True)
    ctx_start = np.append(ctx_first, grams.size).astype(np.int64)
    d = np.where(counts == 1, D[0], np.where(counts == 2, D[1], D[2]))
    ctx_total = np.add.reduceat(counts.astype(np.float64), ctx_first)
    ctx_gam = np.add.reduceat(d, ctx_first)
    return dict(
        ctx_keys=ctx_keys.astype(np.int64), ctx_start=ctx_start,
        ctx_total=ctx_total, ctx_gam=ctx_gam,
        gsym=(grams % V).astype(np.uint8), gdisc=counts - d,
        D=D.tolist(), n14=n14, fallback=fb, ngrams=int(grams.size),
    )


def train(x, orders=ORDERS):
    """Returns (cont, raw): cont[k] for k=1..max(orders)-1, raw[k] for k in orders."""
    cont, raw = {}, {}
    uniq = {}
    maxn = max(orders)
    for k in range(1, maxn + 1):
        g, c = np.unique(gram_codes(x, k), return_counts=True)
        uniq[k] = g
        if k in orders:
            raw[k] = build_table(g, c.astype(np.float64))
    for k in range(1, maxn):
        # continuation count of a k-gram = #distinct (k+1)-grams that end in it
        g, c = np.unique(uniq[k + 1] % (V ** k), return_counts=True)
        cont[k] = build_table(g, c.astype(np.float64))
    return cont, raw


def flatten(tables):
    """Concatenate a list of level tables into flat arrays + offsets for numba."""
    keys = np.concatenate([t["ctx_keys"] for t in tables])
    tot = np.concatenate([t["ctx_total"] for t in tables])
    gam = np.concatenate([t["ctx_gam"] for t in tables])
    gsym = np.concatenate([t["gsym"] for t in tables])
    gdisc = np.concatenate([t["gdisc"] for t in tables])
    coff = np.zeros(len(tables) + 1, dtype=np.int64)
    goff = np.zeros(len(tables) + 1, dtype=np.int64)
    starts = []
    for i, t in enumerate(tables):
        coff[i + 1] = coff[i] + t["ctx_keys"].size
        goff[i + 1] = goff[i] + t["gsym"].size
        # one CSR row-pointer array per level, made global; drop the trailing
        # sentinel of every level but keep a global one at the very end
        starts.append(t["ctx_start"][:-1] + goff[i])
    starts.append(np.array([goff[-1]], dtype=np.int64))
    # NOTE: the end of the last context of level i is the start of the first
    # context of level i+1 (== goff[i+1]), so a single global CSR works.
    return keys, np.concatenate(starts), tot, gam, gsym, gdisc, coff


# ----------------------------------------------------------------------------
# evaluation (numba)
# ----------------------------------------------------------------------------
@nb.njit(cache=True)
def _find(keys, lo, hi, key):
    while lo < hi:
        mid = (lo + hi) >> 1
        if keys[mid] < key:
            lo = mid + 1
        else:
            hi = mid
    if lo < keys.size and keys[lo] == key:
        return lo
    return -1


@nb.njit(cache=True)
def _level(keys, start, tot, gam, gsym, gdisc, coff, lvl, ctx, lower, out):
    """out <- p_level(.|ctx) given lower = p_{level-1}(.|ctx')."""
    i = _find(keys, coff[lvl], coff[lvl + 1], ctx)
    if i < 0 or i >= coff[lvl + 1]:
        for s in range(V):
            out[s] = lower[s]
        return
    T = tot[i]
    g = gam[i] / T
    for s in range(V):
        out[s] = g * lower[s]
    for j in range(start[i], start[i + 1]):
        out[gsym[j]] += gdisc[j] / T


@nb.njit(cache=True)
def dists_at(x, t, ckeys, cstart, ctot, cgam, cgsym, cgdisc, ccoff,
             rkeys, rstart, rtot, rgam, rgsym, rgdisc, rcoff, orders, P, pc):
    """P[i] <- p(.| x[<t]) under the model of max order orders[i].

    pc is scratch of shape (MAXN, 27): pc[k] = continuation-chain distribution
    of level k (pc[0] = uniform).  Only context inside x is used: at t < n-1 the
    order-n model answers with the continuation chain of level t+1.
    """
    maxn = pc.shape[0]
    for s in range(V):
        pc[0, s] = 1.0 / V
    top = 0  # highest continuation level computed
    for k in range(1, maxn):
        if t < k - 1:
            break
        ctx = 0
        for j in range(t - k + 1, t):
            ctx = ctx * V + x[j]
        _level(ckeys, cstart, ctot, cgam, cgsym, cgdisc, ccoff, k - 1, ctx, pc[k - 1], pc[k])
        top = k
    for i in range(orders.size):
        n = orders[i]
        if t < n - 1:
            for s in range(V):
                P[i, s] = pc[top, s]
        else:
            ctx = 0
            for j in range(t - n + 1, t):
                ctx = ctx * V + x[j]
            _level(rkeys, rstart, rtot, rgam, rgsym, rgdisc, rcoff, i, ctx, pc[n - 1], P[i])


@nb.njit(cache=True)
def evaluate(x, ckeys, cstart, ctot, cgam, cgsym, cgdisc, ccoff,
             rkeys, rstart, rtot, rgam, rgsym, rgdisc, rcoff, orders, maxn):
    """Returns (sum of -log2 p per order, max |sum_w p - 1|, min p seen)."""
    P = np.zeros((orders.size, V))
    pc = np.zeros((maxn, V))
    nll = np.zeros(orders.size)
    maxdev = 0.0
    minp = 1.0
    for t in range(x.size):
        dists_at(x, t, ckeys, cstart, ctot, cgam, cgsym, cgdisc, ccoff,
                 rkeys, rstart, rtot, rgam, rgsym, rgdisc, rcoff, orders, P, pc)
        for i in range(orders.size):
            tot = 0.0
            for s in range(V):
                tot += P[i, s]
                if P[i, s] < minp:
                    minp = P[i, s]
            dev = abs(tot - 1.0)
            if dev > maxdev:
                maxdev = dev
            nll[i] -= np.log2(P[i, x[t]])
    return nll, maxdev, minp


class Model:
    def __init__(self, x, orders=ORDERS):
        self.orders = np.array(orders, dtype=np.int64)
        self.maxn = int(max(orders))
        self.cont, self.raw = train(x, tuple(orders))
        self.c = flatten([self.cont[k] for k in range(1, self.maxn)])
        self.r = flatten([self.raw[k] for k in orders])

    def evaluate(self, x):
        nll, maxdev, minp = evaluate(x, *self.c, *self.r, self.orders, self.maxn)
        return nll / x.size, maxdev, minp

    def dists(self, x, t):
        P = np.zeros((self.orders.size, V))
        pc = np.zeros((self.maxn, V))
        dists_at(x, t, *self.c, *self.r, self.orders, P, pc)
        return P


# ----------------------------------------------------------------------------
# slow, independent reference (dicts over python strings) for --selftest
# ----------------------------------------------------------------------------
def reference_dist(train_s, ctx_s, n):
    """p(.|ctx_s) for max order n, straight from the textbook formulas."""
    from collections import Counter, defaultdict
    alpha = " abcdefghijklmnopqrstuvwxyz"
    raw = Counter(train_s[i:i + n] for i in range(len(train_s) - n + 1))
    tables = {n: raw}
    for k in range(n - 1, 0, -1):
        left = defaultdict(set)
        for i in range(len(train_s) - k):          # (k+1)-gram at i
            left[train_s[i + 1:i + 1 + k]].add(train_s[i])
        tables[k] = {g: len(s) for g, s in left.items()}

    def disc(tab):
        cc = Counter(tab.values())
        n1, n2, n3, n4 = cc[1], cc[2], cc[3], cc[4]
        if min(n1, n2, n3, n4) > 0:
            Y = n1 / (n1 + 2 * n2)
            D = (1 - 2 * Y * n2 / n1, 2 - 3 * Y * n3 / n2, 3 - 4 * Y * n4 / n3)
            if all(0 <= D[i] <= i + 1 for i in range(3)):
                return D
        return (0.5, 1.0, 1.5)

    p = {w: 1.0 / 27 for w in alpha}
    for k in range(1, n + 1):
        tab, D = tables[k], disc(tables[k])
        u = ctx_s[len(ctx_s) - (k - 1):] if k > 1 else ""
        cs = {w: tab.get(u + w, 0) for w in alpha}
        T = sum(cs.values())
        if T == 0:
            continue
        gam = sum(D[min(c, 3) - 1] for c in cs.values() if c > 0)
        p = {w: (max(cs[w] - (D[min(cs[w], 3) - 1] if cs[w] else 0), 0) + gam * p[w]) / T
             for w in alpha}
    return np.array([p[w] for w in alpha])


def selftest():
    tr, va, te = load_splits(40_000)
    orders = (3, 4, 5, 6)
    m = Model(tr, orders)
    alpha = " abcdefghijklmnopqrstuvwxyz"
    tr_s = "".join(alpha[i] for i in tr)
    rng = np.random.default_rng(0)
    worst = 0.0
    refs = {}
    for t in rng.integers(8, te.size, size=25):
        ctx_s = "".join(alpha[i] for i in te[t - 7:t])
        P = m.dists(te, int(t))
        for i, n in enumerate(orders):
            ref = reference_dist(tr_s, ctx_s, n)
            worst = max(worst, float(np.abs(P[i] - ref).max()))
            assert abs(ref.sum() - 1) < 1e-9
    print(f"selftest: max |p_numba - p_reference| over 25 positions x {len(orders)} orders = {worst:.3e}")
    assert worst < 1e-10
    print("selftest OK")


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ntrain", type=int, default=1_000_000)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--no-log", action="store_true", help="do not append to results.jsonl")
    args = ap.parse_args()
    if args.selftest:
        selftest()
        return

    tr, va, te = load_splits(args.ntrain)
    t0 = time.time()
    m = Model(tr)
    train_s = time.time() - t0

    # --- probabilities sum to 1: (a) uniformly random contexts (mostly unseen
    # at high order -> exercises the back-off path), (b) every scored position.
    rng = np.random.default_rng(123)
    rnd = rng.integers(0, V, size=(2000, MAXN)).astype(np.uint8)
    worst = 0.0
    for row in rnd:
        P = m.dists(row, MAXN - 1)
        assert (P > 0).all()
        worst = max(worst, float(np.abs(P.sum(1) - 1).max()))
    assert worst < 1e-9, worst
    print(f"sum-to-1 on 2000 random contexts: max dev {worst:.2e}")

    t0 = time.time()
    vb, vdev, vmin = m.evaluate(va)
    tb, tdev, tmin = m.evaluate(te)
    eval_s = time.time() - t0
    assert max(vdev, tdev) < 1e-9, (vdev, tdev)
    assert min(vmin, tmin) > 0
    print(f"sum-to-1 at every valid/test position, all orders: max dev {max(vdev, tdev):.2e}")

    best = int(np.argmin(vb))
    per_order = {str(n): {"valid_bpc": round(float(vb[i]), 5), "test_bpc": round(float(tb[i]), 5)}
                 for i, n in enumerate(ORDERS)}
    for n in ORDERS:
        print(f"order {n}: valid {per_order[str(n)]['valid_bpc']:.4f}  test {per_order[str(n)]['test_bpc']:.4f}")
    levels = {}
    for k in range(1, MAXN):
        t = m.cont[k]
        levels[f"cont{k}"] = {"D": [round(d, 4) for d in t["D"]], "fallback": t["fallback"], "ngrams": t["ngrams"]}
    for k in ORDERS:
        t = m.raw[k]
        levels[f"raw{k}"] = {"D": [round(d, 4) for d in t["D"]], "fallback": t["fallback"], "ngrams": t["ngrams"]}
    rec = {
        "model": "ngram-mkn", "ntrain": args.ntrain, "mode": "static",
        "config": {"best_order": ORDERS[best], "orders_tried": list(ORDERS),
                   "selection": "argmin valid bpc", "per_order": per_order, "levels": levels,
                   "max_sum_to_1_dev": max(vdev, tdev, worst)},
        "valid_bpc": round(float(vb[best]), 5), "test_bpc": round(float(tb[best]), 5),
        "train_seconds": round(train_s, 2), "eval_seconds": round(eval_s, 2),
    }
    if args.no_log:
        print(rec)
    else:
        append_result(rec)


if __name__ == "__main__":
    main()
