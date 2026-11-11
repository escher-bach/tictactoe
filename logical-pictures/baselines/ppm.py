#!/usr/bin/env python3
"""PPM character model (hashed contexts, unbounded int32 counts, no rescaling).

Blending: classic PPM back-off with escape + symbol exclusion.
At a position with history h, for k = K, K-1, ..., 0 (ctx_k = last k chars):
    let c(s) be the count of s in ctx_k, restricted to symbols NOT already
    excluded (i.e. not seen in any higher-order context visited so far),
    T = sum c(s), q = #{s : c(s) > 0}.  If the context is unseen or q == 0,
    back off with escape probability 1 (costs nothing).  Otherwise
      method C:  p(s) = c(s)/(T+q),        p(esc) = q/(T+q)
      method D:  p(s) = (2c(s)-1)/(2T),    p(esc) = q/(2T)
    symbols with c(s) > 0 are then excluded for the lower orders.
    Special case (needed for a proper distribution on a known finite alphabet):
    if q == number of symbols still un-excluded, nothing is left to escape to,
    so p(esc) = 0 and p(s) = c(s)/T.
  order -1: uniform over the symbols that are still un-excluded.
The full 27-way distribution is materialised at EVERY scored position and its
sum is checked (max |sum-1| is asserted < 1e-9), so this is a true probability
model, not just a code-length estimate.

Count updates (after the symbol is revealed):
  --ue 1 (update exclusion, the standard PPMC/PPMD rule): increment the symbol
         in the contexts of order k_found..K, where k_found is the highest order
         whose context already contained it (all orders 0..K if it was novel).
  --ue 0 (full update): increment in every order 0..K.

Modes:
  static : counts are built on train and then frozen.
  dynamic: the model keeps updating while it reads the evaluation text
           (train first, then valid -- for selection -- or train then test;
           the test-time model has NOT seen valid).  Only eval chars are scored.
The context history is reset at the start of the evaluation segment (the last
train chars are not used as context for the first eval chars).

Selection: (max order in {4,5,6,8}) x (method C/D) x (update exclusion on/off)
is chosen per mode by valid bpc; the full grid is stored in the result record.

Usage: python3 ppm.py --ntrain 1000000
"""
import os

os.environ.setdefault("NUMBA_NUM_THREADS", "2")
os.environ.setdefault("OMP_NUM_THREADS", "2")

import argparse
import time

import numba as nb
import numpy as np

from common import V, append_result, load_splits

ORDERS = (4, 5, 6, 8)
KBITS = 40                      # key = (order << 40) | base-27 code (27^8 < 2^39)
GOLD = np.uint64(0x9E3779B97F4A7C15)


@nb.njit(cache=True)
def _slot(key, shift):
    return np.int64((np.uint64(key) * GOLD) >> np.uint64(shift))


@nb.njit(cache=True)
def _lookup(hkeys, hval, shift, key):
    mask = hkeys.size - 1
    i = _slot(key, shift)
    while True:
        k = hkeys[i]
        if k == key:
            return hval[i]
        if k == -1:
            return -1
        i = (i + 1) & mask


@nb.njit(cache=True)
def _insert(hkeys, hval, shift, key, row):
    mask = hkeys.size - 1
    i = _slot(key, shift)
    while hkeys[i] != -1:
        i = (i + 1) & mask
    hkeys[i] = key
    hval[i] = row


@nb.njit(cache=True)
def _dists(counts, rows, kmax, excl, dC, dD):
    """Fill dC, dD with the PPMC / PPMD distributions over the 27 symbols."""
    for s in range(V):
        excl[s] = False
        dC[s] = 0.0
        dD[s] = 0.0
    remC = 1.0
    remD = 1.0
    n_rem = V
    for k in range(kmax, -1, -1):
        r = rows[k]
        if r < 0:
            continue
        T = 0
        q = 0
        for s in range(V):
            if (not excl[s]) and counts[r, s] > 0:
                T += counts[r, s]
                q += 1
        if q == 0:
            continue
        if q == n_rem:
            # every remaining symbol has been seen here: no escape is possible
            for s in range(V):
                if (not excl[s]) and counts[r, s] > 0:
                    dC[s] += remC * counts[r, s] / T
                    dD[s] += remD * counts[r, s] / T
                    excl[s] = True
            n_rem = 0
            break
        for s in range(V):
            if (not excl[s]) and counts[r, s] > 0:
                c = counts[r, s]
                dC[s] += remC * c / (T + q)
                dD[s] += remD * (2.0 * c - 1.0) / (2.0 * T)
                excl[s] = True
        remC *= q / (T + q)
        remD *= q / (2.0 * T)
        n_rem -= q
    if n_rem > 0:
        for s in range(V):
            if not excl[s]:
                dC[s] += remC / n_rem
                dD[s] += remD / n_rem


@nb.njit(cache=True)
def run(x, hkeys, hval, counts, nrows, K, ue, update, score):
    """Process x left to right.  Returns the (possibly re-allocated) state and
    (nllC, nllD, max |sum-1|) in bits, summed over x when score is True."""
    shift = 64
    h = hkeys.size
    while h > 1:
        h >>= 1
        shift -= 1
    pow27 = np.ones(K + 2, dtype=np.int64)
    for k in range(1, K + 2):
        pow27[k] = pow27[k - 1] * V
    rows = np.empty(K + 1, dtype=np.int64)
    keys = np.empty(K + 1, dtype=np.int64)
    excl = np.zeros(V, dtype=np.bool_)
    dC = np.zeros(V)
    dD = np.zeros(V)
    nllC = 0.0
    nllD = 0.0
    maxdev = 0.0
    hist = 0  # base-27 code of the last min(t, K) chars
    for t in range(x.size):
        s = x[t]
        kmax = t if t < K else K
        for k in range(kmax + 1):
            keys[k] = (np.int64(k) << KBITS) | (hist % pow27[k])
            rows[k] = _lookup(hkeys, hval, shift, keys[k])
        if score:
            _dists(counts, rows, kmax, excl, dC, dD)
            sc = 0.0
            sd = 0.0
            for j in range(V):
                sc += dC[j]
                sd += dD[j]
            dev = max(abs(sc - 1.0), abs(sd - 1.0))
            if dev > maxdev:
                maxdev = dev
            nllC -= np.log2(dC[s])
            nllD -= np.log2(dD[s])
        if update:
            klow = 0
            if ue:
                for k in range(kmax, -1, -1):
                    if rows[k] >= 0 and counts[rows[k], s] > 0:
                        klow = k
                        break
            for k in range(klow, kmax + 1):
                r = rows[k]
                if r < 0:
                    if nrows == counts.shape[0]:
                        new = np.zeros((2 * counts.shape[0], V), dtype=np.int32)
                        new[:nrows] = counts
                        counts = new
                    if 2 * (nrows + 1) > hkeys.size:
                        nk = np.full(2 * hkeys.size, -1, dtype=np.int64)
                        nv = np.zeros(2 * hkeys.size, dtype=np.int32)
                        shift -= 1
                        for i in range(hkeys.size):
                            if hkeys[i] != -1:
                                _insert(nk, nv, shift, hkeys[i], hval[i])
                        hkeys = nk
                        hval = nv
                    r = nrows
                    nrows += 1
                    _insert(hkeys, hval, shift, keys[k], r)
                counts[r, s] += 1
        hist = (hist * V + s) % pow27[K]
    return hkeys, hval, counts, nrows, nllC, nllD, maxdev


def new_state():
    return (np.full(1 << 16, -1, dtype=np.int64), np.zeros(1 << 16, dtype=np.int32),
            np.zeros((1 << 15, V), dtype=np.int32), 0)


def run_config(tr, va, te, K, ue):
    """Train one (K, ue) model and score valid/test in both modes, both methods."""
    out = {}
    t0 = time.time()
    hk, hv, cn, nr, _, _, _ = run(tr, *new_state(), K, ue, True, False)
    train_s = time.time() - t0
    devs = []
    # static: frozen counts
    for name, seg in (("valid", va), ("test", te)):
        _, _, _, _, nc, nd, dev = run(seg, hk, hv, cn, nr, K, ue, False, True)
        out[("static", "C", name)] = nc / seg.size
        out[("static", "D", name)] = nd / seg.size
        devs.append(dev)
    # dynamic on valid: work on a copy so the test-time model has not seen valid
    _, _, _, _, nc, nd, dev = run(va, hk.copy(), hv.copy(), cn[:nr].copy(), nr, K, ue, True, True)
    out[("dynamic", "C", "valid")] = nc / va.size
    out[("dynamic", "D", "valid")] = nd / va.size
    devs.append(dev)
    # dynamic on test: continues from the train-only state (mutates it; last use)
    _, _, _, nr2, nc, nd, dev = run(te, hk, hv, cn, nr, K, ue, True, True)
    out[("dynamic", "C", "test")] = nc / te.size
    out[("dynamic", "D", "test")] = nd / te.size
    devs.append(dev)
    return out, train_s, max(devs), int(nr), int(nr2)


def selftest():
    """Compare against a tiny dict-based PPM written independently."""
    from collections import defaultdict
    tr, va, te = load_splits(20_000)
    x = np.concatenate([tr[:3000], te[:500]])
    for K in (2, 4):
        for ue in (0, 1):
            _, _, _, _, nc, nd, dev = run(x, *new_state(), K, ue, True, True)
            tab = defaultdict(lambda: defaultdict(int))
            ref = {"C": 0.0, "D": 0.0}
            for t in range(x.size):
                s = int(x[t])
                kmax = min(t, K)
                ctxs = [tuple(x[t - k:t].tolist()) for k in range(kmax + 1)]
                for m in "CD":
                    excl, rem, p = set(), 1.0, None
                    for k in range(kmax, -1, -1):
                        cs = {a: c for a, c in tab[ctxs[k]].items() if a not in excl} if ctxs[k] in tab else {}
                        if not cs:
                            continue
                        T, q = sum(cs.values()), len(cs)
                        if q == V - len(excl):
                            p = rem * cs[s] / T
                            break
                        if s in cs:
                            p = rem * (cs[s] / (T + q) if m == "C" else (2 * cs[s] - 1) / (2 * T))
                            break
                        rem *= q / (T + q) if m == "C" else q / (2 * T)
                        excl |= set(cs)
                    if p is None:
                        p = rem / (V - len(excl))
                    ref[m] -= np.log2(p)
                klow = 0
                if ue:
                    for k in range(kmax, -1, -1):
                        if ctxs[k] in tab and tab[ctxs[k]].get(s, 0) > 0:
                            klow = k
                            break
                for k in range(klow, kmax + 1):
                    tab[ctxs[k]][s] += 1
            print(f"K={K} ue={ue}: numba C {nc:.6f} D {nd:.6f} | ref C {ref['C']:.6f} D {ref['D']:.6f} | sumdev {dev:.1e}")
            assert abs(nc - ref["C"]) < 1e-6 and abs(nd - ref["D"]) < 1e-6 and dev < 1e-9
    print("selftest OK")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ntrain", type=int, default=1_000_000)
    ap.add_argument("--orders", type=int, nargs="+", default=list(ORDERS))
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--no-log", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        selftest()
        return

    tr, va, te = load_splits(args.ntrain)
    grid = {}
    info = {}
    t_all = time.time()
    worst = 0.0
    for K in args.orders:
        for ue in (1, 0):
            out, train_s, dev, nr, nr2 = run_config(tr, va, te, K, ue)
            worst = max(worst, dev)
            info[(K, ue)] = (train_s, nr, nr2)
            for (mode, meth, split), v in out.items():
                grid[(mode, K, meth, ue, split)] = v
            print(f"K={K} ue={ue} train {train_s:6.1f}s contexts {nr:>9d} | "
                  + " | ".join(f"{mode[:3]}-{meth} v {out[(mode, meth, 'valid')]:.4f} t {out[(mode, meth, 'test')]:.4f}"
                               for mode in ("static", "dynamic") for meth in "CD"), flush=True)
    assert worst < 1e-9, worst
    print(f"sum-to-1 at every scored position (all configs, both methods): max dev {worst:.2e}")
    grid_s = time.time() - t_all

    for mode in ("static", "dynamic"):
        cands = [(grid[(mode, K, m, ue, "valid")], K, m, ue)
                 for K in args.orders for m in "CD" for ue in (1, 0)]
        vbest, K, m, ue = min(cands)
        table = {f"K{k}_{mm}_ue{u}": {"valid_bpc": round(grid[(mode, k, mm, u, "valid")], 5),
                                      "test_bpc": round(grid[(mode, k, mm, u, "test")], 5)}
                 for k in args.orders for mm in "CD" for u in (1, 0)}
        rec = {
            "model": "ppm", "ntrain": args.ntrain, "mode": mode,
            "config": {"max_order": K, "escape": m, "update_exclusion": bool(ue),
                       "symbol_exclusion": True, "counts": "int32, unbounded, no rescaling",
                       "selection": "argmin valid bpc over orders x {C,D} x update-exclusion {on,off}",
                       "contexts_after_train": info[(K, ue)][1],
                       "contexts_after_test": info[(K, ue)][2] if mode == "dynamic" else info[(K, ue)][1],
                       "max_sum_to_1_dev": worst, "grid": table},
            "valid_bpc": round(vbest, 5), "test_bpc": round(grid[(mode, K, m, ue, "test")], 5),
            "train_seconds": round(info[(K, ue)][0], 2), "grid_seconds": round(grid_s, 1),
        }
        print(f"{mode}: best K={K} method {m} ue={ue}: valid {rec['valid_bpc']:.4f} test {rec['test_bpc']:.4f}")
        if args.no_log:
            continue
        append_result(rec)


if __name__ == "__main__":
    main()
