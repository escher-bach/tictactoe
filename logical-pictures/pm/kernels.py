"""Numba kernels for the picture machine.  Counts and partitions only: no gradients.

Everything here is a single left-to-right pass over the text, so every quantity a
position sees is decided strictly before that position (THEORY 5.2, 6.1).
"""
import numpy as np
from numba import njit

# ----------------------------------------------------------------- profiles
# The profile of a record (n0, n1) is a determinable of the object that holds
# it.  Small records are kept exactly; large ones as (size level, ratio level).
N_EXACT = 8
N_SMALL = (N_EXACT + 1) * (N_EXACT + 2) // 2          # 45 exact profiles, n <= 8
N_LEV, N_RAT = 8, 28
NPROF = N_SMALL + N_LEV * N_RAT + 1                   # last index: unoccupied
UNOCC = NPROF - 1
COUNT_LIMIT = 1020                                     # records are halved beyond this
NPB = 33                                               # buckets of the incoming description
ST_MAX = 8.0


@njit(cache=True, nogil=True)
def profile(n0, n1):
    n = n0 + n1
    if n <= N_EXACT:
        return n * (n + 1) // 2 + n1
    if n < 13:
        lev = 0
    elif n < 20:
        lev = 1
    elif n < 32:
        lev = 2
    elif n < 56:
        lev = 3
    elif n < 100:
        lev = 4
    elif n < 200:
        lev = 5
    elif n < 500:
        lev = 6
    else:
        lev = 7
    r = (n1 + 0.5) / (n + 1.0)
    s = np.log(r / (1.0 - r))
    if s < -7.0:
        s = -7.0
    if s > 7.0:
        s = 7.0
    rb = int((s + 7.0) / 14.0 * (N_RAT - 0.001))
    return N_SMALL + lev * N_RAT + rb


@njit(cache=True, nogil=True)
def count_pass(fkey, flabel, freeze, log2size, q):
    """Records of one sort.  fkey[i] names the object fact i is about (0: none).

    For each fact, in text order: write the profile of the object's record as it
    stood before the fact, then add the fact to the record.

    From fact index `freeze` on, the records of objects that existed at that point
    are read but no longer written: what the corpus fixed stays fixed.  Objects
    first named after it are particulars of the new text and keep their records
    (THEORY 5.2: universals carry what the corpus fixed, particulars are new in
    every text).
    """
    size = 1 << log2size
    mask = np.uint64(size - 1)
    chk = np.zeros(size, np.uint32)
    c0 = np.zeros(size, np.uint16)
    c1 = np.zeros(size, np.uint16)
    late = np.zeros(size, np.bool_)
    used = 0
    limit = int(size * 0.85)
    for i in range(fkey.shape[0]):
        k = fkey[i]
        if k == 0:
            q[i] = UNOCC
            continue
        h = k * np.uint64(0x9E3779B97F4A7C15)
        h ^= h >> np.uint64(29)
        h *= np.uint64(0xBF58476D1CE4E5B9)
        slot = (h >> np.uint64(24)) & mask
        fp = np.uint32(h & np.uint64(0xFFFFFF)) | np.uint32(0x1000000)
        found = False
        while True:
            c = chk[slot]
            if c == fp:
                found = True
                break
            if c == 0:
                break
            slot = (slot + np.uint64(1)) & mask
        if found:
            q[i] = profile(c0[slot], c1[slot])
        else:
            q[i] = 0
        if not found:
            if used >= limit:
                continue
            chk[slot] = fp
            used += 1
            late[slot] = i >= freeze
        elif i >= freeze and not late[slot]:
            continue
        if flabel[i]:
            c1[slot] += 1
        else:
            c0[slot] += 1
        if c0[slot] + c1[slot] > COUNT_LIMIT:
            c0[slot] >>= 1
            c1[slot] >>= 1
    return used


@njit(cache=True, nogil=True)
def fit_table(st, q, label, lo, hi):
    """A configuration sort: (bucket of the description so far, profile of a new object).

    Its record is a count per cell.  Where a cell has little or no record, the default
    is independence (THEORY P4, Proposition 5.2): the new object's evidence is added to
    the description in log-odds.  The evidence of a profile is read off its margin.
    """
    T = np.zeros((NPB, NPROF, 2), np.float64)
    E = np.zeros(NPROF, np.float64)                    # what the description expected, per profile
    for i in range(lo, hi):
        s = st[i]
        pos = (s + ST_MAX) * (NPB - 1) / (2 * ST_MAX)
        j = int(pos)
        if j >= NPB - 1:
            j = NPB - 2
        w = pos - j
        T[j, q[i], label[i]] += 1.0 - w
        T[j + 1, q[i], label[i]] += w
        E[q[i]] += 1.0 / (1.0 + np.exp(-s))
    beta = np.zeros(NPROF, np.float64)
    for k in range(NPROF):
        n1 = T[:, k, 1].sum()
        n = n1 + T[:, k, 0].sum()
        if n > 0:
            a = (n1 + 0.5) / (n + 1.0)
            e = (E[k] + 0.5) / (n + 1.0)
            beta[k] = np.log(a / (1.0 - a)) - np.log(e / (1.0 - e))
    P = np.empty((NPB, NPROF), np.float32)
    for j in range(NPB):
        s = -ST_MAX + j * (2 * ST_MAX) / (NPB - 1)
        for k in range(NPROF):
            pc = 1.0 / (1.0 + np.exp(-(s + beta[k])))
            P[j, k] = (T[j, k, 1] + pc) / (T[j, k, 0] + T[j, k, 1] + 1.0)
    return P


@njit(cache=True, nogil=True)
def apply_table(P, st, q, out):
    """Read the configuration's record; returns the new description as a stretch."""
    for i in range(st.shape[0]):
        pos = (st[i] + ST_MAX) * (NPB - 1) / (2 * ST_MAX)
        j = int(pos)
        if j >= NPB - 1:
            j = NPB - 2
        w = pos - j
        p = (1.0 - w) * P[j, q[i]] + w * P[j + 1, q[i]]
        if p < 3e-4:
            p = 3e-4
        if p > 1 - 3e-4:
            p = 1 - 3e-4
        out[i] = np.log(p / (1.0 - p))


@njit(cache=True, nogil=True)
def bits(st, label, lo, hi):
    tot = 0.0
    for i in range(lo, hi):
        s = st[i] if label[i] else -st[i]
        # -log2 sigmoid(s)
        if s > 30:
            continue
        tot += np.log1p(np.exp(-s)) if s > -30 else -s
    return tot / np.log(2.0)


@njit(cache=True, nogil=True)
def gain_of(st, q, label, lo_fit, hi_fit, lo_val, hi_val):
    """Residue removed on the validation facts by conjoining one more object."""
    P = fit_table(st, q, label, lo_fit, hi_fit)
    tot = 0.0
    for i in range(lo_val, hi_val):
        pos = (st[i] + ST_MAX) * (NPB - 1) / (2 * ST_MAX)
        j = int(pos)
        if j >= NPB - 1:
            j = NPB - 2
        w = pos - j
        p = (1.0 - w) * P[j, q[i]] + w * P[j + 1, q[i]]
        if p < 3e-4:
            p = 3e-4
        if p > 1 - 3e-4:
            p = 1 - 3e-4
        tot -= np.log(p if label[i] else 1.0 - p)
    return tot / np.log(2.0)


# ----------------------------------------------------------------- links and series
@njit(cache=True, nogil=True)
def recency(key, x, log2size):
    """Recency link of a criterion: the most recent earlier position naming the same object.

    Returns, per position t: the sign that stood at that earlier position (F),
    the sign at the one before it (F2), and the distance on a log scale (A).
    -1 where the object has not been named before.
    """
    n = key.shape[0]
    size = 1 << log2size
    mask = np.uint64(size - 1)
    chk = np.zeros(size, np.uint64)
    last = np.full(size, -1, np.int32)
    prev = np.full(size, -1, np.int32)
    F = np.full(n, -1, np.int64)
    F2 = np.full(n, -1, np.int64)
    A = np.full(n, -1, np.int64)
    used = 0
    limit = int(size * 0.85)
    for t in range(n):
        k = key[t]
        if k == 0:
            continue
        h = k * np.uint64(0x9E3779B97F4A7C15)
        h ^= h >> np.uint64(31)
        slot = (h >> np.uint64(20)) & mask
        while chk[slot] != k and chk[slot] != 0:
            slot = (slot + np.uint64(1)) & mask
        if chk[slot] == 0:
            if used >= limit:
                continue
            chk[slot] = k
            used += 1
        u = last[slot]
        if u >= 0:
            F[t] = x[u]
            d = t - u
            a = 0
            while d > 1:
                d >>= 1
                a += 1
            A[t] = a
            if prev[slot] >= 0:
                F2[t] = x[prev[slot]]
        prev[slot] = u
        last[slot] = t
    return F, F2, A


@njit(cache=True, nogil=True)
def series(x, is_boundary, kmax):
    """Series along adjacency (TLP 5.2521): the signs since the last boundary sign, as one object.

    W[t]   = the run of signs after the most recent boundary before t (1 if empty)
    PW[k][t] = the k-th completed run before that
    """
    n = x.shape[0]
    W = np.empty(n, np.int64)
    PW = np.zeros((kmax, n), np.int64)
    cur = np.uint64(1)
    hist = np.zeros(kmax, np.uint64)
    for t in range(n):
        W[t] = cur
        for k in range(kmax):
            PW[k, t] = hist[k]
        s = x[t]
        if is_boundary[s]:
            for k in range(kmax - 1, 0, -1):
                hist[k] = hist[k - 1]
            hist[0] = cur
            cur = np.uint64(1)
        else:
            cur = (cur * np.uint64(0x100000001B3) + np.uint64(s + 1)) * np.uint64(0x9E3779B97F4A7C15)
            cur ^= cur >> np.uint64(32)
            cur = (cur >> np.uint64(2)) | np.uint64(2)
    return W, PW


def mix(a, b):
    """Conjunction of two criteria: the configuration object.  0 stays unoccupied."""
    a = a.astype(np.uint64, copy=False)
    b = b.astype(np.uint64, copy=False)
    with np.errstate(over="ignore"):
        h = (a ^ np.uint64(0x2545F4914F6CDD1D)) * np.uint64(0xFF51AFD7ED558CCD)
        h ^= h >> np.uint64(33)
        h = (h + b) * np.uint64(0xC4CEB9FE1A85EC53)
        h ^= h >> np.uint64(29)
        h |= np.uint64(1)
    return h


@njit(cache=True, nogil=True)
def partner_table(f, x, card, n):
    """Shown identities between values of a small determinable (move M1 for a recency key).

    For every pair (v, w): among positions whose value is v, how often does the sign
    there repeat the sign that stood at the most recent position whose value was w?
    One pass; card memory cells per position.
    """
    agree = np.zeros((card, card), np.int64)
    total = np.zeros((card, card), np.int64)
    last = np.full(card, -1, np.int64)
    for t in range(n):
        v = f[t]
        if v == 0 or v >= card:
            continue
        for w in range(1, card):
            if last[w] >= 0:
                total[v, w] += 1
                if last[w] == x[t]:
                    agree[v, w] += 1
        last[v] = x[t]
    return agree, total
