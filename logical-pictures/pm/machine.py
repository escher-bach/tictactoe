"""The picture machine: a form (sorts, determinables, links) learned by analysis.

Vocabulary (THEORY.md 4.3, 5.2):
  feature  a determinable of the context of position t, decided before t
  key      a conjunction of features: a criterion of identity.  Its classes are objects
  sort     a key plus, optionally, a pointer.  A sort keeps one record per object
  record   counts of what was shown at the object's earlier occurrences
  chain    the read-out: each sort's object is conjoined with the description so far

Feature specs are tuples:
  ('S', j)        the sign j steps back                       (path along adjacency)
  ('C', j, d)     its class at depth d of the sign code       (tabulation, M3)
  ('W', c)        the run of signs since the last sign of boundary class c   (series)
  ('PW', c, k)    the k-th completed run before it            (path along the recency link of c)
  ('RF', key)     what stood at the most recent position naming the same key-object (recency)
  ('RF2', key)    the one before that
  ('RA', key)     how long ago, on a log scale                (a scale)
  ('EQ', f, g)    whether two features show the same value    (sameness of sign)
  ('CLS', f)      the class of f's object under any-split over its record (abstraction)
  ('XC', f)       the class of f's value under identities shown by recency (a learned key)
  ('WC', k, d)    the class, at depth d of the object tree, of the k-th completed object back
  ('CUR', d)      the class at depth d of the object now being spelled (stated before its signs)
"""
import pickle
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from numba import njit

from . import kernels as K
from .tower import Tower


# ----------------------------------------------------------------- sign code (M3)
def build_code(x, V, eps=0.1):
    """Tabulate the signs: a tree of two-valued determinables, by any-split on sign objects.

    A sign-object satisfies the test "followed by b" (or "preceded by b") when its
    occurrences do so more often than its class does.  Among the tests that
    eps-split the class we take the one whose split says most about the neighbour;
    Theorem 1 says the choice is free, so this only affects the size of records.
    """
    big = np.zeros((V, V), np.float64)
    np.add.at(big, (x[:-1], x[1:]), 1.0)
    cnt = big.sum(1) + 1e-9
    nodes = []          # (members, child0, child1); a child is ('leaf', sign) or ('node', id)

    def mi(joint):
        tot = joint.sum()
        if tot <= 0:
            return 0.0
        pj = joint / tot
        pa = pj.sum(1, keepdims=True)
        pb = pj.sum(0, keepdims=True)
        nz = pj > 0
        return float((pj[nz] * np.log2(pj[nz] / (pa @ pb)[nz])).sum())

    def rec(A):
        if len(A) == 1:
            return ("leaf", A[0])
        A = np.array(A)
        mass = cnt[A]
        best = None
        for direction in (0, 1):
            M = big if direction == 0 else big.T          # M[a, b]: a followed/preceded by b
            dens = M[A] / mass[:, None]
            avg = M[A].sum(0) / mass.sum()
            for b in range(V):
                side = dens[:, b] > avg[b]
                if side.all() or not side.any():
                    continue
                share = mass[side].sum() / mass.sum()
                splits = eps <= share <= 1 - eps
                # what the split bit of a neighbour says about the sign, both ways
                j1 = np.stack([big[A[side]].sum(0), big[A[~side]].sum(0)])
                j2 = np.stack([big[:, A[side]].sum(1), big[:, A[~side]].sum(1)])
                score = (1 if splits else 0, mi(j1) + mi(j2))
                if best is None or score > best[0]:
                    best = (score, side.copy())
        side = best[1] if best is not None else np.arange(len(A)) < len(A) // 2
        me = len(nodes)
        nodes.append(None)
        c1 = rec(list(A[side]))
        c0 = rec(list(A[~side]))
        nodes[me] = (list(A), c0, c1)
        return ("node", me)

    present = [s for s in range(V) if cnt[s] > 0.5]
    absent = [s for s in range(V) if cnt[s] <= 0.5]
    root = rec(present) if len(present) > 1 else ("leaf", present[0])
    # signs never seen in the code-building text get a place under one extra node
    if absent:
        def chain(lst):
            if len(lst) == 1:
                return ("leaf", lst[0])
            me = len(nodes)
            nodes.append(None)
            h = len(lst) // 2
            nodes[me] = (lst, chain(lst[:h]), chain(lst[h:]))
            return ("node", me)
        me = len(nodes)
        nodes.append(None)
        nodes[me] = (present + absent, root, chain(absent))
        root = ("node", me)

    paths = {}

    def walk(c, path):
        if c[0] == "leaf":
            paths[c[1]] = path
        else:
            walk(nodes[c[1]][1], path + [(c[1], 0)])
            walk(nodes[c[1]][2], path + [(c[1], 1)])
    walk(root, [])
    D = max(len(p) for p in paths.values())
    codenode = np.full((V, D), -1, np.int16)
    codebit = np.zeros((V, D), np.uint8)
    depth = np.zeros(V, np.int8)
    classid = np.zeros((V, D + 1), np.int64)        # class at depth d: node id, or a leaf's own id
    for s, p in paths.items():
        depth[s] = len(p)
        for d, (v, b) in enumerate(p):
            codenode[s, d] = v
            codebit[s, d] = b
        for d in range(D + 1):
            classid[s, d] = (p[d][0] + 1) if d < len(p) else (len(nodes) + 1 + s)
    members = [set(nd[0]) for nd in nodes]
    return dict(V=V, D=D, codenode=codenode, codebit=codebit, depth=depth, classid=classid,
                members=members, nnodes=len(nodes))


def boundary_candidates(code, x, lo=0.03, hi=0.45):
    """Sign classes that may serve as boundaries of a series: code-tree classes and frequent signs."""
    V = code["V"]
    freq = np.bincount(x, minlength=V) / len(x)
    out = []
    for m in code["members"]:
        share = sum(freq[s] for s in m)
        if lo <= share <= hi and frozenset(m) not in out:
            out.append(frozenset(m))
    for s in np.argsort(-freq)[:8]:
        if freq[s] >= lo and frozenset([int(s)]) not in out:
            out.append(frozenset([int(s)]))
    return out


# ----------------------------------------------------------------- fact keys
@njit(cache=True, nogil=True)
def fkeys_literal(key, fpos, fnode, out):
    for i in range(fpos.shape[0]):
        k = key[fpos[i]]
        if k == 0:
            out[i] = 0
        else:
            h = (k ^ np.uint64(np.int64(fnode[i]) + 1)) * np.uint64(0xD6E8FEB86659FD93)
            h ^= h >> np.uint64(32)
            out[i] = h | np.uint64(1)


@njit(cache=True, nogil=True)
def fkeys_pointer(key, ptr, fpos, fdepth, fnode, fbit, codenode, codebit, out, lab):
    """A pointer sort records whether the fact repeats the fact at the link's target."""
    for i in range(fpos.shape[0]):
        t = fpos[i]
        k = key[t]
        s = np.int64(ptr[t]) - 1
        d = fdepth[i]
        if k == 0 or s < 0 or codenode[s, d] != fnode[i]:
            out[i] = 0
            lab[i] = fbit[i]
        else:
            h = (k ^ np.uint64(d + 1)) * np.uint64(0xD6E8FEB86659FD93)
            h ^= h >> np.uint64(32)
            out[i] = h | np.uint64(1)
            lab[i] = fbit[i] ^ codebit[s, d]


def mentions_cur(spec):
    """Does a feature or key depend, at any depth, on the class of the object being spelled?"""
    if not isinstance(spec, tuple):
        return False
    if spec and spec[0] == "CUR":
        return True
    return any(mentions_cur(part) for part in spec)


def canon_key(feats):
    return tuple(sorted(set(feats), key=repr))


# ----------------------------------------------------------------- a text with its picture
class Picture:
    """A text, its facts under a sign code, and the features built over it so far."""

    def __init__(self, x, code, bounds, ntrain, aux=None, tower=None, open_at=None, all_nodes=False):
        self.x = np.ascontiguousarray(x, dtype=np.int64)
        self.n = len(x)
        self.code, self.bounds, self.ntrain = code, bounds, ntrain
        self.aux = {} if aux is None else aux          # learned class maps, keyed by feature spec
        self.tower = tower
        self.open_at = open_at                          # positions where the picture says an object begins
        if all_nodes:
            self._all_node_facts(ntrain)
            return
        sd = code["depth"][self.x].astype(np.int64)
        cd = np.zeros(self.n, np.int64)                 # object facts stated where an object begins
        if tower is not None:
            typ_at = np.zeros(self.n, np.int64)
            typ_at[tower.start] = tower.type_of
            cd[tower.start] = tower.depth[tower.type_of]
        self.off = np.concatenate([[0], np.cumsum(cd + sd)])
        self.M = int(self.off[-1])
        self.fpos = np.repeat(np.arange(self.n, dtype=np.int32), cd + sd)
        j = np.arange(self.M) - self.off[self.fpos]
        self.fobj = j < cd[self.fpos]                   # is this a fact about an object?
        jd = np.where(self.fobj, 0, j - cd[self.fpos])
        self.fdepth = jd.astype(np.int8)
        fnode = code["codenode"][self.x[self.fpos], jd].astype(np.int64)
        fbit = code["codebit"][self.x[self.fpos], jd]
        if tower is not None and self.fobj.any():
            ty = typ_at[self.fpos[self.fobj]]
            jo = j[self.fobj]
            fnode[self.fobj] = code["nnodes"] + 1 + tower.pnode[ty, jo]
            fbit = fbit.copy()
            fbit[self.fobj] = tower.pbit[ty, jo]
        self.fnode = fnode.astype(np.int32)
        self.fbit = fbit.astype(np.uint8)
        self.Mtrain = int(self.off[ntrain])
        self.fupd = np.ones(self.M, np.uint8)
        self.cache = {}
        self.keycache = {(): np.ones(self.n, np.uint64)}
        self.log2pos = max(12, int(np.ceil(np.log2(self.n * 1.4))))

    def _all_node_facts(self, ntrain):
        """A picture that asks, at every position, about every determinable of the sign code,
        not only those on the path of the sign that occurs.  Facts off that path are read
        from the records and never written.  Used to see how determinate the picture is."""
        code, n = self.code, self.n
        nn = code["nnodes"]
        ndepth = np.zeros(nn, np.int64)
        for s in range(code["V"]):
            for d in range(code["depth"][s]):
                ndepth[code["codenode"][s, d]] = d
        self.off = np.arange(n + 1, dtype=np.int64) * nn
        self.M = n * nn
        self.fpos = np.repeat(np.arange(n, dtype=np.int32), nn)
        node = np.tile(np.arange(nn), n)
        d = ndepth[node]
        self.fdepth = d.astype(np.int8)
        self.fnode = node.astype(np.int32)
        xs = self.x[self.fpos]
        on = code["codenode"][xs, d] == node
        self.fupd = on.astype(np.uint8)
        self.fbit = np.where(on, code["codebit"][xs, d], 0).astype(np.uint8)
        self.fobj = np.zeros(self.M, bool)
        self.Mtrain = ntrain * nn
        self.cache = {}
        self.keycache = {(): np.ones(n, np.uint64)}
        self.log2pos = max(12, int(np.ceil(np.log2(n * 1.4))))

    # -- features ---------------------------------------------------------
    def feature(self, spec):
        if spec in self.cache:
            return self.cache[spec]
        kind = spec[0]
        n, x = self.n, self.x
        if kind == "S":
            j = spec[1]
            out = np.zeros(n, np.uint64)
            out[j:] = x[:n - j] + 1
        elif kind == "C":
            j, d = spec[1], spec[2]
            out = np.zeros(n, np.uint64)
            out[j:] = self.code["classid"][x[:n - j], d]
        elif kind in ("W", "PW"):
            c = spec[1]
            isb = np.zeros(self.code["V"], np.bool_)
            isb[list(self.bounds[c])] = True
            W, PW = K.series(x, isb, 3)
            self.cache[("W", c)] = W.astype(np.uint64)
            for k in range(3):
                self.cache[("PW", c, k + 1)] = PW[k].astype(np.uint64)
            return self.cache[spec]
        elif kind in ("RF", "RF2", "RA"):
            key = self.key(spec[1])
            F, F2, A = K.recency(key, x, self.log2pos)
            self.cache[("RF", spec[1])] = (F + 1).astype(np.uint64)
            self.cache[("RF2", spec[1])] = (F2 + 1).astype(np.uint64)
            self.cache[("RA", spec[1])] = (A + 1).astype(np.uint64)
            return self.cache[spec]
        elif kind == "EQ":
            a, b = self.feature(spec[1]), self.feature(spec[2])
            out = np.where((a == 0) | (b == 0), 0, 1 + (a == b)).astype(np.uint64)
        elif kind == "CLS":
            obj = self.feature(spec[1])
            if spec not in self.aux:
                self.aux[spec] = self.fit_classes(obj)
            ids, cls = self.aux[spec]
            if len(ids):
                pos = np.searchsorted(ids, obj)
                pos[pos >= len(ids)] = 0
                out = np.where(ids[pos] == obj, cls[pos], np.uint64(1))
            else:
                out = np.ones(n, np.uint64)
            out = np.where(obj == 0, np.uint64(0), out).astype(np.uint64)
        elif kind == "RUN":
            W, PW = K.series_open(x, self.open_at, 3)
            self.cache[("RUN", 0)] = W.astype(np.uint64)
            for k in range(3):
                self.cache[("RUN", k + 1)] = PW[k].astype(np.uint64)
            return self.cache[spec]
        elif kind in ("WC", "CUR"):
            tw = self.tower
            cur = tw.run_tok
            if kind == "CUR":
                tok = cur
            else:
                # the k-th completed object before the one in progress
                last = np.searchsorted(tw.start, np.arange(n), side="right") - 1
                tok = np.where(cur >= 0, cur - spec[1], last - (spec[1] - 1))
            d = min(spec[-1], tw.max_depth)
            out = np.where(tok >= 0, tw.classid[tw.type_of[np.maximum(tok, 0)], d], 0).astype(np.uint64)
        elif kind == "XC":
            fv = self.feature(spec[1])
            if spec not in self.aux:
                self.aux[spec] = self.fit_partners(fv)
            cmap = self.aux[spec]
            out = np.where(fv < len(cmap), cmap[np.minimum(fv, len(cmap) - 1).astype(np.int64)],
                           np.uint64(0)).astype(np.uint64)
        else:
            raise ValueError(spec)
        self.cache[spec] = out
        return out

    def fit_partners(self, fv, margin=0.15, min_count=20):
        """Classes of values that recency shows to be the same key (THEORY 7.2, M1).

        v and w are identified when the sign after v repeats the sign after the most
        recent w more often than v's median partner does, by a margin.  Classes are the
        connected components of that relation; a value with no partner stays alone.
        """
        card = int(fv[:self.ntrain].max()) + 1
        if card > 512:
            return np.zeros(1, np.uint64)
        agree, total = K.partner_table(fv, self.x, card, self.ntrain)
        rate = agree / np.maximum(total, 1)
        ok = total >= min_count
        parent = list(range(card))

        def find(a):
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a
        for v in range(1, card):
            if not ok[v].any():
                continue
            base = np.median(rate[v][ok[v]])
            for w in range(1, card):
                if w != v and ok[v, w] and rate[v, w] >= base + margin:
                    parent[find(v)] = find(w)
        cmap = np.array([0] + [find(v) for v in range(1, card)], np.uint64)
        return cmap

    def key(self, keyspec):
        """Conjunction: the configuration of the constituents; unoccupied where any of them is."""
        if keyspec in self.keycache:
            return self.keycache[keyspec]
        k = self.keycache[()]
        occ = np.ones(self.n, bool)
        for f in keyspec:
            fv = self.feature(f)
            k = K.mix(k, fv)
            occ &= fv != 0
        return np.where(occ, k, np.uint64(0))

    def remember_key(self, keyspec):
        self.keycache[keyspec] = self.key(keyspec)

    # -- abstraction: classes of objects by any-split over their records --------
    def fit_classes(self, obj, cmin=30, eps=0.12, floor=0.004, maxdepth=2, min_obs=4):
        """Objects of a sort, classed by any-split (WITNESSES Thm 1) with a size floor.

        An object's determinables are the coarse facts shown at its occurrences (code
        nodes of depth <= maxdepth).  A test holds of an object when most of its
        occurrences show it; unoccupied counts as false.  Classes are split by the most
        balanced test that eps-splits them and are left alone below the floor.
        Only the training part of the text is used.
        """
        ntr = self.ntrain
        ids, counts = np.unique(obj[:ntr], return_counts=True)
        keep = (counts >= cmin) & (ids != 0)
        ids, counts = ids[keep], counts[keep].astype(np.float64)
        if len(ids) < 4:
            return ids, np.full(len(ids), 2, np.uint64)
        sel = (self.fdepth[:self.Mtrain] <= maxdepth) & ~self.fobj[:self.Mtrain]
        fo = obj[self.fpos[:self.Mtrain][sel]]
        fn = self.fnode[:self.Mtrain][sel].astype(np.int64)
        fb = self.fbit[:self.Mtrain][sel]
        pos = np.searchsorted(ids, fo)
        pos[pos >= len(ids)] = 0
        ok = ids[pos] == fo
        nn = self.code["nnodes"]
        flat = pos[ok] * nn + fn[ok]
        n1 = np.bincount(flat[fb[ok] == 1], minlength=len(ids) * nn).reshape(len(ids), nn)
        n0 = np.bincount(flat[fb[ok] == 0], minlength=len(ids) * nn).reshape(len(ids), nn)
        usable = (n0 + n1).sum(0) > 0
        R = np.concatenate([(n1 > n0) & (n1 + n0 >= min_obs), (n0 > n1) & (n1 + n0 >= min_obs)], 1)
        R = R[:, np.concatenate([usable, usable])]
        total = counts.sum()
        cls = np.zeros(len(ids), np.uint64)
        nxt = [2]                                       # class 1 is "rare or unseen"

        def split(members):
            w = counts[members]
            m = w.sum()
            if m >= floor * total and len(members) > 1:
                share = (R[members] * w[:, None]).sum(0) / m
                bal = np.minimum(share, 1 - share)
                s = int(np.argmax(bal))
                if bal[s] >= eps:
                    side = R[members, s]
                    split(members[side])
                    split(members[~side])
                    return
            cls[members] = nxt[0]
            nxt[0] += 1
        split(np.arange(len(ids)))
        return ids, cls

    # -- sorts ---------------------------------------------------------------
    def fact_keys(self, sort):
        keyspec, ptr = sort
        key = self.key(keyspec)
        fkey = np.empty(self.M, np.uint64)
        if ptr is None:
            fkeys_literal(key, self.fpos, self.fnode, fkey)
            if self.tower is not None and mentions_cur(keyspec):
                fkey[self.fobj] = 0                     # an object's class is not known while it is being stated
            return fkey, self.fbit
        lab = np.empty(self.M, np.uint8)
        fkeys_pointer(key, self.feature(ptr), self.fpos, self.fdepth, self.fnode, self.fbit,
                      self.code["codenode"], self.code["codebit"], fkey, lab)
        if self.tower is not None:                      # pointers show signs, not object facts
            fkey[self.fobj] = 0
            lab[self.fobj] = self.fbit[self.fobj]
        return fkey, lab

    def profiles(self, sort, freeze=None):
        fkey, lab = self.fact_keys(sort)
        q = np.empty(self.M, np.uint16)
        nocc = int(np.count_nonzero(fkey))
        log2 = max(10, min(27, int(np.ceil(np.log2(max(nocc, 1) * 2.2)))))
        K.count_pass(fkey, lab, self.fupd, self.M if freeze is None else freeze, log2, q)
        return q, lab


# ----------------------------------------------------------------- the learning rule
class Learner:
    """Analysis (THEORY 7): admit the candidate identity that removes most residue.

    Candidates are produced from what is already in the picture by path,
    conjunction, recency and series, and by any-split classes of admitted objects.
    No formulas are enumerated: every candidate is one existing key and one more feature.
    """

    def __init__(self, x, V, ntrain, max_key=6, max_offset=12, class_depths=(1, 2, 3),
                 threads=3, min_gain=0.002, log=print, use=("S", "C", "W", "R", "EQ", "CLS", "PTR", "XC"),
                 code_eps=0.1, tower=None, tower_depths=(2, 3, 5, 7, 10, 16), open_at=None):
        self.code = build_code(x[:ntrain], V, code_eps)
        self.bounds = boundary_candidates(self.code, x[:ntrain]) if "W" in use else []
        self.tower_args = tower
        tw = None
        if tower is not None:                           # tower: dict(boundary=..., rounds=...)
            tw = Tower(x, V, ntrain=ntrain, log=log, **tower)
        self.tower_depths = tower_depths
        self.pic = Picture(x, self.code, self.bounds, ntrain, tower=tw, open_at=open_at)
        self.use, self.max_key, self.max_offset = set(use), max_key, max_offset
        self.class_depths = [d for d in class_depths if d < self.code["D"]]
        self.threads, self.min_gain, self.log = threads, min_gain, log
        p = self.pic
        self.st = np.zeros(p.M, np.float32)
        self.sorts, self.tables, self.history = [], [], []
        self.pool, self.ptrs = [], [None]
        self.gains = {}                                 # candidate -> (gain in bits, round)
        self.round = 0
        self.nval = p.n - ntrain
        self._admit(((), None))
        self._add(("S", 1))
        for d in self.class_depths:
            self._add(("C", 1, d))
        for c in range(len(self.bounds)):
            self._add(("W", c))
        if "EQ" in self.use:
            self._add(("EQ", ("S", 1), ("S", 2)))
        self._add(("XC", ("S", 1)))
        if open_at is not None:
            self.pool.append(("RUN", 0))
        if tw is not None:
            for d in self.tower_depths:
                self.pool.append(("WC", 1, d))
                self.pool.append(("CUR", d))

    # -- bookkeeping -------------------------------------------------------
    def _add(self, f):
        kind = f[0]
        tag = {"S": "S", "C": "C", "W": "W", "PW": "W", "RF": "R", "RF2": "R", "RA": "R",
               "EQ": "EQ", "CLS": "CLS", "XC": "XC"}[kind]
        if tag not in self.use or f in self.pool:
            return
        if kind in ("S", "C") and f[1] > self.max_offset:
            return
        self.pool.append(f)

    def val_bits(self):
        p = self.pic
        return K.bits(self.st, p.fbit, p.Mtrain, p.M)

    def _admit(self, sort):
        p = self.pic
        q, lab = p.profiles(sort)
        if sort[1] is None:
            P = K.fit_table(self.st, q, lab, p.fupd, 0, p.Mtrain)
            new = np.empty_like(self.st)
            K.apply_table(P, self.st, q, new)
        else:
            new = self._apply_pointer(sort, q, lab)[0]
            P = None
        before = self.val_bits()
        self.st = new
        after = self.val_bits()
        self.sorts.append(sort)
        self.history.append(after / self.nval)
        p.remember_key(sort[0])
        self.log(f"  admitted #{len(self.sorts)}: {fmt_sort(sort, self.bounds)}   "
                 f"valid {before / self.nval:.4f} -> {after / self.nval:.4f} bpc")
        # operations on what is now in the picture produce the next candidates
        keyspec, ptr = sort
        for f in keyspec:
            if f[0] in ("S", "C"):
                self._add(("S", f[1] + 1))
                for d in self.class_depths:
                    self._add(("C", f[1] + 1, d))
                if f[1] + 1 <= 4:
                    self._add(("EQ", ("S", 1), ("S", f[1] + 1)))
            if f[0] == "W":
                self._add(("PW", f[1], 1))
                self._add(("CLS", f))
            if f[0] == "RUN" and f[1] < 3 and ("RUN", f[1] + 1) not in self.pool:
                self.pool.append(("RUN", f[1] + 1))
            if f[0] == "WC" and f[1] < 3:
                for d in self.tower_depths:
                    if ("WC", f[1] + 1, d) not in self.pool:
                        self.pool.append(("WC", f[1] + 1, d))
            if f[0] == "PW":
                if f[2] < 3:
                    self._add(("PW", f[1], f[2] + 1))
                self._add(("CLS", f))
        if ptr is None and keyspec and "R" in self.use:
            self._add(("RF", keyspec))
            self._add(("RA", keyspec))
            self._add(("EQ", ("RF", keyspec), ("RF2", keyspec)))
            if "PTR" in self.use and ("RF", keyspec) not in self.ptrs:
                self.ptrs.append(("RF", keyspec))
        if ptr is not None and ptr not in self.ptrs:
            self.ptrs.append(ptr)                       # an admitted bare pointer may now take keys

    def _apply_pointer(self, sort, q, lab):
        """Conjoin a pointer sort.  Its record speaks about agreement with the pointed sign,
        so the description is turned to face the pointed bit, conjoined, and turned back."""
        p = self.pic
        sign = self._pointer_sign(sort)
        st_rel = self.st * sign
        P = K.fit_table(st_rel, q, lab, p.fupd, 0, p.Mtrain)
        new = np.empty_like(self.st)
        K.apply_table(P, st_rel, q, new)
        return new * sign, P

    def _pointer_sign(self, sort):
        """+1 where the pointed sign's bit is 0 or the pointer is unoccupied, -1 where it is 1."""
        p = self.pic
        ptr = p.feature(sort[1]).astype(np.int64)[p.fpos] - 1
        cb = p.code["codebit"][np.maximum(ptr, 0), p.fdepth]
        cn = p.code["codenode"][np.maximum(ptr, 0), p.fdepth]
        flip = (ptr >= 0) & (cn == p.fnode) & (cb == 1) & ~p.fobj
        return np.where(flip, -1.0, 1.0).astype(np.float32)

    # -- candidates ----------------------------------------------------------
    def candidates(self):
        out = []
        for ptr in self.ptrs:
            bases = [()] + [s[0] for s in self.sorts if s[1] == ptr and s[0]]
            if ptr is not None:
                out.append(((), ptr))
            elif "PTR" in self.use:
                # bare pointers: "repeats the sign j back", and "repeats what followed the
                # most recent position with the same value of one determinable"
                for j in range(1, 5):
                    out.append(((), ("S", j)))
                for f in self.pool:
                    if f[0] in ("S", "C", "XC", "CLS") and "R" in self.use:
                        out.append(((), ("RF", (f,))))
            for b in bases:
                if len(b) >= self.max_key:
                    continue
                for f in self.pool:
                    if f in b or f == ptr:
                        continue
                    out.append((canon_key(b + (f,)), ptr))
        have = set(self.sorts)
        return [c for c in dict.fromkeys(out) if c not in have]

    def evaluate(self, sort):
        p = self.pic
        q, lab = p.profiles(sort)
        st = self.st if sort[1] is None else self.st * self._pointer_sign(sort)
        after = K.gain_of(st, q, lab, p.fupd, 0, p.Mtrain, p.Mtrain, p.M)
        return self.cur - after

    def _eval_many(self, sorts):
        for s in sorts:                                 # build features outside the threads
            for f in s[0]:
                self.pic.feature(f)
            if s[1] is not None:
                self.pic.feature(s[1])
        if self.threads > 1 and len(sorts) > 1:
            with ThreadPoolExecutor(self.threads) as ex:
                gains = list(ex.map(self.evaluate, sorts))
        else:
            gains = [self.evaluate(s) for s in sorts]
        for s, g in zip(sorts, gains):
            self.gains[s] = (g, self.round)

    def step(self):
        """One move of analysis.  Returns False when no candidate identity is shown."""
        self.round += 1
        self.cur = self.val_bits()
        cands = self.candidates()
        t0 = time.time()
        fresh = [c for c in cands if c not in self.gains]
        self._eval_many(fresh)
        live = set(cands)
        n_re = 0
        if not live:
            return False
        while True:                                     # lazy re-evaluation of stale gains
            ranked = sorted(live, key=lambda c: -self.gains[c][0])
            if self.gains[ranked[0]][1] == self.round:
                break
            stale = [c for c in ranked[:self.threads * 3] if self.gains[c][1] != self.round]
            self._eval_many(stale)
            n_re += len(stale)
        best = ranked[0]
        g = self.gains[best][0]
        self.log(f"round {self.round}: {len(fresh)} new + {n_re} re-tested candidates, "
                 f"{time.time() - t0:.0f}s, best removes {g / self.nval:.4f} bpc")
        if g < self.min_gain * self.nval:
            return False
        self._admit(best)
        del self.gains[best]
        return True

    def run(self, max_sorts=40):
        while len(self.sorts) < max_sorts and self.step():
            pass
        return self.form()

    def form(self):
        return dict(code=self.code, bounds=self.bounds, sorts=list(self.sorts),
                    history=list(self.history), tower=self.tower_args)


def fmt_feature(f, bounds=None):
    k = f[0]
    if k == "S":
        return f"sign[-{f[1]}]"
    if k == "C":
        return f"class{f[2]}[-{f[1]}]"
    if k in ("W", "PW"):
        b = "".join(sorted(chr(97 + s) if s < 26 else "_" for s in bounds[f[1]])) if bounds else str(f[1])
        return (f"run<{b}>" if k == "W" else f"run<{b}>-{f[2]}")
    if k in ("RF", "RF2", "RA"):
        inner = " & ".join(fmt_feature(g, bounds) for g in f[1])
        return {"RF": "last", "RF2": "last2", "RA": "age"}[k] + "{" + inner + "}"
    if k == "EQ":
        return f"same({fmt_feature(f[1], bounds)}, {fmt_feature(f[2], bounds)})"
    if k == "CLS":
        return f"cls({fmt_feature(f[1], bounds)})"
    if k == "XC":
        return f"key({fmt_feature(f[1], bounds)})"
    if k == "RUN":
        return "object" if f[1] == 0 else f"object[-{f[1]}]"
    if k == "WC":
        return f"kind{f[2]}[-{f[1]}]"
    if k == "CUR":
        return f"kind{f[1]}[this]"
    return str(f)


def fmt_sort(sort, bounds=None):
    key = " & ".join(fmt_feature(f, bounds) for f in sort[0]) or "(everything)"
    return key if sort[1] is None else f"{key}  => repeats {fmt_feature(sort[1], bounds)}"


# ----------------------------------------------------------------- using a learned form
def run_form(form, x_train, x_test, static=False, log=None, sorts=None, all_nodes=False, open_at=None):
    """Picture a new text in a form already found.  Returns bpc on x_test after each sort.

    static=False: every record keeps growing while the test text is read (each fact is
    scored before it is counted).  static=True: the records of objects that exist after
    the training text are frozen; objects first named in the test text, and recency
    links, are particulars of that text and behave as during training.
    """
    x = np.concatenate([x_train, x_test])
    tw = None
    if form.get("tower"):
        tw = Tower(x, form["code"]["V"], ntrain=len(x_train), **form["tower"])
    p = Picture(x, form["code"], form["bounds"], len(x_train), tower=tw, open_at=open_at, all_nodes=all_nodes)
    st = np.zeros(p.M, np.float32)
    ntest = len(x_test)
    ladder = []
    for sort in (form["sorts"] if sorts is None else sorts):
        q, lab = p.profiles(sort, freeze=p.Mtrain if static else None)
        if sort[1] is None:
            P = K.fit_table(st, q, lab, p.fupd, 0, p.Mtrain)
            new = np.empty_like(st)
            K.apply_table(P, st, q, new)
        else:
            ptr = p.feature(sort[1]).astype(np.int64)[p.fpos] - 1
            cb = p.code["codebit"][np.maximum(ptr, 0), p.fdepth]
            cn = p.code["codenode"][np.maximum(ptr, 0), p.fdepth]
            sign = np.where((ptr >= 0) & (cn == p.fnode) & (cb == 1) & ~p.fobj, -1.0, 1.0).astype(np.float32)
            rel = st * sign
            P = K.fit_table(rel, q, lab, p.fupd, 0, p.Mtrain)
            new = np.empty_like(st)
            K.apply_table(P, rel, q, new)
            new *= sign
        st = new
        p.remember_key(sort[0])
        bpc = 0.0 if all_nodes else K.bits(st, p.fbit, p.Mtrain, p.M) / ntest
        ladder.append(bpc)
        if log:
            log(f"  {len(ladder):2d}. {fmt_sort(sort, form['bounds']):70s} {bpc:.4f}")
    return ladder, st, p


def save_form(form, path):
    with open(path, "wb") as f:
        pickle.dump(form, f)


def load_form(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def determinacy(form, x_train, x_test, static=False):
    """How determinate the picture is before each sign: the entropy, in bits, of what the
    form expects there.  Computed before the sign is read, so it may be used at that sign."""
    _, st, p = run_form(form, x_train, x_test, static=static, all_nodes=True)
    code = form["code"]
    nn, V = code["nnodes"], code["V"]
    S = st.reshape(p.n, nn).astype(np.float64)
    logp = np.zeros((p.n, V))
    for s in range(V):
        for d in range(code["depth"][s]):
            z = S[:, code["codenode"][s, d]]
            z = z if code["codebit"][s, d] else -z
            logp[:, s] -= np.log1p(np.exp(-z))
    pr = np.exp(logp)
    return -(pr * logp).sum(1) / np.log(2), pr.sum(1)


def open_positions(H):
    """An object ends where the picture stops becoming more determinate: position t opens a
    new object when the picture is less determinate before t than it was before t-1."""
    o = np.zeros(len(H), np.bool_)
    o[0] = True
    o[1:] = H[1:] > H[:-1]
    return o
