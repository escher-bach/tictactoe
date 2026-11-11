"""The second level of the tower: objects, their forms, and facts about objects.

THEORY 2.1 (P1): an object is simple; nothing is true of it but which combinations
it *can* enter (its form) and which it *does* enter.  This module takes that literally.

  object        a completed run of signs between boundary signs (the boundary class is the
                one the learning rule chose at the level below)
  form          for an object: which sample objects it is attested next to, more often
                than chance (types over witnesses, THEORY 5.6), and the determinables of
                its own parts (its last signs, its length)
  determinable  a two-valued test on forms.  Objects are tabulated (M3) into a tree of
                such determinables by any-split with a floor (SOLUTIONS Theorem 1'')
  refinement    the witnesses of the next round are the classes of this one
                (SOLUTIONS Proposition 4), so forms stop being sparse
  object fact   at the position where an object begins, the picture states the object's
                determinables, one two-valued fact each, before any of its signs

An object the corpus never showed has an empty combinatorial form, but its own parts
are still there, so it still gets determinate values.
"""
import numpy as np
from numba import njit


@njit(cache=True)
def tokenize(x, isb):
    """Objects of the text: maximal runs of non-boundary signs."""
    n = x.shape[0]
    T = 0
    for t in range(n):
        if not isb[x[t]] and (t == 0 or isb[x[t - 1]]):
            T += 1
    start = np.empty(T, np.int64)
    end = np.empty(T, np.int64)
    h = np.empty(T, np.uint64)
    run_tok = np.full(n, -1, np.int64)       # the object in progress at (or just ended before) t
    k = -1
    cur = np.uint64(1)
    for t in range(n):
        if not isb[x[t]]:
            if t == 0 or isb[x[t - 1]]:
                k += 1
                start[k] = t
                cur = np.uint64(1)
            cur = (cur * np.uint64(0x100000001B3) + np.uint64(x[t] + 1)) * np.uint64(0x9E3779B97F4A7C15)
            cur ^= cur >> np.uint64(32)
            end[k] = t + 1
            h[k] = cur
            run_tok[t] = k
        elif t > 0 and not isb[x[t - 1]]:
            run_tok[t] = k
    return start, end, h, run_tok


class Tower:
    """Objects of a text and their tabulation into determinables."""

    def __init__(self, x, V, boundary, ntrain, cmin=5, n_witness=150, eps=0.02, floor=0.0004,
                 min_gain=0.01, max_depth=16, rounds=2, log=None):
        self.x, self.V, self.boundary, self.ntrain = x, V, frozenset(boundary), ntrain
        isb = np.zeros(V, np.bool_)
        isb[list(boundary)] = True
        self.start, self.end, h, self.run_tok = tokenize(x, isb)
        self.T = len(h)
        uniq, first, self.type_of = np.unique(h, return_index=True, return_inverse=True)
        self.NT = len(uniq)
        self.rep = first                                    # one occurrence of each object
        self.in_train = self.end <= ntrain
        self.count = np.bincount(self.type_of[self.in_train], minlength=self.NT).astype(np.float64)
        self.known = self.count >= cmin                      # objects that have a record
        self.cmin, self.eps, self.floor, self.max_depth, self.min_gain = cmin, eps, floor, max_depth, min_gain
        self.log = log or (lambda s: None)
        self._pairs()
        own_names, own = self._own_tests()
        wit = np.argsort(-self.count)[:n_witness]
        names, tests = self._witness_tests(wit, [self.word(w) for w in wit])
        ctx = self._contexts(wit, n_witness)
        self.rounds = []
        self._tabulate(names, tests, own_names, own, ctx)
        for r in range(1, rounds):                           # refinement: classes as witnesses
            cnames, ctests = self._class_tests()
            ctx = self._contexts(None, int(self.leaf.max()) + 1)
            self._tabulate(names + cnames, np.concatenate([tests, ctests], 1), own_names, own, ctx)

    def _contexts(self, wit, K):
        """What each object stands next to: counts of its right and left neighbours, by
        witness (first round) or by class (later rounds).  Used only to choose splits."""
        if wit is not None:
            idx = np.full(self.NT, K, np.int64)
            idx[wit] = np.arange(K)
            K += 1
        else:
            idx = self.leaf
        R = np.zeros((self.NT, K), np.float32)
        L = np.zeros((self.NT, K), np.float32)
        np.add.at(R, (self.left, idx[self.right]), 1)
        np.add.at(L, (self.right, idx[self.left]), 1)
        return np.concatenate([R, L], 1), K

    # -- strings -------------------------------------------------------------
    def word(self, typ):
        t = self.rep[typ]
        return "".join(chr(97 + s) if s < 26 else "_" for s in self.x[self.start[t]:self.end[t]]) \
            if self.V == 27 else " ".join(str(int(s)) for s in self.x[self.start[t]:self.end[t]])

    # -- what each object does combine with ---------------------------------
    def _pairs(self):
        ty = self.type_of
        ok = self.in_train[1:] & self.in_train[:-1]
        self.left, self.right = ty[:-1][ok], ty[1:][ok]      # consecutive objects of the corpus
        self.npairs = len(self.left)

    def _attested(self, c, cnt_s, cnt_w):
        """Attested beyond chance: at least twice, and more often than independence gives."""
        return (c >= 2) & (c * float(self.npairs) > cnt_s[:, None] * cnt_w[None, :])

    def _witness_tests(self, wit, labels):
        K = len(wit)
        idx = np.full(self.NT, -1, np.int64)
        idx[wit] = np.arange(K)
        after = np.zeros((K, self.NT), np.float32)           # after[s, w]: s then w
        before = np.zeros((K, self.NT), np.float32)          # before[s, w]: w then s
        m = idx[self.left] >= 0
        np.add.at(after, (idx[self.left[m]], self.right[m]), 1)
        m = idx[self.right] >= 0
        np.add.at(before, (idx[self.right[m]], self.left[m]), 1)
        cs = self.count[wit]
        A = self._attested(after, cs, self.count) & self.known[None, :]
        B = self._attested(before, cs, self.count) & self.known[None, :]
        names = [f"after '{w}'" for w in labels] + [f"before '{w}'" for w in labels]
        return names, np.concatenate([A, B], 0).T

    def _class_tests(self):
        """Witnesses of the next round: the classes of this one."""
        leaf = self.leaf
        C = int(leaf.max()) + 1
        after = np.zeros((C, self.NT), np.float32)
        before = np.zeros((C, self.NT), np.float32)
        np.add.at(after, (leaf[self.left], self.right), 1)
        np.add.at(before, (leaf[self.right], self.left), 1)
        cs = np.bincount(leaf, weights=self.count, minlength=C)
        A = self._attested(after, cs, self.count) & self.known[None, :]
        B = self._attested(before, cs, self.count) & self.known[None, :]
        lab = [self.describe_class(c, 3) for c in range(C)]
        names = [f"after class {{{w}}}" for w in lab] + [f"before class {{{w}}}" for w in lab]
        return names, np.concatenate([A, B], 0).T

    # -- the determinables of an object's own parts ----------------------------
    def _own_tests(self):
        x, V = self.x, self.V
        s, e = self.start[self.rep], self.end[self.rep]
        length = e - s
        last = x[e - 1]
        last2 = np.where(length >= 2, x[np.maximum(e - 2, 0)] * V + last, -1)
        last3 = np.where(length >= 3, x[np.maximum(e - 3, 0)] * V * V + last2, -1)
        names, cols = [], []

        def sign(v):
            return chr(97 + v) if (V == 27 and v < 26) else ("_" if V == 27 else str(v))

        for v in range(V):
            if self.count[last == v].sum() > 0:
                names.append(f"ends in {sign(v)}")
                cols.append(last == v)
        for arr, k, top in ((last2, 2, 40), (last3, 3, 30)):
            vals, inv = np.unique(arr, return_inverse=True)
            mass = np.bincount(inv, weights=self.count)
            for i in np.argsort(-mass)[:top]:
                if vals[i] < 0 or mass[i] == 0:
                    continue
                v = int(vals[i])
                txt = "".join(sign((v // V ** j) % V) for j in range(k - 1, -1, -1))
                names.append(f"ends in {txt}")
                cols.append(arr == vals[i])
        for L in range(2, 10):
            names.append(f"has at least {L} signs")
            cols.append(length >= L)
        return names, np.stack(cols, 1)

    # -- tabulation --------------------------------------------------------
    def _tabulate(self, names, tests, own_names, own, ctx):
        """Objects into a tree of two-valued determinables (THEORY 7.2, M3).

        The first determinable is occupancy: does the object have a record?  Below it,
        objects with a record are divided by tests on their form, and objects without
        one by tests on their own parts.  WITNESSES Theorem 1 leaves the choice among
        splitting tests free; here it is the test whose two sides stand next to the most
        different things (an exact count: the information the split carries about the
        neighbours).  A class is left alone below the floor, or when no test tells it apart
        from chance by the margin (SOLUTIONS Theorems 1'' and 3-eps).
        """
        C, K = ctx
        w = self.count
        total = w.sum()
        NT = self.NT
        fam = [(names, tests), (own_names, own)]
        self.names = ["has a record"] + names + own_names
        offs = [1, 1 + len(names)]
        nodes, leaves = [], []
        leaf = np.zeros(NT, np.int64)
        depth = np.zeros(NT, np.int64)
        pnode = np.full((NT, self.max_depth), -1, np.int64)
        pbit = np.zeros((NT, self.max_depth), np.uint8)

        def info(P1, tot):
            """Bits the split carries about one side's neighbours, for every test at once."""
            P0 = tot[None, :] - P1
            m1, m0 = P1.sum(1, keepdims=True), P0.sum(1, keepdims=True)
            m = m1 + m0
            out = np.zeros(P1.shape[0])
            for Pb, mb in ((P1, m1), (P0, m0)):
                with np.errstate(divide="ignore", invalid="ignore"):
                    t = Pb * np.log2(Pb * m / (mb * tot[None, :]))
                out += np.where(Pb > 0, t, 0).sum(1)
            return out

        def split(members, d, f):
            m = w[members].sum()
            if d < self.max_depth and m >= self.floor * total and len(members) > 1:
                tf = fam[f][1][members]
                live = w[members] > 0
                T = tf[live].astype(np.float32)
                Cm = C[members][live]
                share = (w[members][live].astype(np.float32) @ T) / m
                P = T.T @ Cm                                      # tests x contexts
                tot = Cm.sum(0)
                gain = info(P[:, :K], tot[:K]) + info(P[:, K:], tot[K:])
                gain[(share < self.eps) | (share > 1 - self.eps)] = -1
                s = int(np.argmax(gain))
                if gain[s] >= self.min_gain * m:
                    return commit(members, d, f, tf[:, s], offs[f] + s, gain[s] / m)
            leaf[members] = len(leaves)
            leaves.append((d, m))
            return ("leaf", len(leaves) - 1)

        def commit(members, d, f, side, name, g):
            me = len(nodes)
            nodes.append([name, d, float(w[members].sum()), None, None, float(g)])
            pnode[members, d] = me
            pbit[members, d] = side
            depth[members] = d + 1
            f1 = f if f is not None else 0
            f0 = f if f is not None else 1
            nodes[me][4] = split(members[side], d + 1, f1)
            nodes[me][3] = split(members[~side], d + 1, f0)
            return ("node", me)

        commit(np.arange(NT), 0, None, self.known, 0, 0.0)
        self.nodes, self.leaves = nodes, leaves
        self.leaf, self.depth, self.pnode, self.pbit = leaf, depth, pnode, pbit
        nn = len(nodes)
        cid = np.empty((NT, self.max_depth + 1), np.int64)        # class at depth d: node reached, or leaf
        for d in range(self.max_depth + 1):
            cid[:, d] = np.where(d < depth, pnode[:, min(d, self.max_depth - 1)] + 1, nn + 1 + leaf)
        self.classid = cid
        self.rounds.append(dict(classes=len(leaves), determinables=nn))
        self.log(f"tabulated {int((self.count > 0).sum())} objects into {len(leaves)} classes "
                 f"with {nn} determinables")

    # -- reading ---------------------------------------------------------------
    def describe_class(self, c, k=12):
        mem = np.flatnonzero((self.leaf == c) & (self.count > 0))
        mem = mem[np.argsort(-self.count[mem])][:k]
        return " ".join(self.word(m) for m in mem)

    def class_mass(self, c):
        return float(self.count[self.leaf == c].sum() / self.count.sum())

    def path_of_class(self, c):
        typ = int(np.flatnonzero(self.leaf == c)[0])
        return [(self.names[self.nodes[self.pnode[typ, d]][0]], int(self.pbit[typ, d]))
                for d in range(self.depth[typ])]

    def members_at(self, typ, d, k=10):
        """The class an object belongs to at depth d of the tree: its heaviest members."""
        mem = np.flatnonzero((self.classid[:, d] == self.classid[typ, d]) & (self.count > 0))
        return [self.word(m) for m in mem[np.argsort(-self.count[mem])][:k]], float(self.count[mem].sum() / self.count.sum())
