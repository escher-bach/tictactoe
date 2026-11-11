"""Nominal analysis: learn the canonical machine of a law from texts (FOUNDATIONS 7, 8).

What is supplied: order, sameness of sign, and counts.  A sign that recurs in a fixed
share of the texts is a universal and is read as itself; any other sign is a name, and
all that is read of a name is which earlier positions carry it (FOUNDATIONS 2.3, 3.2).

A state is a class of pasts whose futures agree up to renaming.  Its memory is the
least set of names the future depends on.  Both are decided by counts:
  forgetting  a remembered name is dropped when it is never referred to again
  identity    two pasts are one state when some matching of their memories makes the
              laws of their next few signs agree (a chi-square test of homogeneity)
"""
import itertools
from collections import Counter, defaultdict

from scipy.stats import chi2

from .worlds import END


def universals(texts, share=0.01):
    """Signs that recur in at least `share` of the texts."""
    c = Counter()
    for t in texts:
        c.update(set(t))
    return {x for x, k in c.items() if k >= share * len(texts)}


def pattern(text, t, mem, D, closed):
    """The next D signs as the picture sees them: a universal, a remembered name by its
    place in the memory, or a name outside it (numbered by first appearance)."""
    idx = {a: i for i, a in enumerate(mem)}
    new, out = {}, []
    for x in text[t:t + D]:
        if x in closed:
            out.append(("c", x))
        elif x in idx:
            out.append(("r", idx[x]))
        else:
            out.append(("n", new.setdefault(x, len(new))))
    return tuple(out)


def relabel(stats, beta):
    out = Counter()
    for pat, k in stats.items():
        out[tuple(("r", beta[s[1]]) if s[0] == "r" else s for s in pat)] += k
    return out


def homogeneous(A, B, alpha):
    """Are two samples of patterns from the same law?  Chi-square, sparse cells pooled."""
    nA, nB = sum(A.values()), sum(B.values())
    rows, pool = [], [0, 0]
    for k in set(A) | set(B):
        a, b = A.get(k, 0), B.get(k, 0)
        if (a + b) * min(nA, nB) / (nA + nB) < 5:
            pool[0] += a
            pool[1] += b
        else:
            rows.append((a, b))
    if pool != [0, 0]:
        rows.append(tuple(pool))
    if len(rows) < 2:
        return True
    stat = 0.0
    for a, b in rows:
        ea, eb = (a + b) * nA / (nA + nB), (a + b) * nB / (nA + nB)
        stat += (a - ea) ** 2 / ea + (b - eb) ** 2 / eb
    return chi2.sf(stat, len(rows) - 1) > alpha


class Machine:
    def __init__(self, closed):
        self.closed = closed
        self.arity = [0]                      # memory size of each state
        self.trans = [dict()]                 # trans[s][symbol] = (next state, which raw slots are kept, in order)

    def symbol(self, x, mem):
        if x in self.closed:
            return ("c", x)
        return ("r", mem.index(x)) if x in mem else ("n",)

    def run(self, text):
        """States and memories before each sign, as far as the machine is defined."""
        s, mem, out = 0, [], []
        for x in text:
            out.append((s, list(mem)))
            sym = self.symbol(x, mem)
            if sym not in self.trans[s]:
                return out, sym
            raw = mem + [x] if sym == ("n",) else mem
            s, keep = self.trans[s][sym]
            mem = [raw[j] for j in keep]
        return out, None


def learn(texts, D=3, alpha=1e-3, min_count=40, share=0.01, log=None):
    closed = universals(texts, share)
    M = Machine(closed)
    while True:
        stats = [Counter() for _ in M.arity]
        pending = defaultdict(list)
        for d, text in enumerate(texts):
            path, stop = M.run(text)
            for t, (s, mem) in enumerate(path):
                stats[s][pattern(text, t, mem, D, closed)] += 1
            if stop is not None:
                t = len(path) - 1
                s, mem = path[-1]
                raw = mem + [text[t]] if stop == ("n",) else mem
                pending[(s, stop)].append((d, t + 1, raw))
        pending = {k: v for k, v in pending.items() if len(v) >= min_count}
        if not pending:
            return M, stats
        key = max(pending, key=lambda k: len(pending[k]))
        occ = pending[key]
        if key[1] == ("c", END):                                   # nothing follows the end of a text
            M.trans[key[0]][key[1]] = (0, ())
            continue
        r_raw = len(occ[0][2])
        # forgetting: a name that is never referred to again is not part of the state
        live = [j for j in range(r_raw) if any(raw[j] in texts[d][t:] for d, t, raw in occ)]
        new = Counter(pattern(texts[d], t, [raw[j] for j in live], D, closed) for d, t, raw in occ)
        target = None
        for s2 in range(len(M.arity)):
            if M.arity[s2] != len(live) or sum(stats[s2].values()) < min_count:
                continue
            for beta in itertools.permutations(range(len(live))):  # a matching of the two memories
                if homogeneous(relabel(new, beta), stats[s2], alpha):
                    inv = [beta.index(k) for k in range(len(live))]
                    target = (s2, tuple(live[inv[k]] for k in range(len(live))))
                    break
            if target:
                break
        if target is None:
            M.arity.append(len(live))
            M.trans.append(dict())
            target = (len(M.arity) - 1, tuple(live))
        M.trans[key[0]][key[1]] = target
        if log:
            log(f"  state {key[0]} reading {key[1]} -> state {target[0]}, memory {len(target[1])}"
                f" ({len(occ)} occurrences)")


def symmetries(M, texts, s, D=3, alpha=1e-3):
    """Rearrangements of a state's memory that leave its law unchanged: is the memory a
    tuple, a set, or something between?  Tested on two disjoint halves of the texts."""
    halves = [Counter(), Counter()]
    for d, text in enumerate(texts):
        path, _ = M.run(text)
        for t, (st, mem) in enumerate(path):
            if st == s:
                halves[d % 2][pattern(text, t, mem, D, M.closed)] += 1
    return [b for b in itertools.permutations(range(M.arity[s]))
            if b != tuple(range(M.arity[s])) and homogeneous(relabel(halves[0], b), halves[1], alpha)]
