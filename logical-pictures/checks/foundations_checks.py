"""Exact checks of the statements in FOUNDATIONS.md on small cases, by brute force.
These are checks of lemmas.  Nothing here is an experiment on data.
Run: python3 checks/foundations_checks.py"""
import itertools
import math
import random

RNG = random.Random(11)
FAILED = []


def report(name, ok, detail):
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")
    if not ok:
        FAILED.append(name)


def set_partitions(items):
    if not items:
        yield []
        return
    first, rest = items[0], items[1:]
    for p in set_partitions(rest):
        yield [[first]] + p
        for i in range(len(p)):
            yield p[:i] + [[first] + p[i]] + p[i + 1:]


def young(blocks, V):
    """All permutations of range(V) that map every block to itself."""
    out = [dict()]
    for b in blocks:
        out = [{**g, **dict(zip(b, q))} for g in out for q in itertools.permutations(b)]
    return [tuple(g[a] for a in range(V)) for g in out]


def invariant_law(blocks, V, n):
    """A random law on strings of length n, invariant under the Young subgroup of `blocks`."""
    strings = list(itertools.product(range(V), repeat=n))
    w = {x: RNG.random() ** 3 for x in strings}
    G = young(blocks, V)
    P = {x: sum(w[tuple(g[c] for c in x)] for g in G) for x in strings}
    Z = sum(P.values())
    return {x: v / Z for x, v in P.items()}, strings


def swap(x, a, b):
    return tuple(b if c == a else a if c == b else c for c in x)


def kinds_of(P, strings, V):
    """The relation 'the law is unchanged by exchanging a and b', as a partition."""
    cls = list(range(V))
    rel = {(a, b): all(abs(P[x] - P[swap(x, a, b)]) < 1e-12 for x in strings)
           for a in range(V) for b in range(V)}
    transitive = all(not (rel[a, b] and rel[b, c]) or rel[a, c]
                     for a in range(V) for b in range(V) for c in range(V))
    for a in range(V):
        for b in range(a):
            if rel[a, b]:
                cls[a] = cls[b]
                break
    blocks = {}
    for a in range(V):
        blocks.setdefault(cls[a], []).append(a)
    return sorted(blocks.values()), transitive


def tau(x, kind):
    """The kinded pattern of a string: the kind at each position and which positions agree."""
    seen = {}
    return tuple((kind[c], seen.setdefault(c, len(seen))) for c in x)


def check_F1():
    """Kinds are canonical: exchange-invariance is an equivalence, and its partition is the
    coarsest one whose Young subgroup preserves the law."""
    ok = True
    n_cases = 0
    for _ in range(25):
        V, n = 5, 3
        blocks = RNG.choice(list(set_partitions(list(range(V)))))
        P, strings = invariant_law(blocks, V, n)
        star, transitive = kinds_of(P, strings, V)
        ok &= transitive
        ok &= all(any(set(b) <= set(s) for s in star) for b in blocks)        # B refines B*
        for part in set_partitions(list(range(V))):
            pres = all(abs(P[x] - P[tuple(g[c] for c in x)]) < 1e-12 for g in young(part, V) for x in strings)
            refines = all(any(set(b) <= set(s) for s in star) for b in part)
            ok &= pres == refines
        n_cases += 1
    report("F1 kinds are canonical", ok,
           f"{n_cases} random invariant laws on 5 signs: exchange-invariance is transitive, and a partition's "
           f"Young subgroup preserves the law iff the partition refines the kinds (all 52 partitions tried)")


def check_F2_F3():
    """The quotient theorem and the pointer form."""
    okq = okp = True
    cases = 0
    for _ in range(20):
        V, n = 5, 4
        blocks = RNG.choice([p for p in set_partitions(list(range(V))) if len(p) < V])
        P, strings = invariant_law(blocks, V, n)
        kind = {a: i for i, b in enumerate(blocks) for a in b}
        size = [len(b) for b in blocks]
        Q = {}
        for x in strings:
            Q[tau(x, kind)] = Q.get(tau(x, kind), 0) + P[x]
        for x in strings:
            used = [len({c for c in x if kind[c] == i}) for i in range(len(blocks))]
            orbit = math.prod(math.prod(size[i] - j for j in range(used[i])) for i in range(len(blocks)))
            okq &= abs(P[x] - Q[tau(x, kind)] / orbit) < 1e-12
        # pointer form: the next sign either repeats an earlier name or is a uniform new name of its kind,
        # and the weights depend on the past only through its kinded pattern
        table = {}
        for t in range(1, n):
            marg = {}
            for x in strings:
                marg[x[:t + 1]] = marg.get(x[:t + 1], 0) + P[x]
            for past in itertools.product(range(V), repeat=t):
                tot = sum(marg[past + (a,)] for a in range(V))
                if tot < 1e-15:
                    continue
                cond = [marg[past + (a,)] / tot for a in range(V)]
                seen = {}
                for c in past:
                    seen.setdefault(c, len(seen))
                rep = tuple(cond[c] for c in seen)                              # weight of each earlier name
                new = []
                for i, b in enumerate(blocks):
                    fresh = [cond[a] for a in b if a not in seen]
                    okp &= max(fresh, default=0) - min(fresh, default=0) < 1e-12  # uniform over unused names
                    new.append(round(sum(fresh), 12))
                key = (t, tau(past, kind))
                val = (tuple(round(v, 10) for v in rep), tuple(round(v, 10) for v in new))
                okp &= table.setdefault(key, val) == val
        cases += 1
    report("F2 quotient theorem", okq,
           f"{cases} invariant laws: P(x) = Q(kinded pattern of x) / product over kinds of falling factorials")
    report("F3 pointer form", okp,
           f"{cases} invariant laws: the next sign repeats an earlier name or is a uniform new name of its "
           f"kind, with weights that depend on the past only through its kinded pattern")


def check_F4():
    """Rank does not police literal memory: the value tests on J offsets have tree rank <= J."""
    from functools import lru_cache
    ok = True
    for _ in range(40):
        J, V, n = RNG.randint(1, 3), RNG.randint(2, 3), RNG.randint(4, 12)
        ctx = [tuple(RNG.randrange(V) for _ in range(J)) for _ in range(n)]
        tests = [[int(c[j] == v) for c in ctx] for j in range(J) for v in range(V)]

        @lru_cache(maxsize=None)
        def ldim(A):
            best = 0
            for s in tests:
                As = frozenset(a for a in A if s[a])
                if As and A - As:
                    best = max(best, 1 + min(ldim(As), ldim(A - As)))
            return best
        ok &= ldim(frozenset(range(n))) <= J
    report("F4 literal windows have small rank", ok, "40 random sets of windows of J signs: tree rank <= J")


if __name__ == "__main__":
    check_F1()
    check_F2_F3()
    check_F4()
    print("ALL PASS" if not FAILED else f"FAILED: {FAILED}")
