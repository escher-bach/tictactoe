"""Exact checks of the statements in SOLUTIONS.md on small instances.

Each check compares a theorem with brute force.  They are checks of lemmas,
not experiments on data.  Run:  python3 checks/solutions_checks.py
Everything is seeded; the output is in checks/OUTPUT.txt.
"""
import itertools
import math
import random
import sys
from functools import lru_cache

sys.setrecursionlimit(100000)
RNG = random.Random(20261111)
FAILED = []


def report(name, ok, detail):
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")
    if not ok:
        FAILED.append(name)


# ---------------------------------------------------------------- helpers
def strahler(tree):
    """Rank of a binary tree given as ('leaf', ...) or ('node', s, left, right)."""
    if tree[0] == "leaf":
        return 0
    a, b = strahler(tree[-2]), strahler(tree[-1])
    return a + 1 if a == b else max(a, b)


def leaves(tree):
    if tree[0] == "leaf":
        return [tree[1]]
    return leaves(tree[-2]) + leaves(tree[-1])


def make_ldim(R):
    """Brute-force tree rank (Littlestone dimension) of contexts over tests."""
    m = len(R[0])

    @lru_cache(maxsize=None)
    def ldim(A):
        best = 0
        for s in range(m):
            As = frozenset(a for a in A if R[a][s])
            B = A - As
            if As and B:
                best = max(best, 1 + min(ldim(As), ldim(B)))
        return best

    return ldim


def is_good(R, A, eps, w=None):
    """A is eps-good: no test eps-splits it (weights optional)."""
    tot = len(A) if w is None else sum(w[a] for a in A)
    for s in range(len(R[0])):
        k = sum((1 if w is None else w[a]) for a in A if R[a][s])
        if eps * tot <= k <= (1 - eps) * tot:
            return False
    return True


def generic_type(R, A):
    return tuple(1 if 2 * sum(R[a][s] for a in A) > len(A) else 0 for s in range(len(R[0])))


def any_split(R, X, eps, rng, floor=0):
    """The any-split rule with a size floor; the splitting test is chosen at random."""
    m = len(R[0])

    def rec(A):
        if len(A) < floor:
            return ("leaf", A)
        cands = []
        for s in range(m):
            k = sum(R[a][s] for a in A)
            if eps * len(A) <= k <= (1 - eps) * len(A) and 0 < k < len(A):
                cands.append(s)
        if not cands:
            return ("leaf", A)
        s = rng.choice(cands)
        As = frozenset(a for a in A if R[a][s])
        return ("node", s, rec(A - As), rec(As))

    return rec(frozenset(X))


def low_rank_relation(n, rng):
    """A random relation of small tree rank: a few Satzsysteme, one scale, sparse noise."""
    cols = []
    for _ in range(rng.randint(1, 2)):                 # determinables (rank 1 each)
        v = rng.randint(2, 4)
        val = [rng.randrange(v) for _ in range(n)]
        cols += [[int(val[a] == j) for a in range(n)] for j in range(v)]
    if rng.random() < 0.5:                              # one scale
        v = rng.randint(2, 5)
        val = [rng.randrange(v) for _ in range(n)]
        cols += [[int(val[a] >= j) for a in range(n)] for j in range(1, v)]
    R = [[c[a] for c in cols] for a in range(n)]
    for a in range(n):                                  # a few exceptions
        if rng.random() < 0.05:
            s = rng.randrange(len(cols))
            R[a][s] ^= 1
    return R


# ---------------------------------------------------------------- S1
def check_S1():
    """Theorem 3-eps: peeling with a margin never backtracks under hereditary purity."""
    n, m, r_max = 400, 6, 2
    theta, n0, eta = 0.2, 25, 0.04
    trials = held = succeeded = 0
    worst_slack = 1e9
    for _ in range(150):
        R = [[RNG.randrange(2) for _ in range(m)] for _ in range(n)]

        def rand_tree(rank, tests):
            if rank == 0 or not tests or RNG.random() < 0.15:
                return ("leaf", RNG.randrange(2))
            s = RNG.choice(tests)
            rest = [x for x in tests if x != s]
            a = rand_tree(rank - 1, rest)
            b = rand_tree(rank, rest)
            return ("node", s, a, b) if RNG.random() < 0.5 else ("node", s, b, a)

        T = rand_tree(r_max, list(range(m)))

        def ev(t, a):
            while t[0] == "node":
                t = t[3] if R[a][t[1]] else t[2]
            return t[1]

        truth = [ev(T, a) for a in range(n)]
        y = [v ^ (RNG.random() < eta) for v in truth]
        r = strahler(T)

        def leaf_ok(A):
            if len(A) < n0:
                return True
            k = sum(y[a] for a in A)
            return min(k, len(A) - k) <= theta * len(A)

        # brute-force hypothesis: every cell on which T is constant passes the leaf test
        hyp = True
        for lits in itertools.product((0, 1, 2), repeat=m):       # 2 = test absent
            C = [a for a in range(n) if all(l == 2 or R[a][s] == l for s, l in enumerate(lits))]
            if C and len({truth[a] for a in C}) == 1 and not leaf_ok(C):
                hyp = False
                break
        trials += 1
        if not hyp:
            continue
        held += 1
        for _rep in range(3):                                      # arbitrary choices
            calls = [0]

            def find(A, rk):
                calls[0] += 1
                if leaf_ok(A):
                    return ("leaf", A)
                if rk == 0:
                    return None
                order = list(range(m))
                RNG.shuffle(order)
                for s in order:
                    As = [a for a in A if R[a][s]]
                    B = [a for a in A if not R[a][s]]
                    if not As or not B:
                        continue
                    sides = [(As, B), (B, As)]
                    RNG.shuffle(sides)
                    for side, other in sides:
                        sub = find(side, rk - 1)
                        if sub is not None:
                            rest = find(other, rk)                 # committed: never revisited
                            return None if rest is None else ("node", s, sub, rest)
                return None

            out = find(list(range(n)), r)
            if out is None:
                report("S1 peel-with-margin", False, "failed although the hypothesis held")
                return
            L = leaves(out)
            assert all(leaf_ok(A) for A in L) and strahler(out) <= r
            err = sum(min(sum(y[a] for a in A), len(A) - sum(y[a] for a in A)) for A in L)
            bound = theta * n + n0 * len(L)
            worst_slack = min(worst_slack, bound - err)
            succeeded += 1
    report("S1 peel-with-margin", held > 20 and worst_slack >= 0,
           f"{trials} instances, hypothesis held in {held}; {succeeded} runs with random "
           f"choices all succeeded without revisiting a committed test, rank<=r, "
           f"error<=theta*n+n0*leaves (min slack {worst_slack:.1f})")

    # the probabilistic sufficient condition: n0 >= (m ln 3 + ln(1/delta)) / (2 gamma^2)
    m2, gamma, delta, eta2 = 4, 0.2, 0.1, 0.05
    n0b = math.ceil((m2 * math.log(3) + math.log(1 / delta)) / (2 * gamma * gamma))
    n2, ok_count, T2 = 1200, 0, 120
    for _ in range(T2):
        R = [[RNG.randrange(2) for _ in range(m2)] for _ in range(n2)]
        truth = [R[a][0] ^ (R[a][1] & R[a][2]) for a in range(n2)]
        y = [v ^ (RNG.random() < eta2) for v in truth]
        good = True
        for lits in itertools.product((0, 1, 2), repeat=m2):
            C = [a for a in range(n2) if all(l == 2 or R[a][s] == l for s, l in enumerate(lits))]
            if len(C) >= n0b and len({truth[a] for a in C}) == 1:
                k = sum(y[a] for a in C)
                if min(k, len(C) - k) > (eta2 + gamma) * len(C):
                    good = False
                    break
        ok_count += good
    report("S1 noise condition", ok_count >= (1 - delta) * T2,
           f"n0={n0b}: hereditary purity held in {ok_count}/{T2} noisy samples (bound says >= {1-delta:.0%})")


# ---------------------------------------------------------------- S2
def check_S2():
    """Uniqueness: closure, overlap, predictive uniqueness, the counterexample."""
    eps = 0.2
    pairs = unions = 0
    for _ in range(60):
        n, m = 10, RNG.randint(2, 4)
        R = [[RNG.random() < 0.5 for _ in range(m)] for _ in range(n)]
        good = [frozenset(A) for k in range(1, n + 1) for A in itertools.combinations(range(n), k)
                if is_good(R, A, eps)]
        gt = {A: generic_type(R, A) for A in good}
        for A, B in itertools.combinations(good, 2):
            if gt[A] != gt[B]:
                pairs += 1
                if not len(A & B) < eps * (len(A) + len(B)):
                    report("S2 overlap bound", False, f"{sorted(A)} {sorted(B)}")
                    return
            elif not (A & B):
                unions += 1
                if not is_good(R, A | B, eps):
                    report("S2 union closure", False, f"{sorted(A)} {sorted(B)}")
                    return
    report("S2 union closure", True, f"{unions} disjoint pairs of good sets with equal generic type: union good")
    report("S2 overlap bound", True, f"{pairs} pairs of good sets with different generic type: |A&B| < eps(|A|+|B|)")

    # predictive uniqueness: two any-split runs, arbitrary function of k tests
    worst = 0.0
    runs = 0
    for _ in range(200):
        n = 120
        R = low_rank_relation(n, RNG)
        m = len(R[0])
        e = 0.08
        t1, t2 = any_split(R, range(n), e, RNG), any_split(R, range(n), e, RNG)
        if not all(is_good(R, A, e) for A in leaves(t1) + leaves(t2)):
            report("S2 predictive uniqueness", False, "any-split left a class that is not good")
            return
        k = RNG.randint(1, min(3, m))
        S = RNG.sample(range(m), k)
        table = {bits: RNG.randrange(2) for bits in itertools.product((0, 1), repeat=k)}
        y = [table[tuple(int(R[a][s]) for s in S)] for a in range(n)]

        def predict(tree):
            p = [0] * n
            for A in leaves(tree):
                maj = int(2 * sum(y[a] for a in A) > len(A))
                for a in A:
                    p[a] = maj
            return p

        p1, p2 = predict(t1), predict(t2)
        d = sum(a != b for a, b in zip(p1, p2))
        e1 = sum(a != b for a, b in zip(p1, y))
        if not (e1 < k * e * n or e1 == 0) or not (d < 2 * k * e * n or d == 0):
            report("S2 predictive uniqueness", False, f"d={d} e1={e1} k={k}")
            return
        worst = max(worst, d / (2 * k * e * n))
        runs += 1
    report("S2 predictive uniqueness", True,
           f"{runs} pairs of any-split runs: predictions of a random k-test function differ on "
           f"< 2k*eps*n contexts (largest ratio to the bound {worst:.2f})")

    # exact case: with "no test has both sides non-empty", the coarsest good partition is the type partition
    ok0 = True
    for _ in range(40):
        n, m = 8, RNG.randint(1, 3)
        R = [[RNG.randrange(2) for _ in range(m)] for _ in range(n)]
        pure = lambda A: all(len({R[a][s] for a in A}) == 1 for s in range(m))
        types = {}
        for a in range(n):
            types.setdefault(tuple(R[a]), []).append(a)
        ok0 &= all(pure(A) for A in types.values())
        ok0 &= all(not pure(A + B) for A, B in itertools.combinations(types.values(), 2))
    report("S2 exact case", ok0, "type classes are pure and no union of two of them is: the type partition is the coarsest")

    # membership can disagree everywhere: one determinable with 11 equally frequent values, eps = 0.1
    R = [[int(a % 11 == v) for v in range(11)] for a in range(110)]
    whole = is_good(R, range(110), 0.1)
    by_value = all(is_good(R, [a for a in range(110) if a % 11 == v], 0.1) for v in range(11))
    report("S2 membership is not determined", whole and by_value,
           "a determinable with 11 values: the whole set is 0.1-good, and so is every value class")

    # the counterexample: one test, three blocks
    a, b0, b1, c, e = 100, 5, 5, 100, 0.1
    R = [[0]] * a + [[0]] * b0 + [[1]] * b1 + [[1]] * c
    A, B, C = range(0, a), range(a, a + b0 + b1), range(a + b0 + b1, a + b0 + b1 + c)
    P = [list(A) + list(B), list(C)]
    Q = [list(A), list(B) + list(C)]
    ok = all(is_good(R, S, e) for S in P + Q)
    ok &= generic_type(R, P[0]) != generic_type(R, P[1]) and generic_type(R, Q[0]) != generic_type(R, Q[1])
    ok &= not is_good(R, list(B), e) and not is_good(R, list(A) + list(B) + list(C), e)
    report("S2 no canonical partition", ok,
           "rank-1 relation with two reduced good partitions that differ on a block of 10 of 210 contexts")


# ---------------------------------------------------------------- S3
def check_S3():
    """Theorem 1'': class count independent of n, with an exceptional set of mass <= eps."""
    # the arithmetic: mu = eps^2 (eps/4)^(2t)  =>  N*mu <= eps and N <= (8e ln(4/eps)/eps)^t,
    # where N is the largest number of leaves Lemma 2.1 allows at depth D = ln(1/mu)/eps + 1
    ok = True
    for t in range(1, 13):
        for eps in (0.5, 0.4, 0.3, 0.2, 0.1, 0.05, 0.01, 0.001):
            mu = eps ** 2 * (eps / 4) ** (2 * t)
            D = math.log(1 / mu) / eps + 1
            N = sum(math.comb(math.floor(D), i) for i in range(t + 1))
            crude = (math.e * D / t) ** t
            ok &= N <= crude and crude * mu <= eps and crude <= (8 * math.e * math.log(4 / eps) / eps) ** t
    report("S3 arithmetic", ok, "with mu = eps^2 (eps/4)^(2t): N*mu <= eps and N <= (8e ln(4/eps)/eps)^t "
           "on a grid of (eps, t), t <= 12")

    worst_rank_gap, checked = 0, 0
    for _ in range(120):
        n = 14
        R = low_rank_relation(n, RNG)
        ldim = make_ldim(tuple(map(tuple, R)))
        t = ldim(frozenset(range(n)))
        eps = RNG.choice((0.1, 0.2, 0.3))
        tree = any_split(R, range(n), eps, RNG, floor=RNG.choice((0, 2, 3)))
        if strahler(tree) > t:
            report("S3 rank of split tree", False, f"strahler {strahler(tree)} > rank {t}")
            return
        worst_rank_gap = max(worst_rank_gap, t - strahler(tree))
        checked += 1
    report("S3 rank of split tree", True,
           f"{checked} random relations: split tree rank <= exact tree rank of the relation (brute force)")

    ok = True
    for _ in range(40):
        n = RNG.choice((300, 1000, 3000))
        R = low_rank_relation(n, RNG)
        eps = 0.15
        mu = 0.002
        tree = any_split(R, range(n), eps, RNG, floor=math.ceil(mu * n))
        L = leaves(tree)
        bad = sum(len(A) for A in L if not is_good(R, A, eps))
        small = sum(len(A) for A in L if len(A) < mu * n)
        ok &= bad <= small <= mu * n * len(L)
    report("S3 exceptional set", ok, "classes that are not good are exactly unsplit small ones; mass <= mu*n*classes")


# ---------------------------------------------------------------- S4
def check_S4():
    """Paths: refinement computes the types over all g^l paths; any order, one limit."""
    ok_types = ok_confl = True
    for _ in range(200):
        n, g = RNG.randint(5, 9), RNG.randint(1, 3)
        links = [[RNG.choice([None] + list(range(n))) for _ in range(n)] for _ in range(g)]
        base = [RNG.randrange(2) for _ in range(n)]

        def canon(col):
            seen = {}
            return tuple(seen.setdefault(c, len(seen)) for c in col)

        P = canon(base)
        L = RNG.randint(1, 4)
        for _lvl in range(L):
            P = canon([(P[x],) + tuple(None if l[x] is None else P[l[x]] for l in links) for x in range(n)])

        def follow(x, path):
            for i in path:
                if x is None:
                    return None
                x = links[i][x]
            return x

        def col(path):
            return [None if follow(x, path) is None else base[follow(x, path)] for x in range(n)]

        paths = [p for l in range(L + 1) for p in itertools.product(range(g), repeat=l)]
        brute = canon([tuple(col(p)[x] for p in paths) for x in range(n)])
        ok_types &= brute == P

        # confluence: split one class by one link at a time, in random order, until stable
        def run_random():
            Q = list(canon(base))
            while True:
                moves = []
                for i in range(g):
                    for c in set(Q):
                        members = [x for x in range(n) if Q[x] == c]
                        sig = {None if links[i][x] is None else Q[links[i][x]] for x in members}
                        if len(sig) > 1:
                            moves.append((i, c))
                if not moves:
                    return canon(Q)
                i, c = RNG.choice(moves)
                members = [x for x in range(n) if Q[x] == c]
                sig = [None if links[i][x] is None else Q[links[i][x]] for x in members]
                new = max(Q) + 1
                for x, sg in zip(members, sig):
                    if sg != sig[0]:
                        Q[x] = new
        a, b = run_random(), run_random()

        def as_sets(Q):
            return {frozenset(x for x in range(n) if Q[x] == c) for c in set(Q)}
        # limit of the full refinement (the coarsest stable partition)
        S = canon(base)
        while True:
            S2 = canon([(S[x],) + tuple(None if l[x] is None else S[l[x]] for l in links) for x in range(n)])
            if S2 == S:
                break
            S = S2
        ok_confl &= as_sets(a) == as_sets(b) == as_sets(S)
    report("S4 path types by refinement", ok_types,
           "partition after l rounds of refinement = types over all paths of length <= l (200 random link systems)")
    report("S4 confluence for fixed links", ok_confl,
           "single-class splits in random order all end at the coarsest stable partition")

    # recency is not monotone: finer criterion, incomparable link partition
    def rec_feature(P, x):
        """For each t: the sign at the most recent earlier position in the same P-class."""
        out = []
        for t in range(len(x)):
            u = max((u for u in range(t) if P[u] == P[t]), default=None)
            out.append(None if u is None else x[u])
        return out

    def refines(P, Q):
        return all(Q[i] == Q[j] for i in range(len(P)) for j in range(len(P)) if P[i] == P[j])

    found = None
    for n in range(3, 6):
        for x in itertools.product("ab", repeat=n):
            for P in itertools.product(range(2), repeat=n):
                Q = (0,) * n                       # the trivial criterion; P refines it
                fP, fQ = rec_feature(P, x), rec_feature(Q, x)
                PP = [(P[i], fP[i]) for i in range(n)]
                QQ = [(Q[i], fQ[i]) for i in range(n)]
                if not refines(PP, QQ):
                    found = ("".join(x), P)
                    break
            if found:
                break
        if found:
            break
    # order-dependence: two moves, each applied until nothing changes, in the two orders.
    #   sign move:    split classes by the position's own sign
    #   recency move: split classes by the class of the position just before the most recent
    #                 earlier position of the same class
    def sign_move(P, x):
        return canon2(list(zip(P, x)))

    def rec_move(P, x):
        f = []
        for t in range(len(x)):
            u = max((u for u in range(t) if P[u] == P[t]), default=None)
            f.append(None if u is None or u == 0 else P[u - 1])
        return canon2(list(zip(P, f)))

    def canon2(col):
        seen = {}
        return tuple(seen.setdefault(c, len(seen)) for c in col)

    def closure(P, x, moves):
        while True:
            Q = P
            for mv in moves:
                while True:
                    Q2 = mv(Q, x)
                    if Q2 == Q:
                        break
                    Q = Q2
            if Q == P:
                return P
            P = Q
    diff = None
    for n in range(2, 6):
        for x in itertools.product("ab", repeat=n):
            A = closure((0,) * n, x, (sign_move, rec_move))
            B = closure((0,) * n, x, (rec_move, sign_move))
            if A != B:
                diff = ("".join(x), A, B)
                break
        if diff:
            break
    report("S4 recency makes the order matter", diff is not None,
           f"text {diff[0]!r}: sign move first ends at {diff[1]}, recency move first ends at {diff[2]}; "
           f"both are fixed points of both moves")
    report("S4 recency is not monotone", found is not None,
           f"text {found[0]!r}, criterion {found[1]} refines the trivial one, but its recency "
           f"refinement does not refine the trivial criterion's")


# ---------------------------------------------------------------- S5
def tree_rank_of_function(X, tests, y):
    """Least rank of a decision tree over `tests` computing y on X (brute force)."""
    @lru_cache(maxsize=None)
    def rk(A):
        if len({y[a] for a in A}) <= 1:
            return 0
        best = 99
        for s in range(len(tests)):
            As = frozenset(a for a in A if tests[s][a])
            B = A - As
            if As and B:
                p, q = rk(As), rk(B)
                best = min(best, p + 1 if p == q else max(p, q))
        return best
    return rk(frozenset(X))


def check_S5():
    """Signed sums are scales: g(sum w_i x_i) is a decision list (rank 1) over the scale."""
    rows = []
    ok = True
    for k in range(1, 5):
        X = list(itertools.product((0, 1), repeat=k))
        idx = range(len(X))
        raw = [[x[i] for x in X] for i in range(k)]
        par = [sum(x) % 2 for x in X]
        cnt = [sum(x) for x in X]
        scale = [[int(c >= j) for c in cnt] for j in range(1, k + 1)]
        r_raw = tree_rank_of_function(idx, raw, par)
        r_scale = tree_rank_of_function(idx, scale, par)
        ok &= r_raw == k and r_scale <= 1
        rows.append((k, r_raw, r_scale))
    report("S5 parity: raw facts vs count scale", ok,
           "(k, rank over the k facts, rank over thresholds of their count) = " + str(rows))
    ok = True
    for _ in range(300):
        k, W = RNG.randint(2, 4), RNG.randint(1, 3)
        w = [RNG.randint(-W, W) for _ in range(k)]
        X = list(itertools.product((0, 1), repeat=k))
        S = [sum(a * b for a, b in zip(w, x)) for x in X]
        g = {v: RNG.randrange(2) for v in set(S)}
        y = [g[v] for v in S]
        scale = [[int(v >= c) for v in S] for c in sorted(set(S))[1:]]
        if scale:
            ok &= tree_rank_of_function(range(len(X)), scale, y) <= 1
    report("S5 any function of an integer form", ok,
           "rank over the thresholds of the scale <= 1 for 300 random g(sum w_i x_i)")


# ---------------------------------------------------------------- S6
def check_S6():
    """Occupancy: rank <= log2(#realised types); exclusive sorts add ranks by max, not by sum."""
    ok = True
    for _ in range(150):
        n, m = RNG.randint(2, 12), RNG.randint(1, 5)
        R = tuple(tuple(RNG.randrange(2) for _ in range(m)) for _ in range(n))
        t = make_ldim(R)(frozenset(range(n)))
        ok &= t <= math.floor(math.log2(len(set(R))))
    report("S6 rank <= log2(types)", ok, "150 random relations, exact rank by brute force")
    # a product of d determinables (Satzsysteme, any number of values) has rank <= d
    ok = True
    full = []
    for _ in range(60):
        d = RNG.randint(1, 3)
        vals = [RNG.randint(2, 4) for _ in range(d)]
        ctx = [tuple(RNG.randrange(v) for v in vals) for _ in range(RNG.randint(4, 14))]
        if RNG.random() < 0.3:
            ctx = list(itertools.product(*[range(v) for v in vals]))[:14]
        tests = [[int(c[i] == j) for c in ctx] for i in range(d) for j in range(vals[i])]
        R = tuple(tuple(t[a] for t in tests) for a in range(len(ctx)))
        t = make_ldim(R)(frozenset(range(len(ctx))))
        ok &= t <= d
        full.append(t == d)
    report("S6 product of d determinables", ok and any(full),
           "rank <= d for 60 random products of 1..3 determinables with 2..4 values (attained in some)")
    rows = []
    ok = True
    for dims in ((1, 1), (2, 2), (2, 1, 1), (3, 1), (2, 2, 2), (3, 3), (1, 1, 1, 1)):
        # each context names an object of exactly one sort j, with dims[j] free two-valued determinables
        ctx, tests = [], []
        for j, d in enumerate(dims):
            for bits in itertools.product((0, 1), repeat=d):
                ctx.append((j, bits))
        for j, d in enumerate(dims):
            tests.append([int(c[0] == j) for c in ctx])                      # occupancy
            for i in range(d):
                tests.append([int(c[0] == j and c[1][i] == 1) for c in ctx])  # value, false where unoccupied
        R = tuple(tuple(t[a] for t in tests) for a in range(len(ctx)))
        t = make_ldim(R)(frozenset(range(len(ctx))))
        hi = math.floor(math.log2(sum(2 ** d for d in dims)))
        ok &= max(dims) <= t <= hi < sum(dims) + 1
        rows.append((dims, t, hi, sum(dims)))
    report("S6 exclusive sorts", ok,
           "(dims, exact rank, floor log2 sum 2^d, rank of the free product) = " + str(rows))


# ---------------------------------------------------------------- S7
def check_S7():
    """The exception measure: profile counts estimate the predictive rule; no lambda matches it."""
    G = [(0.5, 0.02), (0.5, 0.5)]            # mixing law over objects: half nearly constant, half free

    def r_exact(a, b):
        num = sum(w * th ** (a + 1) * (1 - th) ** b for w, th in G)
        den = sum(w * th ** a * (1 - th) ** b for w, th in G)
        return num / den

    # no lambda reproduces r(0,b) for b = 1..6
    prof = [(a, b) for a in range(7) for b in range(7) if a + b <= 6]
    best_gap = min(max(abs(r_exact(a, b) - (a + lam / 2) / (a + b + lam)) for a, b in prof)
                   for lam in [x / 100 for x in range(1, 2000)])
    rng = random.Random(7)
    M, n_occ = 20000, 12
    seq = []
    for _ in range(M):
        th = G[0][1] if rng.random() < G[0][0] else G[1][1]
        seq.append([int(rng.random() < th) for _ in range(n_occ)])
    # profile table from the first half of the objects, scored on the second half
    cnt = {}
    for s in seq[: M // 2]:
        a = b = 0
        for v in s:
            c = cnt.setdefault((a, b), [0, 0])
            c[v] += 1
            a, b = a + v, b + 1 - v
    max_dev = max(abs(c[1] / (c[0] + c[1]) - r_exact(a, b)) for (a, b), c in cnt.items() if c[0] + c[1] >= 2000)

    def bits_profile(s):
        a = b = 0
        tot = 0.0
        for v in s:
            c = cnt.get((a, b), [0, 0])
            p = (c[1] + 0.5) / (c[0] + c[1] + 1.0)
            tot -= math.log2(p if v else 1 - p)
            a, b = a + v, b + 1 - v
        return tot

    def bits_lambda(s, lam):
        a = b = 0
        tot = 0.0
        for v in s:
            p = (a + lam / 2) / (a + b + lam)
            tot -= math.log2(p if v else 1 - p)
            a, b = a + v, b + 1 - v
        return tot

    test = seq[M // 2:]
    trans = {}
    for sq in test:
        a = b = 0
        for v in sq:
            c = trans.setdefault((a, b), [0, 0])
            c[v] += 1
            a, b = a + v, b + 1 - v
    nfacts = len(test) * n_occ

    def bits_rule(rule):
        tot = 0.0
        for (a, b), c in trans.items():
            p = min(max(rule(a, b), 1e-9), 1 - 1e-9)
            tot -= c[1] * math.log2(p) + c[0] * math.log2(1 - p)
        return tot / nfacts

    def prof_rule(a, b):
        c = cnt.get((a, b), [0, 0])
        return (c[1] + 0.5) / (c[0] + c[1] + 1.0)
    grid = [0.02 * 1.35 ** i for i in range(24)]
    bp = bits_rule(prof_rule)
    bl = min(bits_rule(lambda a, b, l=l: (a + l / 2) / (a + b + l)) for l in grid)
    bb = min(bits_rule(lambda a, b, u=u, w=w: (a + u) / (a + b + u + w)) for u in grid for w in grid)
    ideal = bits_rule(r_exact)
    gap_beta = min(max(abs(r_exact(a, b) - (a + u) / (a + b + u + w)) for a, b in prof) for u in grid for w in grid)
    report("S7 lambda-free estimate", gap_beta > 5 * max_dev and bp < bb < bl and bp - ideal < 0.2 * (bb - ideal),
           f"profile frequencies within {max_dev:.3f} of the exact rule; no Beta(u, w) rule comes closer than "
           f"{gap_beta:.3f} on profiles of size <= 6 (symmetric lambda: {best_gap:.3f}); bits/fact: exact rule "
           f"{ideal:.4f}, profile table {bp:.4f}, best Beta(u, w) {bb:.4f}, best symmetric lambda {bl:.4f}")


# ---------------------------------------------------------------- S8
def check_S8():
    """Relabelling-invariant laws = a law on equality patterns times a uniform injection."""
    V, n = 3, 4
    strings = list(itertools.product(range(V), repeat=n))

    def pattern(x):
        seen = {}
        return tuple(seen.setdefault(c, len(seen)) for c in x)

    w = {x: RNG.random() for x in strings}
    perms = list(itertools.permutations(range(V)))
    P = {x: sum(w[tuple(p[c] for c in x)] for p in perms) for x in strings}     # symmetrise
    Z = sum(P.values())
    P = {x: v / Z for x, v in P.items()}
    pat = {}
    for x, v in P.items():
        pat[pattern(x)] = pat.get(pattern(x), 0) + v
    ok = True
    for x in strings:
        k = len(set(x))
        falling = math.prod(V - i for i in range(k))
        ok &= abs(P[x] - pat[pattern(x)] / falling) < 1e-12
    # converse: any law on patterns (with at most |V| blocks) gives an invariant law that sums to 1
    pats = sorted(pat)
    w2 = [RNG.random() for _ in pats]
    Pi = {q: v / sum(w2) for q, v in zip(pats, w2)}
    Q = {x: Pi[pattern(x)] / math.prod(V - i for i in range(len(set(x)))) for x in strings}
    ok &= abs(sum(Q.values()) - 1) < 1e-12
    ok &= all(abs(Q[x] - Q[tuple(p[c] for c in x)]) < 1e-12 for x in strings for p in perms)
    report("S8 pattern law", ok, f"|V|={V}, n={n}: an invariant law is its pattern law over the falling factorial, and conversely")


if __name__ == "__main__":
    for f in (check_S1, check_S2, check_S3, check_S4, check_S5, check_S6, check_S7, check_S8):
        f()
    print("ALL PASS" if not FAILED else f"FAILED: {FAILED}")
