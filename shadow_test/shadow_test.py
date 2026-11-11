"""Are vectors a shadow of a logical picture?  Two tests of THEORY.md section 8.7.

Part A  The product world of Proposition 8.1 ({+-1}^n, noisy-copy positives).
        Calibrates the instrument: the vector optimum is fixed only up to O(n);
        a search for two-valued directions should return the n bits and nothing else.

Part B  Tic-tac-toe. Small transformers trained on next-move prediction over random
        legal games, several seeds. The world's facts are known (cell states), but
        nothing forces the network to hold them as few-valued directions, or to hold
        the same ones in every seed. Predictions 1 and 2 of section 8.7 say it will.

The instrument. For a unit direction u in whitened activation space, y = u.z has
mean 0 and variance 1, and Pearson's inequality gives
        D(u) = kurtosis(y) - skewness(y)^2 - 1 >= 0,
with equality iff y takes exactly two values. D is rotation-equivariant, needs no
labels, and is 2 for a Gaussian. We minimise D from many starts and read each
minimum with small D as a partition of occurrences (threshold at skewness/2, the
midpoint of the two values).

Thresholds were fixed before any run: D < 0.25 counts as two-valued, minority
class >= 1%, |phi| >= 0.95 counts as the same partition.
"""
import json, sys, time
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

D_MAX, MIN_FRAC, PHI_SAME = 0.25, 0.01, 0.95
LINES = [(0,1,2),(3,4,5),(6,7,8),(0,3,6),(1,4,7),(2,5,8),(0,4,8),(2,4,6)]
BOS, END = 9, 10
torch.set_num_threads(4)


# ---------------------------------------------------------------- instrument

def whiten(X, var_keep=0.999, k=None):
    """Centre, keep the top principal components, scale each to unit variance."""
    X = np.asarray(X, dtype=np.float64)
    Xc = X - X.mean(0)
    U, S, _ = np.linalg.svd(Xc, full_matrices=False)
    var = S ** 2
    if k is None:
        k = int(np.searchsorted(np.cumsum(var) / var.sum(), var_keep) + 1)
    return U[:, :k] * np.sqrt(len(X)), k


def defect(Y):
    """Columns of Y have mean 0, variance 1. Returns D and skewness per column."""
    m3 = (Y ** 3).mean(0)
    return (Y ** 4).mean(0) - m3 ** 2 - 1, m3


def find_two_valued(Z, restarts=1024, steps=400, seed=0, fit_n=12000):
    """Minimise D over unit directions from many starts. Returns the distinct
    partitions (as +-1 label columns, N x m), their defects, and all final defects."""
    g = torch.Generator().manual_seed(seed)
    N, k = Z.shape
    Zt = torch.tensor(Z, dtype=torch.float32)
    idx = torch.randperm(N, generator=g)[:fit_n]
    Zf = Zt[idx]
    Zf = (Zf - Zf.mean(0))
    # half random directions, half differences of two data points
    U = torch.randn(restarts, k, generator=g)
    h = restarts // 2
    a, b = torch.randint(len(Zf), (2, h), generator=g)
    U[:h] = Zf[a] - Zf[b] + 1e-3 * U[:h]
    U.requires_grad_(True)
    opt = torch.optim.Adam([U], lr=0.03)
    for _ in range(steps):
        Y = Zf @ F.normalize(U, dim=1).T
        Y = (Y - Y.mean(0)) / Y.std(0, correction=0)
        d, _ = defect(Y)
        opt.zero_grad(); d.sum().backward(); opt.step()
    with torch.no_grad():
        Y = Zt @ F.normalize(U, dim=1).T
        Y = (Y - Y.mean(0)) / Y.std(0, correction=0)
        d, m3 = defect(Y)
        lab = (Y > m3 / 2)
        frac = lab.float().mean(0)
        frac = torch.minimum(frac, 1 - frac)
    d_all = d.numpy()
    keep = ((d < D_MAX) & (frac >= MIN_FRAC)).numpy()
    L = np.where(lab.numpy()[:, keep], 1.0, -1.0)
    dk = d_all[keep]
    order = np.argsort(dk)
    L, dk = L[:, order], dk[order]
    chosen = []
    if L.shape[1]:
        Ls = (L - L.mean(0)) / L.std(0)
        C = np.abs(Ls.T @ Ls) / len(L)
        for j in range(L.shape[1]):
            if all(C[j, i] < PHI_SAME for i in chosen):
                chosen.append(j)
    return L[:, chosen], dk[chosen], d_all


def phi(A, B):
    """|phi| (absolute Matthews correlation) between columns of A and of B."""
    A = (A - A.mean(0)) / (A.std(0) + 1e-12)
    B = (B - B.mean(0)) / (B.std(0) + 1e-12)
    return np.abs(A.T @ B) / len(A)


def cka(X, Y):
    X = X - X.mean(0); Y = Y - Y.mean(0)
    return float(np.linalg.norm(X.T @ Y) ** 2 / (np.linalg.norm(X.T @ X) * np.linalg.norm(Y.T @ Y)))


def matched_corr(X, Y):
    """Mean |corr| of coordinates under the best one-to-one matching."""
    C = phi(X, Y)
    r, c = linear_sum_assignment(-C)
    return float(C[r, c].mean())


def pca_coords(X, k):
    Xc = X - X.mean(0)
    U, S, _ = np.linalg.svd(Xc, full_matrices=False)
    return U[:, :k] * S[:k]


# ---------------------------------------------------------------- part A

def part_a(n=8, lam=0.5, seeds=(0, 1)):
    M = 2 ** n
    Zb = np.array([[1 if (i >> j) & 1 else -1 for j in range(n)] for i in range(M)], dtype=np.float64)
    K = np.ones((M, M))
    for j in range(n):
        K *= (1 + lam * np.outer(Zb[:, j], Zb[:, j])) / 2
    ev = np.sort(np.linalg.eigvalsh(K))[::-1]
    out = {"n": n, "lambda": lam,
           "top_eigenvalues": [round(float(x), 6) for x in ev[:n + 3]],
           "predicted": [1.0] + [lam] * n + [lam ** 2] * 2}
    Kt = torch.tensor(K, dtype=torch.float64)
    reps = []
    for s in seeds:
        torch.manual_seed(s)
        Fm = (0.1 * torch.randn(M, n + 1, dtype=torch.float64)).requires_grad_(True)
        opt = torch.optim.Adam([Fm], lr=0.02)
        for _ in range(6000):  # exact population spectral-contrastive loss
            G = Fm @ Fm.T
            loss = -2 * (Kt * G).sum() / M + (G ** 2).sum() / M ** 2
            opt.zero_grad(); loss.backward(); opt.step()
        reps.append(Fm.detach().numpy())
    out["cka_between_seeds"] = cka(reps[0], reps[1])
    out["raw_coordinate_match"] = matched_corr(reps[0], reps[1])
    # how two-valued are the learned coordinates themselves?
    Zc = (reps[0] - reps[0].mean(0)); Zc = Zc[:, Zc.std(0) > 1e-6]; Zc = Zc / Zc.std(0)
    out["defect_of_raw_coordinates"] = [round(float(x), 3) for x in defect(torch.tensor(Zc))[0]]
    parts = []
    for s, R in zip(seeds, reps):
        Z, k = whiten(R, k=n)
        L, d, d_all = find_two_valued(Z, restarts=512, steps=600, seed=s, fit_n=M)
        P = phi(L, Zb) if L.shape[1] else np.zeros((0, n))
        parts.append(L)
        out[f"seed{s}"] = {"whitened_dim": k, "two_valued_found": int(L.shape[1]),
                           "max_defect_found": float(d.max()) if len(d) else None,
                           "bits_recovered_exactly": int((P.max(0) > 1 - 1e-9).sum()),
                           "found_that_are_not_bits": int((P.max(1) < 1 - 1e-9).sum()) if len(P) else 0,
                           "share_of_starts_reaching_D<0.25": float((d_all < D_MAX).mean())}
    if parts[0].shape[1] and parts[1].shape[1]:
        out["partition_match_between_seeds"] = float((phi(parts[0], parts[1]).max(1) > 1 - 1e-9).mean())
    return out


# ---------------------------------------------------------------- part B: world

def sample_games(B, rng):
    """Uniform random legal play = a random permutation cut at the first win."""
    perm = np.argsort(rng.random((B, 9)), axis=1)   # perm[b, t] = cell played at move t
    when = np.argsort(perm, axis=1)                 # when[b, c] = move at which c is played
    win_t = np.full(B, 9)
    for a, b, c in LINES:
        ta, tb, tc = when[:, a], when[:, b], when[:, c]
        same = (ta % 2 == tb % 2) & (tb % 2 == tc % 2)
        win_t = np.where(same, np.minimum(win_t, np.maximum(np.maximum(ta, tb), tc)), win_t)
    return perm, when, np.minimum(win_t + 1, 9)     # length = number of moves played


def tokens(perm, length):
    B = len(perm)
    seq = np.full((B, 11), END)
    seq[:, 0] = BOS
    seq[:, 1:10] = perm
    j = np.arange(11)[None, :]
    seq = np.where(j > length[:, None], END, seq)
    valid = np.arange(10)[None, :] <= length[:, None]
    return seq[:, :10], seq[:, 1:], valid


def predicates(perm, when, length):
    """Ground-truth facts for every valid occurrence (game b, position j = moves so far)."""
    B = len(perm)
    j = np.arange(10)[None, :, None]                       # B x 10 x 9 over cells
    w = when[:, None, :]
    occ = w < j
    isx = occ & (w % 2 == 0)
    iso = occ & (w % 2 == 1)
    mine = occ & (w % 2 == (j - 1) % 2)                    # owned by whoever just moved
    theirs = occ & ~mine
    last = np.zeros((B, 10, 9), bool)
    last[:, 1:, :] = perm[:, :, None] == np.arange(9)[None, None, :]
    jj = np.broadcast_to(np.arange(10)[None, :], (B, 10))
    valid = jj <= length[:, None]
    cols, names = [], []
    for nm, arr in [("occupied", occ), ("X", isx), ("O", iso), ("mover", mine), ("other", theirs), ("last", last)]:
        for c in range(9):
            cols.append(arr[:, :, c]); names.append(f"{nm}[{c}]")
    cols.append(jj % 2 == 1); names.append("odd(t)")
    cols.append(jj == length[:, None]); names.append("game_over")
    for m in range(1, 10):
        cols.append(jj >= m); names.append(f"t>={m}")
    P = np.stack([c[valid] for c in cols], 1)
    keep = (P.mean(0) > 0) & (P.mean(0) < 1)
    return np.where(P[:, keep], 1.0, -1.0), [n for n, k in zip(names, keep) if k], valid


FAMILIES = ["occupied", "X", "O", "mover", "other", "last", "odd(t)", "game_over", "t>="]


def family(name):
    return next(f for f in FAMILIES if name.startswith(f))


# ---------------------------------------------------------------- part B: model

class Block(nn.Module):
    def __init__(s, d, h):
        super().__init__()
        s.ln1, s.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        s.att = nn.MultiheadAttention(d, h, batch_first=True)
        s.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(s, x, mask):
        y = s.ln1(x)
        x = x + s.att(y, y, y, attn_mask=mask, need_weights=False)[0]
        return x + s.mlp(s.ln2(x))


class TTT(nn.Module):
    def __init__(s, d=64, h=4, layers=2):
        super().__init__()
        s.tok, s.pos = nn.Embedding(11, d), nn.Embedding(10, d)
        s.blocks = nn.ModuleList(Block(d, h) for _ in range(layers))
        s.lnf, s.head = nn.LayerNorm(d), nn.Linear(d, 11)
        s.register_buffer("mask", torch.triu(torch.ones(10, 10, dtype=torch.bool), 1))

    def forward(s, x):
        r = s.tok(x) + s.pos(torch.arange(x.shape[1]))
        resid = [r]
        for b in s.blocks:
            r = b(r, s.mask[:x.shape[1], :x.shape[1]])
            resid.append(r)
        return s.head(s.lnf(r)), resid


def train(seed, steps=4000, batch=1024):
    torch.manual_seed(seed)
    rng = np.random.default_rng(1000 + seed)
    model = TTT()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    for _ in range(steps):
        perm, _, length = sample_games(batch, rng)
        x, y, valid = (torch.tensor(a) for a in tokens(perm, length))
        logits, _ = model(x)
        loss = F.cross_entropy(logits[valid], y[valid])
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    return model.eval()


@torch.no_grad()
def evaluate(model, perm, length):
    x, y, valid = (torch.tensor(a) for a in tokens(perm, length))
    logits, resid = model(x)
    loss = F.cross_entropy(logits[valid], y[valid]).item()
    j = np.arange(10)[None, :]
    opt = np.where(j < length[:, None], np.log(np.maximum(9 - j, 1)), 0.0)[valid.numpy()].mean()
    return loss - float(opt), [r[valid].numpy() for r in resid]


@torch.no_grad()
def history_ratio(model, perm, length, rng):
    """Same board, same last move, different order of the earlier moves.
    Ratio of twin distance to the distance between unrelated prefixes of the same
    length: 0 = the vector is a function of the picture, 1 = no closer than chance."""
    B = len(perm)
    j = np.array([rng.integers(4, L + 1) for L in length])   # prefix length, 4..L
    twin = perm.copy()
    for b in range(B):
        for par in (0, 1):
            slots = np.arange(par, j[b] - 1, 2)
            twin[b, slots] = perm[b, rng.permutation(slots)]
    differs = (twin != perm).any(1)
    xa = torch.tensor(tokens(perm, length)[0]); xb = torch.tensor(tokens(twin, length)[0])
    ra, rb = model(xa)[1], model(xb)[1]
    out = []
    for A, Bm in zip(ra, rb):
        ha = A[np.arange(B), j].numpy()[differs]; hb = Bm[np.arange(B), j].numpy()[differs]
        jj = j[differs]
        num = ((ha - hb) ** 2).sum(1).mean()
        den = []
        for t in np.unique(jj):
            h = ha[jj == t]
            if len(h) > 1:
                den.append((2 * ((h - h.mean(0)) ** 2).sum(1).mean(), len(h)))
        den = sum(d * n for d, n in den) / sum(n for _, n in den)
        out.append(float(num / den))
    return out


def analyse(acts, P, names, seed):
    """Unsupervised two-valued search on one layer's activations."""
    Z, k = whiten(acts)
    L, d, d_all = find_two_valued(Z, seed=seed)
    res = {"whitened_dim": k, "two_valued_found": int(L.shape[1]),
           "share_of_starts_reaching_D<0.25": float((d_all < D_MAX).mean()),
           "median_final_defect": float(np.median(d_all))}
    # Gaussian null: same shape, same (identity) covariance, same search
    Zg = np.random.default_rng(seed).standard_normal(Z.shape)
    Lg, _, dg = find_two_valued(Zg, seed=seed, restarts=256)
    res["gaussian_null"] = {"two_valued_found": int(Lg.shape[1]), "min_defect": float(dg.min())}
    if L.shape[1]:
        C = phi(L, P)                                   # found x predicates
        best = C.max(1)
        res["found_naming_a_world_fact"] = float((best >= PHI_SAME).mean())
        rec = C.max(0) >= PHI_SAME
        fam = {}
        for nme, r in zip(names, rec):
            f = family(nme); fam.setdefault(f, [0, 0]); fam[f][0] += int(r); fam[f][1] += 1
        res["facts_recovered_by_family"] = {f: f"{a}/{b}" for f, (a, b) in fam.items()}
    # supervised ceiling: is each fact linearly there at all?
    Zb = np.c_[Z, np.ones(len(Z))]
    W = np.linalg.lstsq(Zb, P, rcond=None)[0]
    acc = ((Zb @ W > 0) == (P > 0)).mean(0)
    fam = {}
    for nme, a in zip(names, acc):
        f = family(nme); fam.setdefault(f, []).append(a)
    res["linear_probe_accuracy_by_family"] = {f: round(float(np.mean(v)), 4) for f, v in fam.items()}
    return res, L


def part_b(seeds=(0, 1, 2, 3), n_eval=4000):
    rng = np.random.default_rng(7)
    perm, when, length = sample_games(n_eval, rng)
    P, names, _ = predicates(perm, when, length)
    out = {"occurrences": int(len(P)), "world_facts_tested": len(names), "models": {}}
    acts, parts = {}, {}
    runs = [(f"trained{s}", s, True) for s in seeds] + [(f"untrained{s}", s, False) for s in seeds[:2]]
    for tag, s, do_train in runs:
        t0 = time.time()
        if do_train:
            model = train(s)
        else:
            torch.manual_seed(s); model = TTT().eval()
        excess, resid = evaluate(model, perm, length)
        m = {"excess_loss_nats": excess,
             "history_ratio_by_layer": history_ratio(model, perm, length, np.random.default_rng(11))}
        for layer, a in enumerate(resid):
            m[f"layer{layer}"], parts[tag, layer] = analyse(a, P, names, s)
            acts[tag, layer] = a
        out["models"][tag] = m
        print(tag, f"{time.time() - t0:.0f}s", "excess", round(excess, 4), file=sys.stderr, flush=True)
    # agreement between seeds, three ways
    out["between_seeds"] = {}
    for kind, n in (("trained", len(seeds)), ("untrained", 2)):
        for layer in range(3):
            rows = {"cka": [], "raw_coordinate_match": [], "pca_coordinate_match": [], "partition_match": []}
            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    A, B = acts[f"{kind}{seeds[i]}", layer], acts[f"{kind}{seeds[j]}", layer]
                    La, Lb = parts[f"{kind}{seeds[i]}", layer], parts[f"{kind}{seeds[j]}", layer]
                    if La.shape[1] and Lb.shape[1]:
                        rows["partition_match"].append(float((phi(La, Lb).max(1) >= PHI_SAME).mean()))
                    elif La.shape[1]:
                        rows["partition_match"].append(0.0)
                    if i < j:
                        rows["cka"].append(cka(A, B))
                        rows["raw_coordinate_match"].append(matched_corr(A, B))
                        rows["pca_coordinate_match"].append(matched_corr(pca_coords(A, 20), pca_coords(B, 20)))
            out["between_seeds"][f"{kind}_layer{layer}"] = {k: round(float(np.mean(v)), 4) if v else None
                                                           for k, v in rows.items()}
    return out


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    res = {}
    if which in ("a", "all"):
        res["part_a"] = part_a()
    if which in ("b", "all"):
        res["part_b"] = part_b()
    print(json.dumps(res, indent=1))
