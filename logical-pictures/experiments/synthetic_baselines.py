"""Reference models on the worlds of pm/worlds.py: PPM (method C, exclusion) and a
small transformer.  Reads the texts written by experiments/synthetic.py."""
import json
import math
import sys
import time

import numpy as np

sys.path.insert(0, __file__.rsplit("/", 2)[0] + "/baselines")
sys.path.insert(0, __file__.rsplit("/", 2)[0])


def ppm_bits(train, test, V, K=5, dynamic=True):
    """Per-position code length of `test` under PPM-C with symbol exclusion."""
    tabs = [dict() for _ in range(K + 1)]

    def update(hist, s):
        for k in range(min(K, len(hist)) + 1):
            ctx = tuple(hist[len(hist) - k:])
            d = tabs[k].setdefault(ctx, {})
            d[s] = d.get(s, 0) + 1

    hist = []
    for s in train.tolist():
        update(hist, s)
        hist = (hist + [s])[-K:]
    out = np.zeros(len(test))
    hist = []
    for i, s in enumerate(test.tolist()):
        excl = set()
        logp = 0.0
        done = False
        for k in range(min(K, len(hist)), -1, -1):
            d = tabs[k].get(tuple(hist[len(hist) - k:]))
            if not d:
                continue
            live = {a: c for a, c in d.items() if a not in excl}
            if not live:
                continue
            T, q = sum(live.values()), len(live)
            left = V - len(excl)
            if s in live:
                logp += math.log2(live[s] / (T if q == left else T + q))
                done = True
                break
            logp += math.log2(q / (T + q))
            excl |= set(live)
        if not done:
            logp += math.log2(1.0 / (V - len(excl)))
        out[i] = -logp
        if dynamic:
            update(hist, s)
        hist = (hist + [s])[-K:]
    return out


def transformer_bits(train, valid, tests, V, budget, ctx=64, seed=0):
    import torch
    import torch.nn.functional as F
    import transformer as T
    T.V = V
    T.CTX = ctx
    torch.manual_seed(seed)
    torch.set_num_threads(2)
    model = T.CharTransformer(2, 96, 4, 0.0, "xavier")
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=0.01, betas=(0.9, 0.99))
    tr = torch.tensor(train, dtype=torch.long)
    t0, step = time.time(), 0
    while time.time() - t0 < budget:
        frac = (time.time() - t0) / budget
        for g in opt.param_groups:
            g["lr"] = 2e-3 * min(1.0, frac / 0.03) * (0.02 + 0.98 * 0.5 * (1 + math.cos(math.pi * frac)))
        ix = torch.randint(0, len(tr) - ctx - 1, (32,))
        idx = ix[:, None] + torch.arange(ctx)[None, :]
        loss = F.cross_entropy(model(tr[idx]).reshape(-1, V), tr[idx + 1].reshape(-1))
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        step += 1
    model.eval()
    outs = []
    with torch.no_grad():
        for test in tests:
            te = torch.tensor(test, dtype=torch.long)
            bits = np.full(len(te), math.log2(V))
            half = ctx // 2
            for s in range(0, len(te) - ctx - 1, half):
                lg = model(te[s:s + ctx][None])[0]
                nll = F.cross_entropy(lg, te[s + 1:s + ctx + 1], reduction="none").numpy() / math.log(2)
                lo = 0 if s == 0 else half - 1
                bits[s + 1 + lo:s + ctx + 1] = nll[lo:]
            outs.append(bits)
    return outs, step, sum(p.numel() for p in model.parameters())


if __name__ == "__main__":
    from pm.worlds import analogy, respelled
    budget = float(sys.argv[1]) if len(sys.argv) > 1 else 240.0
    res = []

    def add(world, model, mode, bits, mask, extra=None):
        r = dict(world=world, model=model, mode=mode, bits_at_determined=float(bits[mask].mean()),
                 bits_per_sign=float(bits.mean()), **(extra or {}))
        print(r, flush=True)
        res.append(r)

    x, ans, V = respelled(26_000, 1)
    ntr, nva = 9 * 20_000, 9 * 3_000
    tr, va, te, m = x[:ntr], x[ntr:ntr + nva], x[ntr + nva:], ans[ntr + nva:]
    for dyn in (True, False):
        add("respelled recall", "PPM-C order 5", "dynamic" if dyn else "static", ppm_bits(tr, te, V, dynamic=dyn), m)
    (b,), steps, params = transformer_bits(tr, va, [te], V, budget)
    add("respelled recall", "transformer", "static", b, m, dict(steps=steps, params=params, budget=budget))

    V = 401
    xtr, _ = analogy(24_000, 2, np.arange(1, 201))
    x_in, m_in = analogy(3_000, 4, np.arange(1, 201))
    x_out, m_out = analogy(3_000, 5, np.arange(301, 401))
    for name, te, m in (("analogy, signs of the corpus", x_in, m_in), ("analogy, signs never seen", x_out, m_out)):
        for dyn in (True, False):
            add(name, "PPM-C order 5", "dynamic" if dyn else "static", ppm_bits(xtr, te, V, dynamic=dyn), m)
    (b1, b2), steps, params = transformer_bits(xtr, None, [x_in, x_out], V, budget)
    add("analogy, signs of the corpus", "transformer", "static", b1, m_in, dict(steps=steps, params=params, budget=budget))
    add("analogy, signs never seen", "transformer", "static", b2, m_out, dict(steps=steps, params=params, budget=budget))
    json.dump(res, open("runs/synthetic_baselines.json", "w"), indent=1)
