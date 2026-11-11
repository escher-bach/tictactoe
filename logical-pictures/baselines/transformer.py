#!/usr/bin/env python3
"""Small decoder-only char transformer baseline (PyTorch, CPU, 2 threads).

Architecture: char embedding (27) + learned positions (256), pre-LN blocks
(causal multi-head self-attention + 4x GELU MLP), final LayerNorm, linear head.
Optimiser: AdamW, linear warmup then cosine decay.  Because the budget is
wall-clock, the schedule is a function of elapsed time / budget, not of steps.

Budget (strict): the whole training loop -- optimisation steps AND the periodic
validation-subset evaluations used for checkpoint selection -- is stopped by
time checks so that it ends inside --budget seconds.  Only the final full
valid/test evaluation of the selected checkpoint happens after the budget.

Checkpoint selection: lowest bpc on a fixed validation subset (first 51,200
valid chars, non-overlapping 256-windows), evaluated ~12 times during training
and once at the end; the best state is restored for the final evaluation.

Final evaluation (valid and test, exactly the same procedure):
  windows of 256 input chars with stride 128.  The first window scores its
  targets 1..256 (context lengths 1..256); every later window scores only its
  last 128 targets, so each of those chars is predicted from >= 129 and <= 256
  preceding chars.  A last end-aligned window covers the tail.  Every char is
  scored exactly once; char 0 (no context at all) is charged log2(27) bits.
  bpc = total bits / number of chars in the split.

Usage: python3 transformer.py --ntrain 1000000        (uses the chosen config)
       python3 transformer.py --ntrain 1000000 --budget 120 --layers 2 --d 96 \
               --tag pilot --out pilots.jsonl          (a pilot run)
"""
import argparse
import copy
import math
import os
import time

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from common import HERE, V, append_result, load_splits

torch.set_num_threads(2)
try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass

CTX = 256
BUDGETS = {1_000_000: 600.0, 10_000_000: 1500.0}
# Chosen from the pilot runs recorded in pilots.jsonl (see README.md).
CONFIGS = {
    1_000_000: dict(layers=2, d=96, heads=4, bs=8, lr=3e-3, dropout=0.1, qkv_init="xavier"),
    10_000_000: dict(layers=2, d=96, heads=4, bs=8, lr=3e-3, dropout=0.0, qkv_init="xavier"),
}


class Block(nn.Module):
    def __init__(self, d, heads, dropout):
        super().__init__()
        self.heads = heads
        self.ln1 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)
        self.ln2 = nn.LayerNorm(d)
        self.fc = nn.Linear(d, 4 * d)
        self.fc_out = nn.Linear(4 * d, d)
        self.dropout = dropout

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(self.ln1(x)).view(B, T, 3, self.heads, C // self.heads).permute(2, 0, 3, 1, 4)
        # no dropout on the attention map itself: sampling a BxHxTxT mask costs
        # ~2.7x in step time on CPU; dropout is applied to embeddings + residuals
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        a = a.transpose(1, 2).reshape(B, T, C)
        x = x + F.dropout(self.proj(a), self.dropout, self.training)
        h = self.fc_out(F.gelu(self.fc(self.ln2(x))))
        return x + F.dropout(h, self.dropout, self.training)


class CharTransformer(nn.Module):
    def __init__(self, layers, d, heads, dropout, qkv_init="small"):
        super().__init__()
        self.tok = nn.Embedding(V, d)
        self.pos = nn.Embedding(CTX, d)
        self.blocks = nn.ModuleList([Block(d, heads, dropout) for _ in range(layers)])
        self.ln_f = nn.LayerNorm(d)
        self.head = nn.Linear(d, V)
        self.dropout = dropout
        self.apply(self._init)
        for b in self.blocks:  # GPT-2 style scaled init of the residual projections
            for lin in (b.proj, b.fc_out):
                nn.init.normal_(lin.weight, std=0.02 / math.sqrt(2 * layers))
            if qkv_init == "xavier":
                # as in nn.MultiheadAttention: larger query/key/value weights, so
                # attention logits are not ~0 at init (shortens the early plateau)
                nn.init.xavier_uniform_(b.qkv.weight)

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, idx):
        T = idx.shape[1]
        x = self.tok(idx) + self.pos.weight[:T]
        x = F.dropout(x, self.dropout, self.training)
        for b in self.blocks:
            x = b(x)
        return self.head(self.ln_f(x))


@torch.no_grad()
def subset_bpc(model, seg, nchars=51_200, bs=50):
    """Cheap selection metric: non-overlapping 256-windows over seg[:nchars+1]."""
    model.eval()
    nwin = nchars // CTX
    x = seg[:nwin * CTX].view(nwin, CTX)
    y = seg[1:nwin * CTX + 1].view(nwin, CTX)
    tot = 0.0
    for i in range(0, nwin, bs):
        logits = model(x[i:i + bs])
        tot += F.cross_entropy(logits.reshape(-1, V), y[i:i + bs].reshape(-1), reduction="sum").item()
    model.train()
    return tot / (nwin * CTX) / math.log(2)


@torch.no_grad()
def full_bpc(model, seg, bs=64, stride=128):
    """Sliding-window bpc over the whole segment (see module docstring)."""
    model.eval()
    n = seg.numel()
    starts, firsts = [], []      # window start, first target index scored by it
    nxt = 1                      # next unscored target
    s = 0
    while s + CTX <= n - 1:
        starts.append(s)
        firsts.append(nxt)
        nxt = s + CTX + 1
        s += stride
    if nxt <= n - 1:             # tail: one end-aligned window
        starts.append(n - 1 - CTX)
        firsts.append(nxt)
    bits = math.log2(V)          # char 0: no context -> uniform
    scored = 1
    # context length of the first (= least-context) char scored by each window
    min_ctx = min([f - s for s, f in zip(starts[1:], firsts[1:])], default=CTX)
    ar = torch.arange(CTX)
    for i in range(0, len(starts), bs):
        st = torch.tensor(starts[i:i + bs])
        fi = torch.tensor(firsts[i:i + bs])
        idx = st[:, None] + ar[None, :]
        tgt_pos = idx + 1
        logits = model(seg[idx])
        nll = F.cross_entropy(logits.reshape(-1, V), seg[tgt_pos].reshape(-1), reduction="none").view(idx.shape)
        mask = tgt_pos >= fi[:, None]
        bits += (nll * mask).sum().item() / math.log(2)
        scored += int(mask.sum())
    assert scored == n, (scored, n)
    model.train()
    return bits / n, min_ctx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ntrain", type=int, default=1_000_000)
    ap.add_argument("--budget", type=float, default=None, help="training wall-clock seconds")
    ap.add_argument("--layers", type=int, default=None)
    ap.add_argument("--d", type=int, default=None)
    ap.add_argument("--heads", type=int, default=None)
    ap.add_argument("--bs", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--dropout", type=float, default=None)
    ap.add_argument("--qkv_init", choices=["small", "xavier"], default=None)
    ap.add_argument("--wd", type=float, default=0.1)
    ap.add_argument("--warmup_frac", type=float, default=0.03)
    ap.add_argument("--n_evals", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="final")
    ap.add_argument("--out", default="results.jsonl")
    ap.add_argument("--skip_test", action="store_true", help="pilots: do not touch test")
    args = ap.parse_args()

    cfg = dict(CONFIGS.get(args.ntrain, CONFIGS[1_000_000]))
    for k in cfg:
        if getattr(args, k) is not None:
            cfg[k] = getattr(args, k)
    budget = args.budget if args.budget is not None else BUDGETS.get(args.ntrain, 600.0)

    torch.manual_seed(args.seed)
    tr, va, te = (torch.from_numpy(a.astype(np.int64)) for a in load_splits(args.ntrain))
    model = CharTransformer(cfg["layers"], cfg["d"], cfg["heads"], cfg["dropout"], cfg["qkv_init"])
    nparams = sum(p.numel() for p in model.parameters())
    decay = [p for p in model.parameters() if p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.dim() < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": args.wd},
                             {"params": no_decay, "weight_decay": 0.0}],
                            lr=cfg["lr"], betas=(0.9, 0.99))
    print(f"config {cfg} budget {budget:.0f}s params {nparams} threads {torch.get_num_threads()}", flush=True)

    bs = cfg["bs"]
    ar = torch.arange(CTX)
    gen = torch.Generator().manual_seed(args.seed)
    base_lr, min_lr = cfg["lr"], cfg["lr"] * 0.02

    def lr_at(p):
        if p < args.warmup_frac:
            return base_lr * max(p / args.warmup_frac, 0.02)
        q = min(1.0, (p - args.warmup_frac) / (1.0 - args.warmup_frac))
        return min_lr + 0.5 * (base_lr - min_lr) * (1.0 + math.cos(math.pi * q))

    curve = []
    load_start = os.getloadavg()[0]
    best = (float("inf"), None, -1)
    model.train()
    t0 = time.time()
    # eval at init: gives the first curve point and measures how long one eval takes
    te0 = time.time()
    v = subset_bpc(model, va)
    eval_dur = time.time() - te0
    curve.append({"t": round(time.time() - t0, 1), "step": 0, "tokens": 0, "lr": 0.0,
                  "train_bpc": None, "valid_sub_bpc": round(v, 4)})
    reserve = 3.0 * eval_dur + 3.0            # time kept back for the final in-budget eval
    train_end = budget - reserve              # optimisation must stop here
    eval_times = [train_end * (i + 1) / args.n_evals for i in range(args.n_evals - 1)]
    step, run_loss, run_n, lr = 0, 0.0, 0, 0.0
    while True:
        el = time.time() - t0
        if el >= train_end:
            break
        if eval_times and el >= eval_times[0]:
            eval_times.pop(0)
            v = subset_bpc(model, va)
            curve.append({"t": round(time.time() - t0, 1), "step": step, "tokens": step * bs * CTX,
                          "lr": round(lr, 6), "train_bpc": round(run_loss / max(run_n, 1) / math.log(2), 4),
                          "valid_sub_bpc": round(v, 4)})
            print(curve[-1], flush=True)
            run_loss, run_n = 0.0, 0
            if v < best[0]:
                best = (v, copy.deepcopy(model.state_dict()), step)
            continue
        lr = lr_at(el / train_end)
        for g in opt.param_groups:
            g["lr"] = lr
        ix = torch.randint(0, tr.numel() - CTX - 1, (bs,), generator=gen)
        idx = ix[:, None] + ar[None, :]
        logits = model(tr[idx])
        loss = F.cross_entropy(logits.view(-1, V), tr[idx + 1].view(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        step += 1
        run_loss += loss.item()
        run_n += 1
    # final in-budget eval
    v = subset_bpc(model, va)
    curve.append({"t": round(time.time() - t0, 1), "step": step, "tokens": step * bs * CTX,
                  "lr": round(lr, 6), "train_bpc": round(run_loss / max(run_n, 1) / math.log(2), 4),
                  "valid_sub_bpc": round(v, 4)})
    print(curve[-1], flush=True)
    if v < best[0]:
        best = (v, copy.deepcopy(model.state_dict()), step)
    train_seconds = time.time() - t0
    load_end = os.getloadavg()[0]
    assert train_seconds <= budget, f"budget exceeded: {train_seconds:.1f}s > {budget}s"

    model.load_state_dict(best[1])
    t1 = time.time()
    valid_bpc, vctx = full_bpc(model, va)
    if args.skip_test:
        test_bpc, tctx = None, None
    else:
        test_bpc, tctx = full_bpc(model, te)
    eval_seconds = time.time() - t1
    rec = {
        "model": "transformer", "ntrain": args.ntrain, "mode": "static", "tag": args.tag,
        "config": {**cfg, "ctx": CTX, "wd": args.wd, "betas": [0.9, 0.99], "grad_clip": 1.0,
                   "schedule": f"time-based: linear warmup {args.warmup_frac:.0%} of budget, cosine to 2% of peak lr",
                   "budget_seconds": budget, "seed": args.seed, "threads": torch.get_num_threads(),
                   "eval": "256-char windows, stride 128, >=129 chars of context for every char after the first 256",
                   "min_context_after_first_window": vctx,
                   "selected_step": best[2], "selected_valid_sub_bpc": round(best[0], 4)},
        "valid_bpc": round(valid_bpc, 5), "test_bpc": None if test_bpc is None else round(test_bpc, 5),
        "train_seconds": round(train_seconds, 1), "eval_seconds": round(eval_seconds, 1),
        "params": nparams, "steps": step, "tokens_seen": step * bs * CTX,
        "epochs": round(step * bs * CTX / args.ntrain, 3),
        "tokens_per_second": round(step * bs * CTX / train_seconds, 1),
        "loadavg_1min_start_end": [round(load_start, 2), round(load_end, 2)],
        "curve": curve,
    }
    append_result(rec, os.path.join(HERE, args.out))


if __name__ == "__main__":
    main()
