# Character-LM baselines on text8

Three reference models for the non-neural learner: a Kneser–Ney n-gram, PPM, and a
small transformer. All numbers are **bits per character on the 500k-char test
split**, i.e. the mean of −log2 p(x_t | x_<t).

## Data and protocol

| split | bytes of `text8` | size |
|---|---|---|
| train | `[0 : N]`, N = 1,000,000 or 10,000,000 | N |
| valid | `[90,000,000 : 90,200,000]` | 200,000 |
| test  | `[95,000,000 : 95,500,000]` | 500,000 |

Alphabet: 27 symbols (space + a–z). Valid is used for every model-selection
decision; test is only scored. Each evaluation segment starts with an **empty
context**: the chars just before byte 90,000,000 / 95,000,000 are not used, so
the first few chars of a segment are predicted from a short history. That costs
well under 0.001 bpc.

## Results

| model | mode | N = 1M valid | N = 1M test | N = 10M valid | N = 10M test |
|---|---|---|---|---|---|
| modified Kneser–Ney n-gram (best order on valid) | static | 1.9888 | **2.0503** (8-gram) | 1.7368 | **1.7976** (8-gram) |
| PPM (best config on valid) | static | 2.0711 | **2.1322** (PPMD, order 5) | 1.8046 | **1.8613** (PPMD, order 6) |
| PPM (best config on valid) | dynamic | 1.8988 | **1.9260** (PPMD, order 5) | 1.7450 | **1.7872** (PPMD, order 6) |
| transformer, 2 layers × d=96, 253,659 params | static | 2.2604 | **2.2800** | 1.9320 | **1.9788** |

Read these with three things in mind:

- **The transformer is the weakest model at both sizes, and that reflects the
  compute budget, not the architecture.** Details and curves are below.
- **Static vs dynamic is not a like-for-like comparison.** Dynamic PPM learns from
  the test text as it reads it; the n-gram, static PPM and the transformer do not.
- Test is 0.02–0.06 bpc harder than valid for every model, so the gap is a
  property of the two text regions and not a sign of selection overfitting.

## 1. `ngram.py` — interpolated modified Kneser–Ney

`python3 ngram.py --ntrain N` (about 10 s). Counts come from train only and
nothing is updated afterwards.

"Order n" here means an n-gram: **n−1 context chars** predict the next char.

For a model of maximum order n:

- the top level uses raw counts; every lower level uses continuation counts
  N1+(• u w), the number of distinct left extensions;
- p_k(w|u) = (c(uw) − D(c(uw))) / c(u•) + [D1·N1(u•) + D2·N2(u•) + D3+·N3+(u•)] / c(u•) · p_{k−1}(w|u′);
- below the unigram level sits a uniform 1/27; an unseen context passes the
  lower-order distribution through unchanged;
- D1, D2, D3+ are the Chen–Goodman closed-form estimates from each level's own
  count-of-counts.

| n-gram order | N = 1M valid | N = 1M test | N = 10M valid | N = 10M test |
|---|---|---|---|---|
| 3 | 2.9442 | 2.9524 | 2.9162 | 2.9160 |
| 4 | 2.4358 | 2.4566 | 2.3531 | 2.3679 |
| 5 | 2.1284 | 2.1727 | 1.9857 | 2.0233 |
| 6 | 2.0136 | 2.0735 | 1.8163 | 1.8681 |
| 7 | 1.9909 | 2.0501 | 1.7604 | 1.8161 |
| 8 | **1.9888** | **2.0503** | **1.7368** | **1.7976** |

Train time 0.3 s (1M) and 3.4 s (10M).

Checks:

- The full 27-way distribution is built at every valid and test position for
  every order, and on 2,000 uniformly random 7-char contexts. The largest
  |Σp − 1| seen is 9e-16, and the script asserts it stays below 1e-9.
- `python3 ngram.py --selftest` compares the numba code against a slow
  dict-based implementation written from the formulas. They agree to 2e-16.

Caveats:

- **The best order is 8 at both sizes, which is the top of the allowed range.**
  At N = 10M the curve is still falling (7 → 8 gains 0.019 on test), so a 9- or
  10-gram would probably score lower. At N = 1M it has flattened (7 and 8 tie on
  test).
- The closed-form discounts are degenerate for the unigram continuation level,
  because all 27 symbols have large continuation counts. That one level uses
  D = (0.5, 1, 1.5), the same convention as KenLM's `--discount_fallback`. Every
  other level uses the closed form. The discounts are stored per level in
  `results.jsonl`.
- No tuning of discounts on valid was done.

## 2. `ppm.py` — PPM with escape and symbol exclusion

`python3 ppm.py --ntrain N` (25 s for 1M, 80 s for 10M, for the whole grid).

"Order K" here means **K context chars**, so PPM order 5 has the same context
length as a 6-gram.

Contexts of every order 0..K live in one open-addressing hash table. Counts are
int32 and are never rescaled or capped. At each position the model walks from
order K down to 0:

- T is the total count and q the number of distinct symbols in the context,
  counting only symbols not already seen at a higher order (exclusion);
- method C: p(s) = c(s)/(T+q), p(escape) = q/(T+q);
- method D: p(s) = (2c(s)−1)/(2T), p(escape) = q/(2T);
- an unseen context, or one with nothing new to offer, is skipped at no cost;
- order −1 is uniform over the symbols still not excluded.

Two modes, both reported:

- **static**: counts are built on train and frozen.
- **dynamic**: the model reads train, then keeps updating while it reads the
  evaluation text. Only evaluation chars are scored. For the test number the
  model has read train + test so far; it has **not** read valid. The valid
  number used for selection comes from a separate copy that read train + valid.

Selection is per mode and per N, by valid bpc, over max order {4, 5, 6, 8} ×
escape {C, D} × update rule {update exclusion, full update}. Method D with
update exclusion won in all four cases.

| max order | N = 1M static | N = 1M dynamic | N = 10M static | N = 10M dynamic |
|---|---|---|---|---|
| 4 | 2.1559 / 2.2045 | 2.0004 / 2.0336 | 1.9949 / 2.0321 | 1.9522 / 1.9910 |
| 5 | **2.0711 / 2.1322** | **1.8988 / 1.9260** | 1.8381 / 1.8895 | 1.7873 / 1.8302 |
| 6 | 2.0855 / 2.1430 | 1.8997 / 1.9208 | **1.8046 / 1.8613** | **1.7450 / 1.7872** |
| 8 | 2.1690 / 2.2247 | 1.9745 / 1.9903 | 1.8439 / 1.9058 | 1.7730 / 1.8150 |

Cells are valid / test for method D with update exclusion; bold is the row
chosen on valid. The other three method/update combinations are in the `grid`
field of each PPM line in `results.jsonl`. On valid, method C is 0.004–0.033
worse than D under update exclusion. Full update is worse than update exclusion
everywhere for method D; for method C at order 8 it is slightly better in three
of the four columns.

Train time for the chosen config: 0.3 s (1M), 5.4 s (10M).

Checks:

- The full 27-way distribution is built at every scored position, for both
  methods and all 8 trained models per N. The largest |Σp − 1| is 1.3e-15, and
  the script asserts it stays below 1e-9. So these are true probabilities, not
  code-length estimates.
- `python3 ppm.py --selftest` compares total code length against an independent
  dict-based PPM for K ∈ {2, 4}, both methods, both update rules. They agree to
  6 decimals.

Caveats:

- **One deliberate departure from textbook PPM.** If a context has seen every
  symbol that is still un-excluded, there is nothing left to escape to, so the
  escape probability is set to 0 and the counts are normalised by T. Without
  this, C and D leak probability mass on a known 27-symbol alphabet. It mostly
  affects orders 0 and 1 and can only lower bpc slightly relative to a leaky
  implementation.
- "Update exclusion" means a symbol's count is incremented only in orders at or
  above the highest order that already contained it. This is the standard
  PPMC/PPMD rule.
- This is plain PPM: no information inheritance, no secondary escape
  estimation, no deterministic-context bonus. PPMII or PPMd-style variants would
  score lower.
- At N = 1M, dynamic order 6 has a lower test bpc (1.9208) than the selected
  order 5 (1.9260). Valid preferred order 5 by 0.001, so order 5 is what is
  reported.
- Static PPM is 0.06–0.08 bpc worse than the static n-gram at both sizes.
  That ordering is expected for PPM against Kneser–Ney smoothing.

## 3. `transformer.py` — small decoder-only transformer

`python3 transformer.py --ntrain N` (PyTorch 2.15 CPU build, 2 threads).

Architecture: char embedding + learned absolute positions (256), pre-LN blocks
(causal multi-head attention, 4× GELU MLP), final LayerNorm, linear head.
Context length 256. Training samples random 256-char crops of train.

Optimiser: AdamW, lr 3e-3, betas (0.9, 0.99), weight decay 0.1 on matrices,
gradient clip 1.0, batch 8 × 256. The schedule is a function of elapsed time
over the budget: linear warmup for the first 3 %, then cosine decay to 2 % of
peak.

**Budget.** The whole training loop is stopped by wall-clock checks: 600 s for
N = 1M, 1500 s for N = 10M. The loop includes the optimisation steps and the 13
short validation-subset evaluations used for checkpoint selection. Only the
final full valid/test evaluation of the selected checkpoint runs after it.

**Checkpoint selection.** Lowest bpc on a fixed validation subset (first 51,200
valid chars, non-overlapping windows). In both runs the last checkpoint was the
best one.

**Final evaluation, identical for valid and test.** Windows of 256 input chars
with stride 128. The first window scores its 256 targets (context 1..256 chars).
Every later window scores only its last 128 targets, so each of those chars is
predicted from 129–256 preceding chars. A final end-aligned window covers the
tail. Every char is scored exactly once. Char 0 has no context and is charged
log2(27) bits. The minimum context after the first window is recorded in the
result (129).

| | N = 1M | N = 10M |
|---|---|---|
| layers × d × heads | 2 × 96 × 4 | 2 × 96 × 4 |
| parameters | 253,659 | 253,659 |
| dropout | 0.1 | 0.0 |
| training budget / used | 600 s / 595.5 s | 1500 s / 1496.2 s |
| steps | 8,733 | 21,645 |
| tokens seen | 17.9M (17.9 epochs) | 44.3M (4.4 epochs) |
| throughput | 30.0k tokens/s | 29.6k tokens/s |
| 1-min load average, start → end | 3.99 → 4.38 | 4.22 → 4.93 |
| valid bpc | 2.2604 | 1.9320 |
| **test bpc** | **2.2800** | **1.9788** |

Dropout is on the embeddings and on both residual branches. 0.1 was chosen for
N = 1M because the model passes over the data about 18 times; 0.0 for N = 10M
because a 254k-parameter model that passes over 10M chars about 4 times has
little room to memorise them.

### Validation curves

`train` is the mean training loss since the previous point (with dropout active
for N = 1M). `valid subset` is the checkpoint-selection subset, **not** the
reported valid bpc: it is a harder passage (the 8-gram trained on 10M scores
1.783 on it against 1.737 on all of valid) and its windows give shorter
contexts. That is why it sits 0.03 (1M) and 0.12 (10M) above the final valid
number.

N = 1M:

| time (s) | step | lr | train bpc | valid subset bpc |
|---|---|---|---|---|
| 397 | 5,753 | 8.4e-4 | 2.1225 | 2.3279 |
| 446 | 6,485 | 5.2e-4 | 2.0901 | 2.3171 |
| 496 | 7,207 | 2.7e-4 | 2.0709 | 2.3010 |
| 546 | 8,029 | 1.1e-4 | 2.0479 | 2.2979 |
| 596 | 8,733 | 6.0e-5 | 2.0426 | 2.2931 |

N = 10M:

| time (s) | step | lr | train bpc | valid subset bpc |
|---|---|---|---|---|
| 998 | 14,290 | 8.4e-4 | 1.9975 | 2.1392 |
| 1,123 | 15,832 | 5.2e-4 | 1.9607 | 2.1103 |
| 1,247 | 17,738 | 2.7e-4 | 1.9250 | 2.0765 |
| 1,372 | 19,734 | 1.1e-4 | 1.8985 | 2.0598 |
| 1,496 | 21,645 | 6.0e-5 | 1.8817 | 2.0534 |

Full curves are in the `curve` field of each transformer line in `results.jsonl`.

### How the size was chosen (pilots, `pilots.jsonl`)

Twelve pilots of 120 s each, valid only (test was never touched). Each used the
same schedule compressed into 120 s.

| # | N | layers | d | batch | lr | dropout | q/k/v init | params | steps | tokens/s | load avg | valid bpc |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 1M | 2 | 96 | 16 | 2e-3 | 0 | 0.02 | 253,659 | 1,027 | 36.3k | — | 3.061 |
| 2 | 1M | 2 | 128 | 16 | 2e-3 | 0 | 0.02 | 436,507 | 703 | 25.0k | — | 3.225 |
| 3 | 1M | 3 | 128 | 16 | 2e-3 | 0 | 0.02 | 634,779 | 392 | 14.0k | — | 3.419 |
| 4 | 1M | 2 | 96 | 8 | 3e-3 | 0 | 0.02 | 253,659 | 1,246 | 22.3k | 4.6 → 5.0 | 3.402 |
| 5 | 1M | 2 | 96 | 8 | 3e-3 | 0 | Xavier | 253,659 | 2,296 | 40.5k | 4.5 → 3.2 | **2.397** |
| 6 | 1M | 2 | 96 | 8 | 6e-3 | 0 | Xavier | 253,659 | 2,417 | 42.7k | 3.2 → 3.1 | 2.423 |
| 7 | 1M | 2 | 96 | 4 | 3e-3 | 0 | Xavier | 253,659 | 4,171 | 36.8k | 3.1 → 3.1 | 2.433 |
| 8 | 1M | 2 | 96 | 8 | 3e-3 | 0.1 | Xavier | 253,659 | 2,052 | 36.1k | 1.9 → 2.3 | 2.603 |
| 9 | 10M | 2 | 96 | 8 | 3e-3 | 0 | Xavier | 253,659 | 2,102 | 37.1k | 2.2 → 4.3 | **2.373** |
| 10 | 10M | 3 | 128 | 8 | 3e-3 | 0 | Xavier | 634,779 | 1,027 | 18.3k | 4.4 → 4.7 | 2.808 |
| 11 | 10M | 4 | 128 | 8 | 3e-3 | 0 | Xavier | 833,051 | 665 | 11.9k | 4.7 → 4.9 | 3.443 |
| 12 | 10M | 4 | 192 | 8 | 3e-3 | 0 | Xavier | 1,839,387 | 162 | 3.0k | 5.1 → 5.7 | 3.549 |

Three further pilots of the first batch (4×128, 3×192, 4×192 with the 0.02
init) were killed part-way once the first three showed that every model was
still stuck at about 3.4 bpc; they left no record.

What the pilots showed:

- **Early training has a long plateau near 3.4 bpc** (bigram level) before
  attention starts working. With GPT-2 style 0.02 init on the query/key/value
  weights the plateau lasted most of the 120 s. Xavier init on those weights
  (the `nn.MultiheadAttention` default) cut it to about 30 s (pilot 4 vs 5).
- Batch 8 at lr 3e-3 was marginally best among pilots 5–7; the differences
  (0.03) are within what the varying machine load could cause.
- In 120 s the smallest model won by a wide margin at both sizes, because the
  larger ones had barely left the plateau.

The final runs therefore use 2 layers × d=96 for both N.

### Caveats for the transformer numbers

- **It is clearly undertrained.** At N = 10M the validation subset fell by 0.086
  over the last 500 s and flattened only because the learning rate was annealed
  to its floor. The result (1.979 test) is 0.18 bpc behind the 8-gram and behind
  the 6-gram too. The number measures what 25 contended CPU minutes buy, not
  what the architecture can do.
- **At N = 1M it is data-limited as well.** Training loss ends at 2.04 (with
  dropout active) against 2.26 on valid, and the validation subset is only
  creeping down (2.301 → 2.298 → 2.293 over the last 100 s).
- **The size choice rests on 2-minute pilots, which favour small models.** A
  120 s pilot mostly measures how fast a model escapes the plateau. Whether
  3 × 128 or 4 × 128 would beat 2 × 96 over the full 10 or 25 minutes was not
  tested, because full-budget comparisons were outside the pilot allowance. For
  N = 10M in particular a somewhat larger model could plausibly do better.
- **Dropout was set by reasoning, not by a pilot.** The one dropout pilot (#8)
  was worse at 120 s, as expected that early; it cannot show the benefit over
  18 epochs.
- **The machine was oversubscribed.** Other jobs were using about 3 of the 4
  cores for most of the session, so the 2 threads here were contended (load
  average up to 5.7). The same 2 × 96 model ran at 22k–43k tokens/s across
  pilots, and the 4 × 192 pilot ran at 3k tokens/s against 10k in a benchmark
  taken at lower load. That makes the pilot comparisons noisy, and the same
  budget would buy more steps on an idle machine.
- There is no dropout on the attention map itself. Sampling that mask made each
  step about 2.7× slower on CPU, so it was removed before any pilot.
- Single seed (0). No repeat runs, so run-to-run variance is unknown.
- Causality was checked directly (changing tokens ≥ 100 leaves logits at
  positions < 100 bit-identical), and the sliding evaluation was checked against
  a brute-force per-char recomputation on a short segment.

## Files

| file | what |
|---|---|
| `common.py` | split loading (fixed offsets) and JSONL logging |
| `ngram.py` | modified Kneser–Ney n-gram, orders 3–8 |
| `ppm.py` | PPM C/D, static and dynamic |
| `transformer.py` | transformer baseline; final configs are in `CONFIGS` |
| `results.jsonl` | one line per reported result (8 lines) |
| `pilots.jsonl` | the 12 transformer pilot runs |

Every `results.jsonl` line has `model`, `ntrain`, `mode`, `config`, `valid_bpc`,
`test_bpc`, `train_seconds`; transformer lines add `params`, `steps`,
`tokens_seen` and `curve`. Scripts append, so re-running adds lines.

Set `TEXT8_PATH` to point at the unzipped `text8` if it is not at
`/tmp/claude-0/data/text8`.
