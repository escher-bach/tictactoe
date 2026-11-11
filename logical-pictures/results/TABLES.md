## headline (test slice A = text8[95.0M:95.5M])

| Model | 100k | 1M | 10M |
|---|---|---|---|
| Kneser–Ney n-gram, order 8 (static) | 2.468 | 2.050 | 1.798 |
| PPM, best order (static) | 2.592 | 2.132 | 1.861 |
| PPM, best order (dynamic) | 2.027 | 1.926 | 1.787 |
| Transformer, 2 layers, 254k weights, 10 / 25 CPU-minutes (static) | – | 2.280 | 1.979 |
| **Picture machine (static)** | 2.249 | 1.723 | 1.550 |
| **Picture machine (dynamic)** | 2.091 | 1.674 | 1.536 |

## slice B = text8[97.0M:97.5M]

| Model | 1M | 10M |
|---|---|---|
| Kneser–Ney n-gram, order 8 (static) | 2.013 | 1.779 |
| PPM (static) | 2.090 | 1.842 |
| PPM (dynamic) | 1.909 | 1.766 |
| **Picture machine (static)** | 1.713 | 1.540 |
| **Picture machine (dynamic)** | 1.671 | 1.527 |

## ablations (records on 1M characters, test slice A)

| Operations allowed | Sorts admitted | Static | Dynamic |
|---|---|---|---|
| path only: `sign[-j]` | 12 | 1.918 | 1.824 |
| + classes of signs | 13 | 1.911 | 1.818 |
| + series (`run<C>`) | 13 | 1.850 | 1.761 |
| + recency inside keys, sameness | 17 | 1.746 | 1.697 |
| + pointer sorts | 19 | 1.727 | 1.675 |
| + classes of objects, learned keys (everything) | 19 | 1.726 | 1.675 |
| everything, margin 0.0003 | 22 | 1.723 | 1.674 |
| everything, keys of one determinable | 18 | 1.784 | 1.728 |
| everything, keys of at most two | 16 | 1.748 | 1.690 |

## synthetic worlds: bits at the determined positions (bits per sign overall)

| Model | respelled recall | analogy, signs of the corpus | analogy, signs never seen |
|---|---|---|---|
| picture machine, dynamic | 0.018 (2.35) | 0.004 (3.10) | 0.006 (2.95) |
| picture machine, static | 0.003 (2.34) | 0.004 (3.10) | 0.641 (5.59) |
| PPM-C order 5, dynamic | 3.181 (3.91) | 3.703 (5.31) | 7.255 (8.20) |
| PPM-C order 5, static | 3.303 (3.97) | 3.828 (5.37) | 16.868 (16.41) |
| transformer, static | 1.035 (2.04) | 0.022 (3.90) | 19.719 (19.08) |
