# Logical pictures: the open problems, and a machine that learns

This folder continues two documents that are not in it: **THEORY** ("Logical pictures", the picture theory of meaning as a theory of representation) and **WITNESSES** ("Witness selection"). It does two things.

1. **[SOLUTIONS.md](SOLUTIONS.md)** takes the open problems of both documents one at a time. Each is either proved, proved false as posed and replaced, or narrowed with the remaining gap named. Every proof is checked against brute force on small cases by [checks/solutions_checks.py](checks/solutions_checks.py).
2. **[ARCHITECTURE.md](ARCHITECTURE.md)** turns the theory into a learner, the *picture machine*, and runs it: on text8 against an n-gram, PPM and a small transformer, and on two small worlds built so that only the non-classical objects of the theory can solve them.

Two rules were set for the work and are audited in ARCHITECTURE §0 and §6: the learner must not be a classical method under another name, and nothing may be substituted for the logical picture.

## Layout

| Path | What |
|---|---|
| `SOLUTIONS.md` | the open problems: theorems, proofs, counterexamples |
| `ARCHITECTURE.md` | the machine, the learning rule, the experiments, the collapse audit |
| `checks/solutions_checks.py`, `checks/OUTPUT.txt` | exact checks of the theorems (S1–S8) |
| `checks/causality.py` | the machine sees no future, and its probabilities sum to one |
| `pm/kernels.py` | the passes: records, profiles, tables, recency, series |
| `pm/machine.py` | sign code, features, sorts, the learning rule, reading a text in a learned form |
| `pm/worlds.py` | the two small worlds |
| `experiments/` | every run reported in ARCHITECTURE §5 |
| `baselines/` | Kneser–Ney n-gram, PPM, transformer (reference models; their own README) |
| `results/` | the numbers, as written by the runs |

## Reproducing

Needs Python 3 with numpy and numba (torch only for the transformer baseline), and `text8` under `$PM_DATA` (default `/tmp/claude-0/data`; the scripts unzip `text8.zip` from https://mattmahoney.net/dc/text8.zip if it is there).

```
python3 checks/solutions_checks.py            # the theorems, ~30 s
python3 checks/causality.py                   # causality and normalisation of the machine
python3 experiments/learn_text8.py --out runs/form.pkl --min-gain 0.0003 --max-sorts 48
python3 experiments/eval_text8.py --form runs/form.pkl --n 10000000
experiments/ablations.sh                      # which operations earn the result
python3 experiments/synthetic.py              # the two small worlds
```
