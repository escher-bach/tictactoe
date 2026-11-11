# Logical pictures

This folder continues two notes that are not in it: **THEORY** ("Logical pictures": the picture theory of meaning as a theory of representation) and **WITNESSES** ("Witness selection").

| File | What |
|---|---|
| [FOUNDATIONS.md](FOUNDATIONS.md) | The current state. What the objects of a law are, proved: kinds (the classes of exchange-invariance), the quotient theorem, the pointer form, the least memory of a past, and the canonical machine of states up to renaming. Why that is not a classical method and why it is the form transformers have. Units in the limit. Then the learner those theorems license and what it learns |
| `nominal/learn.py` | The learner: states, memories and their symmetries from order, sameness and counts |
| `nominal/worlds.py`, `nominal/run.py`, `nominal/OUTPUT.txt` | Three worlds with a known generating machine, and the check that the learned representation is that machine |
| `checks/foundations_checks.py` | Brute-force checks of the theorems of FOUNDATIONS.md on small laws |
| [SOLUTIONS.md](SOLUTIONS.md), `checks/solutions_checks.py` | Earlier work on the open problems listed in the two notes. Refereed. Partly superseded, as its header says |

**Status, stated plainly (FOUNDATIONS.md §10).** The theorems about objects stand. The learner is induction of a state machine with a memory of names: by the source notes' own four-part test that is a classical method with one column added, and its size is the product of whatever is independent in the law, so it cannot scale to text. The missing theorem is the decomposition of a state into independent situations with shared kinds.

There is no benchmark here. An earlier version had a learner whose objects were supplied by hand; it has been removed. The experiment that is here is on texts whose signs are already the units. Nothing yet licenses one on raw bytes, and FOUNDATIONS.md §7 and §10 say what is missing.

```
python3 checks/foundations_checks.py
python3 nominal/run.py
```
