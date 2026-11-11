# Logical pictures

This folder continues two notes that are not in it: **THEORY** ("Logical pictures": the picture theory of meaning as a theory of representation) and **WITNESSES** ("Witness selection").

| File | What |
|---|---|
| [FOUNDATIONS.md](FOUNDATIONS.md) | The current state. What the two notes are, why the first attempt to build a learner from them failed, and what is proved before anything is run: kinds as the classes of exchange-invariance of the law (canonical), the quotient theorem, the pointer form, why this is not a classical method at the level of signs, and why it is the form transformers have. Ends with the open problem that blocks any experiment: composite objects |
| [SOLUTIONS.md](SOLUTIONS.md) | Earlier work on the open problems listed in the two notes: peeling with a margin, uniqueness, class counts, paths, cancellation, occupancy, the exception measure, ordered texts. Refereed and corrected. Partly superseded by FOUNDATIONS.md, as its header says |
| `checks/foundations_checks.py` | Brute-force checks of the theorems of FOUNDATIONS.md on small cases |
| `checks/solutions_checks.py` | The same for SOLUTIONS.md |

There is no learner, no experiment and no benchmark here. An earlier version had all three; its objects were supplied by hand (a word-splitter with a menu of boundaries, and spelling tests), so its results said nothing about whether logical representations can be learned, and it has been removed. FOUNDATIONS.md §6.3 states what has to be proved before an experiment would mean anything.

```
python3 checks/foundations_checks.py
python3 checks/solutions_checks.py
```
