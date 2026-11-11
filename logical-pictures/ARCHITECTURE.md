# The picture machine

*The learner that THEORY.md and WITNESSES.md describe, built and run on text.* Code in [pm/](pm/), experiments in [experiments/](experiments/), every number below reproducible from the commands in §8. Marks: **[P]** proved (here or in [SOLUTIONS.md](SOLUTIONS.md)), **[M]** measured, **[Reading]** interpretation.

---

## 0. In brief

**What it is.** A text is turned into two-valued facts. A *form* is a list of *sorts*. Each sort is a criterion of identity, decided strictly before the position it is applied to, and keeps one record of counts per object. The probability of the next sign is read off the records of the objects the position names. Learning is analysis: produce candidate criteria from the objects already in the picture, and admit the one that removes most residue from text it has not counted. There are no vectors, no weights and no gradients anywhere; the only numbers are counts.

**What it does [M].** text8, bits per character on a held-out half million characters (text8[95.0M:95.5M]). "Static" freezes everything the corpus fixed; "dynamic" keeps counting through the test text. All models saw the same training text.

| Training text | 100k | 1M | 10M |
|---|---|---|---|
| Kneser–Ney n-gram, order 8 (static) | 2.468 | 2.050 | 1.798 |
| PPM, best order (static) | 2.592 | 2.132 | 1.861 |
| PPM, best order (dynamic) | 2.027 | 1.926 | 1.787 |
| Transformer, 2 layers, 254k weights, 10 and 25 CPU-minutes (static) | – | 2.280 | 1.979 |
| **Picture machine (static)** | 2.249 | **1.723** | **1.550** |
| **Picture machine (dynamic)** | 2.091 | **1.674** | **1.536** |

The transformer is a small one trained on a CPU for minutes and is undertrained (§5.1). The table does not show that the machine beats transformers. It shows that it beats the two strongest classical families by 0.2 to 0.3 bits at equal text, and a small transformer at equal compute on this hardware.

**Against the two dogmas.**

| Dogma | Verdict |
|---|---|
| The logical picture theory is respected | Yes. Every component is a clause of THEORY or a theorem of WITNESSES or SOLUTIONS (§1). The representation of a context is the set of objects it names; nothing is a point in a vector space. The one place where two descriptions are combined uses the theory's own rule (independent facts add in log-odds, THEORY 5.2) as the default and a count wherever the corpus shows otherwise |
| Convergence to a classical method is failure | Mixed, and stated exactly in §6. The class of forms is not classical and the rule does find the non-classical objects where they are the whole story: a recency key learned across spellings, and a rule stated in sameness alone that transfers to signs never seen (§5.3). On text8 the machine beats the n-gram, PPM and, at the compute tried here, a small transformer (§5.1). But most of what it gains on text8 it gains from literal criteria. Of the 2.42 bits the form removes below the unigram, sorts outside the classical rows of THEORY 3.3 remove about 0.05. The read-out, which is the part most directly taken from the theory, is what separates it from an n-gram using the same contexts (1.92 against 2.05) |

---

## 1. From the theory to the machine

| Theory | Machine | Source |
|---|---|---|
| The text-fact (Pos, <, ≈) | An array of sign ids. Only order and sameness are used | THEORY 5.1 |
| Tabulation (M3): signs recoded into determinables | The sign code: a tree of two-valued determinables, built by any-split on sign objects | THEORY 7.2, WITNESSES Thm 1 |
| Elementary fact | One branch taken at one node of the code at one position | THEORY 4.3 |
| Determinable, Satzsystem, unoccupied | A feature: one value per position, or none | SOLUTIONS 6.1 |
| Path, conjunction, recency | Feature operations (§2.2) | THEORY 5.4 |
| Series, "and so on" | The run of signs since the last sign of a class | THEORY P7 |
| Criterion of identity | A key: a conjunction of features | THEORY 5.2 |
| Object | The positions that share a key value | THEORY 5.2 |
| What recurs with an object | Its record: counts of the facts shown at its earlier occurrences | THEORY 2.2 |
| Universal and particular | Objects that exist when the corpus ends, and objects first named in the text being read; recency links | THEORY 5.2, SOLUTIONS Thm 8.3 |
| Objects as classes over witnesses | Classes of objects by any-split over their records, with a floor | THEORY 5.6, SOLUTIONS Thm 1″ |
| Shown identity (M1) | Admission: residue removed on text not counted, beyond a margin | THEORY 7.2, SOLUTIONS Thm 3ε |
| Exceptions (M2), rank-1 peeling | The chain: each sort is conjoined with the description so far and overrides it where its record is strong | WITNESSES Thm 3 |
| The tied read-out | Independence as the default of every table | THEORY Prop 5.2 |
| The exception measure | Profile tables; no λ | SOLUTIONS Prop 7 |
| Paths are not enumerated | A candidate is one admitted key and one more feature | SOLUTIONS Prop 4 |
| Residue | Bits of the held-out text | THEORY Prop 5.1 |

---

## 2. The machine

### 2.1 Facts

Signs are tabulated once, from bigram counts, into a binary tree: the *sign code*. Each internal node is a two-valued determinable of signs that is occupied only below its parent, so a sign is a path and a text of n signs is a sequence of M two-valued facts (text8: 4.75 facts per sign). The tree is built by the any-split rule on sign objects: a sign satisfies "followed by b" when it is so more often than its class, and a class is divided by a test that ε-splits it. WITNESSES Theorem 1 says the choice among such tests is free. It is: three different codes gave 1.918, 1.925 and 1.918 bits per character with everything else equal **[M]**.

The probability of a sign is the product of the probabilities of the facts on its path. This is the chain rule and assumes nothing. In particular the machine never treats coordinates of a sign as independent.

### 2.2 Features

A feature is a determinable of the context of position t, computed from the text strictly before t. A feature may be unoccupied. All features are produced by a small set of operations on what is already in the picture:

| Feature | Operation | Meaning |
|---|---|---|
| `sign[-j]` | path along adjacency | the sign j positions back |
| `class_d[-j]` | tabulation | its class at depth d of the sign code |
| `run<C>` | series | the signs since the most recent sign of class C, as one object. C is any class of the sign code or any frequent sign; which C is useful is learned |
| `run<C>-k` | path along the recency link of C | the k-th completed run before that |
| `last{key}` | recency | the sign that stood at the most recent earlier position naming the same key-object |
| `last2{key}`, `age{key}` | recency; a scale | the one before it; how long ago, on a logarithmic scale |
| `same(f, g)` | sameness of sign | whether two features show the same sign |
| `cls(f)` | abstraction | the class of f's object under any-split over its record, with a floor (SOLUTIONS Thm 1″) |
| `key(f)` | abstraction from shown identities | the class of f's value under the relation "what follows v repeats what followed the most recent w", admitted pair by pair by a count with a margin |

Adjacency is the recency link of the trivial criterion (THEORY 5.4), so `sign[-j]` is also a recency path. `run<C>` is clause P7: one operation applied to its own result until a boundary.

### 2.3 Sorts, objects, records

A **key** is a conjunction of features. A **sort** is a key with one record per object. There are two kinds.

- A *literal* sort records, for each object and each node of the sign code, how often each branch was taken at the object's earlier occurrences.
- A *pointer* sort has in addition a pointed feature that shows a sign, for instance `last{run<_>}`. It records how often the fact at this position *repeated* the corresponding fact of the pointed sign. Its facts are relative: "the same as there", "different". This is the distinction of THEORY Proposition 5.1 between repeated and new facts, kept in the record. A pointer sort is unoccupied wherever the pointed sign has already been excluded by the facts read so far.

A record is two counts. The machine never turns it into a probability by a formula. It turns it into a *profile*: the record itself when it is small, and (size level, ratio level) when it is large, 270 profiles in all. The profile is a determinable of the object.

### 2.4 Scope

SOLUTIONS Theorem 8.3 separates what an object is in every text (η) from how it figures in this one (ζ). The machine has both, and they are distinguished by when the object was first named, not by a flag.

- **Dynamic reading.** Every record keeps growing while the test text is read. Each fact is scored before it is counted.
- **Static reading.** The records of objects that exist when the corpus ends are frozen. Objects first named in the test text are particulars of that text, and so are recency links; they behave exactly as during training. Nothing the corpus fixed changes.

A particular can have a universal key. `last{run<_>}` is the most recent position at which the current word began the same way; `key(sign[-1])` in §5.3 is a class of signs learned from the corpus, used to find a position in the present text. That is the object THEORY 9.2 named as the first one outside the classical families.

### 2.5 The read-out

Each position names one object per sort. Their records have to be combined into one probability per fact, and this is where a classical model would back off, interpolate or mix by gradient. The machine does what the theory says.

**The chain.** Sorts are read in the order they were admitted. The description so far is a log-odds s for the fact at hand. The next sort contributes the profile q of its object. The pair (s rounded to one of 33 levels, q) is a *configuration*: an object of a further sort, whose record is a count of how such facts turned out in the corpus. The new description is that record. For a pointer sort the description is first turned to face the pointed fact, then conjoined, then turned back.

**The default is independence [P].** A cell of that table may have no record. THEORY Proposition 5.2 says what to believe then: if the new object's exceptions are independent of the description (clause P4), its evidence adds in log-odds. The evidence β_q of a profile is read off the table's own margin. So every cell starts at s + β_q, and counts move it wherever the corpus shows that independence fails. A cell with much data is purely a count; a cell with none is exactly the tied read-out. This one rule replaced "ignore the new object when unsure" and took the transfer to never-seen signs in §5.3 from 14 bits to 0.6 **[M]**.

**No λ [P].** The table is indexed by profiles, so it *is* the rule r(a, b) of SOLUTIONS Proposition 7, estimated over all objects of the sort at once, and conditioned on what the rest of the picture already said.

**A list with measures.** A sort whose object has no record leaves the description as it is. A sort whose object has a decided record overrides it. So the chain is the decision list of WITNESSES §5 ("if s₁ then v₁, else …", rank 1 over sorts) with a measure at each step, and the admission test of §3 is the peeling step of SOLUTIONS Theorem 3ε.

### 2.6 Cost

One sort is one left-to-right pass with a hash table: O(M) time, one slot per (object, node) seen. A form of S sorts reads text in S passes. Nothing is iterated to convergence. On the machine used here (4 CPU cores, no GPU) a form of 22 sorts reads 10.5 million characters in about three and a half minutes.

---

## 3. The learning rule

**State.** A form, the picture of the corpus in it, and the description of every fact.

**Candidates.** For each admitted key (and the empty key), and each feature in the pool: the key with that feature added, as a literal sort and as a pointer sort for each pointed feature in play. Bare pointers ("repeats the sign j back", "repeats what followed the most recent position with the same value of f") are candidates from the start. A feature enters the pool only when an operation of §2.2 can produce it from an admitted sort: one more step back from an offset in use, the previous run of a run in use, the recency link of an admitted key, the classes of an admitted run. So the candidate list is (admitted keys) × (pool), a few hundred, and never a list of paths or formulas (SOLUTIONS Proposition 4).

**The test.** A candidate is scored by the residue it removes: the bits saved on a slice of text that is *not* part of the text the records were counted on (for text8: counted on the first million characters, tested on 200,000 characters ninety million further on). It is admitted if it is the best candidate and removes more than a margin. This is the shown identity of THEORY 7.2, and the margin is the floor of SOLUTIONS Corollary 3.1.

**One move.** Score the new candidates; re-score the best old ones until the leader's score is current; admit the leader; let the operations add to the pool; repeat. Stop when no candidate passes the margin.

**What holds.**

- *Monotone and terminating* **[P]**. Each admission lowers the held-out residue by at least the margin, and the residue is bounded below.
- *Cost per candidate* **[P]**. One pass with a hash on its key.
- *No backtracking*. A sort is never removed. By SOLUTIONS Theorem 3ε this loses nothing when the dependence has rank 1 over the candidates and purity is hereditary. Beyond that it is a heuristic, and the place where it visibly bites is noted in §7.
- *Where it stops* **[P]**. At a form in which no candidate identity is shown: the residue is independent relative to the candidates, which is THEORY 7.3.
- *Not established*: uniqueness. SOLUTIONS §2.3 shows why (recency is not monotone) and what survives (Theorem U: forms that reach good classes for the same tests predict alike).

**What the rule is not.** It is not gradient descent: there is no parameter that is nudged. It is not EM: no posterior over hidden structure is kept. It is not program synthesis: no expression is enumerated. Its only primitive is "count, with this key, and see whether the residue drops".

---

## 4. What is supplied

For the no-dogma criterion, the complete list of what is not learned:

- the interface: order and sameness of sign;
- the operations of §2.2 (path, conjunction, recency, series, sameness, and the two abstractions);
- the profile scale (how records are rounded) and the 33 levels of a description;
- the margin of the admission test, and the floor and ε of the classes;
- the symmetry: names are arbitrary, used only through SOLUTIONS Proposition 7.

Not supplied: what a word is, that the space matters, any offset, any context length, any class of letters. The boundary class of a run is chosen by the residue among twenty-seven candidates.

---

## 5. Experiments

### 5.1 text8 [M]

**Protocol.** The form was learned once: records counted on text8[0:1M], admissions tested on text8[90.0M:90.2M], margin 0.0003 bits per character. The search took 11 minutes on three CPU threads and admitted 22 sorts. The same form was then used at every training size: its records and tables were recounted on text8[0:N] and the test slice text8[95.0M:95.5M] was read once. No model saw the test slice in training. Baselines are in [baselines/](baselines/) with their own README: a modified Kneser–Ney n-gram (orders 3 to 8, best on validation), PPM (orders 4 to 8, escapes C and D, with and without update exclusion, best on validation, static and dynamic), and a decoder-only transformer (2 layers, width 96, context 256, AdamW, chosen from twelve two-minute pilots).

The headline table is in §0. Four things about it.

1. **The transformer row is a weak reference.** It was trained for 10 and 25 minutes on two CPU threads of a loaded machine, its validation curve was still falling when the schedule ended, and its size was chosen by short pilots that favour small models. A transformer trained properly on a GPU would do better than 1.98 at ten million characters; with the whole of text8 the published figures are far below anything here. What the row supports is narrow: on this hardware and at these budgets, the machine is ahead by 0.4 to 0.55 bits. The machine's own cost was 11 minutes of search and 3.5 minutes to read the text.
2. **"Static" is not the same thing for every model.** For the machine it means that everything the corpus fixed is frozen; objects first named in the test text keep records, as §2.4 defines. The static n-gram and PPM have no such objects. The transformer has its 256 characters of context; the machine's recency links reach back without limit. The dynamic rows compare like with like for PPM.
3. **At 100,000 characters dynamic PPM is ahead** (2.03 against 2.09). The form was learned at a million characters, and its later sorts cost bits when their tables are filled from a tenth of that.
4. **The last admission is not robust.** The 22nd sort was admitted for 0.0004 bits at one million characters and costs 0.006 at ten million. The form learned with the coarser margin (19 sorts) gives 1.545 static at ten million.

**A slice never read during development.** Slice A was read many times while the machine was being built, and one design decision (independence as the default of a table, §2.5) was confirmed on it. So the comparison was repeated on text8[97.0M:97.5M], which nothing had touched.

| Training text | 1M | 10M |
|---|---|---|
| Kneser–Ney n-gram, order 8 (static) | 2.013 | 1.779 |
| PPM (static) | 2.090 | 1.842 |
| PPM (dynamic) | 1.909 | 1.766 |
| **Picture machine (static)** | **1.713** | **1.540** |
| **Picture machine (dynamic)** | **1.671** | **1.527** |

### 5.2 Which operations earn it [M]

Each row learns a form with only the operations named, records on one million characters, and reads slice A.

| Operations allowed | Sorts admitted | Static | Dynamic |
|---|---|---|---|
| path only: `sign[-j]` | 12 | 1.918 | 1.824 |
| + classes of signs | 13 | 1.911 | 1.818 |
| + series: `run<C>` | 13 | 1.850 | 1.761 |
| + recency inside keys, sameness | 17 | 1.746 | 1.697 |
| + pointer sorts | 19 | 1.727 | 1.675 |
| + classes of objects, learned keys | 19 | 1.726 | 1.675 |
| the same, margin 0.0003 (the headline form) | 22 | 1.723 | 1.674 |

- **The read-out.** With path alone the machine's criteria are literal windows, the same contexts an n-gram uses. It scores 1.918 against 2.050 for Kneser–Ney and 2.132 for static PPM. The difference is §2.5: profiles, configuration tables, independence as the default.
- **Series** is worth 0.07. The rule chose the space as the boundary by itself, and then runs between vowels and between classes of consonants.
- **Recency** is worth 0.10 inside keys and a further 0.02 as pointers.
- **Abstraction of objects is worth nothing here.** Classes of runs by any-split and learned keys were offered throughout. One class entered the headline form, at 0.0007 bits. This is the measured answer, on this corpus at this size, to THEORY's open problem 1.

### 5.3 Two worlds where the non-classical objects are the whole story [M]

[pm/worlds.py](pm/worlds.py). Bits at the positions that are determined by the rule of the world (ideal: 0), and in brackets bits per sign over the whole test text. PPM is method C, order 5. The transformer is the one of §5.1, trained for four CPU-minutes per world.

| Model | respelled recall | analogy, signs of the corpus | analogy, signs never seen |
|---|---|---|---|
| Picture machine, dynamic | 0.018 (2.35) | 0.004 (3.10) | 0.006 (2.95) |
| Picture machine, static | 0.003 (2.34) | 0.004 (3.10) | 0.641 (5.59) |
| PPM, dynamic | 3.181 (3.91) | 3.703 (5.31) | 7.255 (8.20) |
| PPM, static | 3.303 (3.97) | 3.828 (5.37) | 16.868 (16.41) |
| Transformer, static | 1.035 (2.04) | 0.022 (3.90) | 19.719 (19.08) |

**Respelled recall** (THEORY 9.2). Each episode binds four keys to values, written "k v", and then asks for each with the key in a different spelling, "K ?". The literal context of a question has never preceded its answer. The rule admitted, as its second sort,

> `(everything) => repeats last{key(sign[-1])}`

"The next sign repeats what followed the most recent sign of the same key-class", where the key-classes {k, K} were learned from the corpus by counting which pairs of signs show that repetition. This is a particular identified through a universal. The answers cost 0.003 to 0.02 bits. PPM pays 3.2, the entropy of a value. The transformer was partway there after four minutes.

Over the whole text the transformer is better (2.04 against 2.35; the ideal is 1.77). The difference is in the keys, not the answers: which keys can still be asked is a fact about a *set* of earlier positions, and no operation of §2.2 produces it.

**Analogy** (THEORY 4.4). Each episode is "a b b a ;" for a fresh pair of signs. The rule admitted

> `(everything) => repeats sign[-3]`, then `same(sign[-1], sign[-2]) => repeats sign[-3]`

a rule that mentions no sign, only which positions carry the same one. It therefore holds for signs that never occurred in the corpus: 0.006 bits. PPM cannot do this, and the transformer cannot either, since it has no trained embedding for a new sign. In static reading the machine pays 0.64 bits, all at the first fact of each new sign, where its frozen prior says that sign does not exist (§7).

Two things should be said about this world. First, it was built to favour the picture theory, and the transfer is the theory's own prediction, not a surprise. Second, the rule did not find the relational sort until admissions were tested on text written in new signs. With the admission slice in the corpus's own signs, memory of the pairs removed the residue first and the relational sort was never shown. That is the "literal absorption" THEORY 9.2 feared, observed. The remedy used, testing identities on text with new names, is the symmetry of THEORY 4.1 put to work, and it is a choice the experimenter made for this world.

### 5.4 How analysed is the picture [M]

SOLUTIONS §9.3 gives the distribution. In short: 42% of the held-out facts are given probability above 0.99 and carry 0.3% of the bits, and a fully analysed form would have the rest at one half. It does not. A quarter of the facts sit between 0.2 and 0.8 and carry 61% of the bits. The picture is well short of the bottom of analysis.

---

## 6. Is it a classical method?

**By THEORY's definition, no.** THEORY 3.3 counts a theory as collapsed when its criteria are fixed in advance or all come from one row of its table, and asks for four things together. The learned form has all four **[M]**:

| Requirement | In the form learned on text8 |
|---|---|
| Criteria that are learned | Every key was admitted by the rule. Nothing about words, spaces or offsets was supplied; the rule chose the space as a boundary among 27 candidates |
| Objects of both scopes | 12 literal sorts and 10 pointer sorts over recency links; objects first named in a text keep records within it |
| Criteria defined over objects already constructed | Six sorts are built over a recency object or a learned class, up to three levels deep (`same(last{cls(run<aoq>) & …}, last2{…}) => repeats last{… last{…} …}`) |
| Several objects named at one position | 22 |

**By weight, mostly yes.** The audit ([experiments/audit.py](experiments/audit.py)) assigns each admitted sort to a row of THEORY 3.3 and adds up the residue each row removed. Bits per character removed below the unigram, static:

| Row | Classical family | Sorts | 1M | 10M |
|---|---|---|---|---|
| Literal run: a key of runs and literal signs | n-gram and PPM, with a learned boundary | 8 | 2.264 | 2.453 |
| Plain cache: repeats the last occurrence of a literal key | PPM, LZ, cache models | 3 | 0.098 | 0.091 |
| Gated repeat: the same, keyed by a scale or a class | borderline: a cache with a learned condition | 4 | 0.029 | 0.034 |
| Key over constructed objects | none | 6 | 0.026 | 0.013 |

So about 0.05 of the 2.42 bits come from sorts outside the classical rows. The ablations say the same from the other side: the best form with literal criteria only reaches 1.850, and everything that involves recency takes it to 1.723, most of that through sorts a compression engineer would call caches.

**What this amounts to.**

1. The machine is not any one classical method. It contains three of the rows at once and sorts that are in none, and it is ahead of each classical baseline by a margin that the baselines' own tuning does not close.
2. On text8 it is, in substance, a word-level literal model with caches, discovered from order and sameness alone, and read out by counts. Abstraction, which is what the picture theory is for and what a transformer is believed to add, carried almost none of the gain. If "not converging to a classical method" means that the non-classical objects do the work, that has **not** been shown on real text.
3. Two parts of the result do come from the theory and from nothing classical. One is the read-out: profile tables in place of smoothing (SOLUTIONS Proposition 7) and independence as the default in place of back-off or a trained mixer (THEORY Proposition 5.2). It accounts for the gap between the path-only form and the n-gram. The nearest classical relative is context mixing, whose state maps are engineered and whose mixers are trained by gradient. The other is that the criteria are an output: the form is found, in minutes, by one rule.
4. Where literal criteria cannot remove the residue, the rule does find the objects the theory names: a learned key for a recency link, and a rule in sameness alone (§5.3).

The first dogma is therefore met for the class of forms and for the worlds of §5.3, and not met in its strong sense on text8. I would not describe the text8 result as a non-classical learner beating a transformer. I would describe it as a count-only learner, derived from the picture theory, that selects its own literal criteria and reads them out better than the classical methods do.

---

## 7. Limits

1. **Abstraction on real text is unproven** (§6, and SOLUTIONS §10 item 6). Either ten million characters do not demand classes of words, or the two abstraction operations are too weak to show the demand. The experiment that would tell them apart is the same run at a hundred times the text, which this implementation cannot hold in memory.
2. **The rule is greedy.** A candidate adds one feature to one key. A dependence that shows only when two new features are added together is invisible to it, which is the rank-2 case of WITNESSES Theorem 3 and would cost the square of the pool. Sorts are never removed, and the last admissions are not robust (§5.1, point 4).
3. **Relative facts should be read first.** A pointer sort says "the same as there", but the machine still spells the sign in a fixed code and lets the pointer speak at each node. So a sign the corpus never showed costs the frozen prior of its first node, however certain the pointer is (0.64 bits in §5.3). THEORY Proposition 5.1 suggests the order: decide "repeated or new" before anything about which sign.
4. **No sets.** The operations produce an object from a position. "Which of the bound keys has not yet been asked" is a fact about a set of positions, and the machine pays for it (§5.3).
5. **The rounding of records and descriptions is supplied** (§4). SOLUTIONS Proposition 7 says why some rounding is unavoidable; it does not say which.
6. **One corpus, two sizes, one small alphabet.** The transformer comparison is at CPU budgets only.
7. **The uniqueness of the learned form is not known**, and SOLUTIONS §2.3 shows it cannot be had in general once recency is present.

---

## 8. Reproducing

```
python3 checks/causality.py                      # no future is seen; probabilities sum to 1
python3 experiments/learn_text8.py --out runs/form_fine.pkl --min-gain 0.0003 --max-sorts 48
python3 experiments/eval_text8.py --form runs/form_fine.pkl --n 10000000 --modes static
python3 experiments/eval_text8.py --form runs/form_fine.pkl --n 10000000 --modes dynamic
experiments/ablations.sh                         # section 5.2 and SOLUTIONS 9.2
python3 experiments/synthetic.py                 # section 5.3
python3 experiments/synthetic_baselines.py 240
python3 experiments/audit.py --form runs/form_fine.pkl
python3 experiments/analysed.py --form runs/form_fine.pkl
python3 experiments/report.py                    # all tables, into results/
```

The numbers as written by these runs are in [results/](results/). The ten-million-character runs need about 10 GB of memory, one mode per process.

