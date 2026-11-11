# The picture machine

*The learner that THEORY.md and WITNESSES.md describe, built and run on text.* Code in [pm/](pm/), experiments in [experiments/](experiments/), every number below reproducible from the commands in §8. Marks: **[P]** proved (here or in [SOLUTIONS.md](SOLUTIONS.md)), **[M]** measured, **[Reading]** interpretation.

---

## 0. In brief

**What it is.** A text is turned into two-valued facts. A *form* is a list of *sorts*. Each sort is a criterion of identity, decided strictly before the position it is applied to, and keeps one record of counts per object. The probability of the next sign is read off the records of the objects the position names. Learning is analysis: produce candidate criteria from the objects already in the picture, and admit the one that removes most residue from text it has not counted. There are no vectors, no weights and no gradients anywhere; the only numbers are counts.

**What it does [M].** text8, bits per character on a held-out half million characters (text8[95.0M:95.5M]). "Static" freezes everything the corpus fixed; "dynamic" keeps counting through the test text. All models saw the same training text.

@@HEADLINE@@

**Against the two dogmas.**

| Dogma | Verdict |
|---|---|
| The logical picture theory is respected | Yes. Every component is a clause of THEORY or a theorem of WITNESSES or SOLUTIONS (§1). The representation of a context is the set of objects it names; nothing is a point in a vector space. The one place where two descriptions are combined uses the theory's own rule (independent facts add in log-odds, THEORY 5.2) as the default and a count wherever the corpus shows otherwise |
| Convergence to a classical method is failure | Mixed, and stated exactly in §6. The class of forms is not classical and the rule does find the non-classical objects where they are the whole story: a recency key learned across spellings, and a rule stated in sameness alone that transfers to signs never seen (§5.3). On text8 the machine beats the n-gram, PPM and, at the compute tried here, a small transformer (§5.1). But most of what it gains on text8 it gains from literal criteria. @@AUDIT_ONE_LINE@@ |

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

One sort is one left-to-right pass with a hash table: O(M) time, one slot per (object, node) seen. A form of S sorts reads text in S passes. Nothing is iterated to convergence. On the machine used here (4 CPU cores, no GPU) a form of 16 sorts reads 10.5 million characters in about 100 seconds.

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

Not supplied: what a word is, that the space matters, any offset, any context length, any class of letters. The boundary class of a run is chosen by the residue among thirty-six candidates.

---

@@EXPERIMENTS@@
