# The open problems

*The open problems of THEORY.md §10 and WITNESSES.md §8, treated one at a time.* THEORY is the file "Logical pictures"; WITNESSES is "Witness selection". Marks as there: **[P]** proved here, **[P✓]** proved and checked exactly on small cases ([checks/solutions_checks.py](checks/solutions_checks.py), output in [checks/OUTPUT.txt](checks/OUTPUT.txt)), **[K]** known and cited from memory, **[M]** measured on text8 by the learner of [ARCHITECTURE.md](ARCHITECTURE.md), **[C]** conjecture.

---

## 0. The answers

| Problem | Where | Result | Status |
|---|---|---|---|
| Peeling with a margin | W 8.2 | Theorem 3 survives a margin with no backtracking, under *hereditary* purity. Independent exceptions give hereditary purity once classes have about m/γ² members (Theorem 3ε) | **solved** [P✓] |
| Uniqueness of the partition | W 8.3 | There is no canonical partition into ε-good sets, even at rank 1. What is unique is everything the partition is used for: two good partitions predict every function of k witnesses alike off 2kε of the contexts (Theorem U) | **solved**, negatively for the partition and positively for its use [P✓] |
| Uniqueness of the tower | T 10.4 | For links that do not depend on the partition, every order of moves ends at one partition, the coarsest stable one. Recency links are not monotone, and that is the only source of order-dependence (§2.3) | **solved** for fixed links [P✓]; **open** for recency, with the obstruction named |
| Class counts independent of n | W 8.4 | Any-split with a size floor gives at most (8e·ln(4/ε)/ε)ᵗ classes, all ε-good outside εn contexts (Theorem 1″) | **solved** [P✓] |
| Choosing paths | W 8.5 | The types over all g^ℓ paths are computed by ℓ rounds of refinement, g hash passes each. Paths cost nothing in time. They cost in classes (Proposition 4) | **solved** for exact types [P✓] |
| The cancellation gap | W 8.6, T 10.2, T 10.8 | Signed sums are already in the theory twice: the read-out of T 5.5 is one, and any function of an integer form is a decision list over a scale. What is left is cancellation *inside a mixture*, which is a real and unconditional gap | **narrowed** [P✓ + K] |
| Occupancy | T 10.5 | "Unoccupied" is one more value of the determinable. Determinables that are never occupied together add rank by maximum, not by sum (Theorem 6) | **solved** [P✓] |
| The exception measure λ | T 10.6 | λ is not a free number. The rule it stands for is a fact about the sort, and is read off by counting over objects with the same record (Proposition 7) | **solved** [P✓]; it is Robbins's empirical Bayes |
| Ordered texts | T 10.7 | Inside one rigid text nothing is forced. The representation theorem applies to the array *texts × objects*, whose entries are sets of positions. It yields four kinds of fact, not three (Theorem 8) | **solved** [P, given K] |
| The rank of text; is language tame? | W 8.1, T 10.3 | The tree rank of a family built from d determinables is at most d, whatever the text. So rank cannot be the number everything turns on. The number that is measured instead is the length of the conjunctions the learner needs: three (§9) | **reframed** [P✓] and **measured** [M] |

The learner in [ARCHITECTURE.md](ARCHITECTURE.md) uses five of these directly: the margin (§1) as its admission test, the floor (§3) for its classes, refinement (§4) as the way paths are generated, occupancy (§6) as a value, and the profile rule (§7) as its only estimator.

---

## 1. Peeling with a margin

WITNESSES Theorem 3 is exact: a side must be pure. The proof used one property, that the rank of y on a subset is no larger than on the whole. A margin breaks it, because a subset of an almost pure set need not be almost pure. The repair is to ask for purity on every *cell*, and that is what independent exceptions deliver.

**[Def]** A *cell* is the set of contexts satisfying a conjunction of tests and negated tests. X is a cell. For a set A let imp(A) be the share of A outside its commonest label. Fix a margin θ and a floor n₀.

- Leaf(A) holds when |A| < n₀ or imp(A) ≤ θ.
- H(A, r) holds when there is a decision tree T of rank at most r such that every cell C ⊆ A on which T is constant satisfies Leaf(C). Write rk_θ(y | Y) ≤ r for H(X, r).

**The rule (peel with a margin).** Find(A, r): if Leaf(A), stop. If r = 0, fail. Otherwise look for a test s that divides A into two non-empty sides, and a side B of it, with Find(B, r − 1) succeeding. On the **first** such pair, commit, and return whatever Find(A ∖ B, r) returns.

**Theorem 3ε [P✓, check S1].** If rk_θ(y | Y) ≤ r, the rule succeeds whatever tests it tries first and never revisits a committed test. The tree it returns has rank at most r, every leaf of it satisfies Leaf, and its majority labels disagree with y on at most θn + n₀·L contexts, where L ≤ Σ_{i≤r} C(m, i) is the number of leaves. The cost is m^O(r) passes.

*Proof.* Every set the rule visits is a cell. *Heredity:* if H(A, r) holds with tree T and A′ ⊆ A is a cell, then H(A′, r) holds with the same T, since cells inside A′ are cells inside A. *Completeness*, by induction on |A|: assume H(A, r) with tree T and not Leaf(A). Then T is not constant on A, so r ≥ 1. Prune T to A. Its root test divides A into two non-empty cells, and one child has rank at most r − 1; call its side B. H(B, r − 1) holds with that child, so by induction Find(B, r − 1) succeeds, and the search for a pair ends. Let (s′, B′) be the pair actually committed to. A ∖ B′ is a smaller cell, H(A ∖ B′, r) holds by heredity, and by induction Find(A ∖ B′, r) succeeds. *Soundness:* a node with children of rank at most r − 1 and at most r has rank at most r. A leaf with imp ≤ θ contributes at most θ|A| errors and a leaf below the floor fewer than n₀. No test is used twice on a branch, so the depth is at most m and Lemma 2.1 of WITNESSES bounds L. ∎

**Corollary 3.1 (independent exceptions) [P✓, check S1].** Let y = T(x) with each label then flipped independently with probability η < ½, T of rank r. Take θ = η + γ and

> n₀ ≥ (m·ln 3 + ln(1/δ)) / (2γ²).

Then rk_θ(y | Y) ≤ r with probability at least 1 − δ.

*Proof.* There are at most 3ᵐ cells. On a cell with at least n₀ members on which T is constant, the number of flipped labels is binomial, and exceeds (η + γ)|C| with probability at most exp(−2γ²n₀) (Hoeffding). Sum over cells. ∎

**What it says.**

- The exception model is the theory's own (THEORY Proposition 5.2: each coordinate wrong independently). Under it nothing in WITNESSES §5 changes except that a class must have about m/γ² members before it is believed. That floor is the "exact count test with a margin" of THEORY 7.2, with its size now stated.
- The theorem does not need the noise to be random. It needs purity to be hereditary on cells. Adversarial exceptions are exactly those that are not, and for them even rank 1 is believed hard **[K]**.
- For labels with more than two values nothing changes: imp is defined for any label set.

---

## 2. Uniqueness

### 2.1 The partition is not unique

**[Def]** For an ε-good set A with ε < ½, every test is nearly unanimous on A. Its *generic type* is the tuple of majority values.

**Example [P✓, check S2].** One test s and three blocks: 100 contexts where s fails, 100 where it holds, and a block B of ten, five each way. With ε = 0.1 both

> {first block ∪ B, second block} and {first block, B ∪ second block}

are partitions into ε-good sets with distinct generic types. B alone is not good, and the whole set is not good. So there is no coarsest partition into good sets and no canonical one, already at rank 1. The two differ on a share of the contexts of order ε, and no refinement of the notion of goodness on single sets can remove that: B has no home.

### 2.2 What is unique

**Theorem U [P✓, check S2].** Let ε < ½.

1. *Closure.* Disjoint ε-good sets with the same generic type have an ε-good union. So every partition into good sets coarsens to one whose classes have pairwise different generic types.
2. *Separation.* Good sets A and B with different generic types satisfy |A ∩ B| < ε(|A| + |B|).
3. *Predictive uniqueness.* Let P and Q be any two partitions into ε-good sets and y any function of any k tests. The majority predictions of y from P and from Q disagree on fewer than 2kε·n contexts.
4. At ε = 0 the partition is unique: it is the partition into types.

*Proof.* (1) For each test the exceptions in the union are the exceptions in the parts, fewer than ε|A| + ε|B|. (2) Take a test on which the generic types differ, say mostly true on A and mostly false on B. A member of both is an exception in A or an exception in B. (3) By WITNESSES Corollary 1.1 each prediction errs on fewer than kε·n contexts. (4) A 0-good set is a set of contexts of one type. ∎

**What it says.** An object is not identified by its extension. It is identified by what it predicts, and that is determined up to the margin. This is the position THEORY 2.2 already took in words ("an object is nothing beyond its criterion of identity and its record of what recurs with it"); Theorem U is the statement that the record is unique where the extension is not. For convergence between learners (THEORY 8.7, prediction 2) the consequence is a correction: learners should be compared on what their partitions predict, sort by sort, and can be expected to disagree about membership on a share ε of occurrences.

### 2.3 The tower

**[Def]** Let g₁, …, g_g be partial functions from positions to positions (links), fixed in advance. A partition is *stable* when two positions in one class have, under each link, images in one class (or both none). A *forced split* divides a class by where one link leads.

**Theorem (confluence for fixed links) [P✓, check S4].** Starting from any partition P₀, every sequence of forced splits ends, and all of them end at the same partition: the coarsest stable partition that refines P₀. It is the partition by types over all paths.

*Proof.* Let P\* identify x and y when P₀ cannot tell p(x) from p(y) for any path p. P\* is stable and refines P₀. Any stable partition that refines P₀ refines P\*, by induction on the length of p. A forced split only separates positions that P\* separates, because P\* is stable. So every partition reached is coarser than or equal to P\*, and one that admits no forced split is stable, hence equal to P\*. ∎

This is the partition-refinement theorem of automata minimisation and colour refinement **[K]**. For the theory it says that order-dependence does not come from path or conjunction.

**It comes from recency [P✓, check S4].** A recency link is defined from the partition it refines, and refining a criterion can coarsen what its link shows. Text `aaba`, criterion P = {0, 1, 3 | 2}, against the trivial criterion Q that it refines. Under Q the link is adjacency, and "the sign at the most recent earlier position of my class" separates position 1 (after a) from position 3 (after b). Under P the most recent earlier position of position 3 in its own class is position 1, the link skips the b, and positions 1 and 3 are no longer separated. So the refinement operator is not monotone, the fixed-point argument above does not apply, and two orders of admission can end at different forms.

**Status.** Uniqueness of the tower holds for the part of it generated by path and conjunction over a fixed set of links, and fails in general once recency is admitted. Theorem U limits the damage: any two towers whose final classes are ε-good for the same tests predict alike. What is not shown is that two orders of moves reach classes that are good for the same tests.

---

## 3. Class counts that do not grow with n

WITNESSES Theorem 1 bounds the classes by (ln n/ε)ᵗ. The n comes from the depth of the split tree, and the depth comes from splitting classes that are already negligible.

**The rule (any-split with a floor).** As any-split, but a class with fewer than μn members is never split.

**Theorem 1″ [P✓, check S3].** Let rk(R) = t ≥ 1, 0 < ε ≤ ½ and μ = ε²(ε/4)^{2t}. Whatever tests the rule picks:

1. the tree of splits has rank at most t;
2. it ends with at most (8e·ln(4/ε)/ε)ᵗ classes, a number that does not depend on n;
3. the classes that are not ε-good have fewer than εn members in total.

*Proof.* (1) is WITNESSES Theorem 1(1), whose proof does not mention the stopping rule. A class that is split has at least μn members and each child at most (1 − ε) of its parent, so leaves lie at depth at most D = ln(1/μ)/ε + 1, and by Lemma 2.1 there are at most N ≤ (eD/t)ᵗ of them. A class that is not good was left unsplit because it is small, so the bad classes have fewer than Nμn members. It remains to check Nμ ≤ ε. With L = ln(1/μ) we have D ≤ 2L/ε, and ln z ≤ (2t/e)·z^{1/(2t)} gives L ≤ (2t/e)·μ^{−1/(2t)}, so N ≤ (4/ε)ᵗ·μ^{−1/2} and Nμ ≤ (4/ε)ᵗ·μ^{1/2} = ε. For (2), L ≤ (2t + 2)·ln(4/ε), so eD/t ≤ 8e·ln(4/ε)/ε. ∎

So the unsupervised step of the learning rule returns a number of objects fixed by the margin and the rank. The price is a set of unclassified contexts of measure ε, which is the same price the regularity lemmas for bounded VC dimension pay **[K]**. Malliaris and Shelah obtain a bound of the same shape for stable graphs with no exceptional set, by a more careful choice of sets **[K]**; what is added here is that the choice of test stays free.

The theorem holds with weights: replace counts by any measure on contexts. The learner uses it that way, with objects weighted by how often they occur.

---

## 4. Choosing paths

With g links and depth ℓ there are g^ℓ paths, and WITNESSES 8.5 left their choice as a list of that length.

**Proposition 4 [P✓, check S4].** Let P₀ be a partition of the positions and let P_{j+1} put two positions together when P_j does and, for every link, P_j puts their images together. Then P_ℓ is exactly the partition by types over the tests "the P₀-class at the end of path p is c", for all paths p of length at most ℓ. Computing P_ℓ takes ℓ·g hash passes.

*Proof.* Induction on ℓ: a path of length j + 1 is a link followed by a path of length j. ∎

So paths are never enumerated. One level of the tower offers g × (classes of the level below) tests, not g^ℓ × (basic tests), and every longer path is a test of a later level. What grows with ℓ is the number of classes, not the time: with the single link "previous position", P_ℓ is the partition into (ℓ+1)-grams. So the cost of paths is paid in the same currency as everything else in WITNESSES, the number of classes, and Theorems 1 and 1″ bound it when the rank is bounded.

The learner generates candidates this way and no other: a candidate is always one admitted key and one more determinable, one link further out.

**What is not covered.** The proposition is exact. With a margin, a class one link away can be good for the tests of its own level and still be divided by a test two links away, and I have no bound on that loss under stability alone (§10, item 3). This is why the learner tests every extension on held-out text and does not trust the composition.

---

## 5. Cancellation

THEORY 10.2 feared that counting truth-grounds is positive arithmetic and cannot match signed sums at equal size. Three cases have to be kept apart.

**5.1 One layer of signed evidence is already in the theory [P].** THEORY Proposition 5.2 is a signed sum: log P(w | h) = const + Σ βᵢ hᵢ cᵢ(w), with hᵢ ∈ {−1, 0, +1}. The signs come from positive arithmetic. Independent facts (clause P4) multiply their truth-grounds, a probability is a ratio of two such products, and the logarithm of a ratio of products is a sum of terms of both signs. So "evidence against" needs no negative weights: it is a count in the denominator. Every linear threshold over independently excepted witnesses is a picture of size one per witness.

**5.2 Cancellation inside a count is a scale [P✓, check S5].** Let x₁, …, x_k be two-valued facts, w₁, …, w_k integers, S = Σ wᵢxᵢ, and g any function of S.

> g(S) is a decision list (rank 1) over the thresholds of the scale S, however many facts enter.
> Over the facts themselves, parity has rank k.

*Proof.* The threshold tests on one scale are nested, so "if S ≥ c₁ then v₁, else if S ≥ c₂ then v₂, …" computes any function of S. The second line is WITNESSES §5. ∎

So a sum with cancellation costs the picture theory a scale, and scales are primitives of the middle-period logical space (the ruler of THEORY 2.2). Parity is not an exception to the rank theory. It is rank 1 over the right scale and rank k over the wrong tests. Counting occurrences of a class along the order (clause P7) builds such scales without choosing a subset: the count is over positions, and the criterion says which.

**5.3 Finding the witnesses of a sum [K].** WITNESSES Proposition 5 found exact parities by elimination. Thresholds are at the other extreme: every relevant witness of a monotone threshold is individually correlated with the label under a product measure, so each is shown by a single-candidate count. Between the two lies the hard case for everyone.

**5.4 What remains [K].** A *mixture* with cancellation: a table of probabilities whose rank is small and whose non-negative rank is large. The gap is unconditional (for instance, the matrix (i − j)² has rank 3 and non-negative rank growing with its size), and it is the gap between weighted automata with negative weights and hidden Markov models. A picture represents a mixture as a count over worlds, which is positive, so on such a table a vector model is smaller. THEORY 10.8 is this same case: a kernel reads as thresholds at the cost of its non-negative rank, not its rank.

**Status.** The cost feared in THEORY 10.2 is paid for sums of evidence (5.1) and for sums that are counts (5.2). It is not paid for mixtures (5.4), and there LP3 is false as stated. Whether text contains such mixtures is not known. Nothing in the learner's residue so far points to one, which is weak evidence at best.

---

## 6. Occupancy

In language most determinables are unoccupied at most positions (THEORY 10.5).

**6.1 Unoccupied is a value [P].** Let A be a determinable of sort s, and let name_s be partial. Extend A to all positions with one more value, ⊥. The extended determinable is again a family of mutually exclusive, jointly exhaustive values: an exact transversal, a Satzsystem. Every test "A = v" is false where A is unoccupied. Occupancy itself is a two-valued determinable of the position, with its own exception measure. Nothing else in THEORY §4.3 changes, and WITNESSES Theorem 4.1 still gives rank 1.

**6.2 What occupancy does to rank.**

**Lemma [P✓, check S6].** rk(R) ≤ ⌊log₂ (number of types realised)⌋. The family of value tests "Aᵢ = v" on d determinables, with any number of values each, has rank at most d.

*Proof.* The leaves of a shattered tree of height t are 2ᵗ contexts of different types. For the second claim, on the branch of a shattered tree where every test holds, each test fixes the value of its determinable, and a second test on the same determinable could not go both ways below it; so the branch uses t different determinables. ∎

**Theorem 6 [P✓, check S6].** Let each context name an object of exactly one of S sorts, sort j having d_j two-valued determinables that combine freely. Then

> max_j d_j ≤ rk ≤ ⌊log₂ Σ_j 2^{d_j}⌋ ≤ max_j d_j + ⌈log₂ S⌉,

against Σ_j d_j if all the determinables were occupied together.

*Proof.* The upper bound is the lemma, with Σ_j 2^{d_j} types. The lower bound is WITNESSES Theorem 4.3 inside one sort. ∎

**What it says.** Recombination (clause P3) creates rank only among determinables that are occupied *together*. Sparse occupancy turns the product of THEORY 4.3 into a sum, and the rank of a sum is the rank of its largest term. This is the precise sense in which a language with very many determinables, few occupied at a time, is tamer than its number of determinables suggests. In the learner, unoccupied is a value of every key and has its own row in every record table.

---

## 7. The exception measure

THEORY 5.5 left one non-logical number per determinable, Carnap's λ, and 10.6 asked whether the tower removes it.

**Setting.** A sort with many objects. Each object shows a two-valued determinable at each of its occurrences. Names are arbitrary: the law is unchanged by permuting objects, and by permuting the occurrences of one object. By de Finetti's theorem **[K]**, each object o then has a propensity θ_o, the occurrences of o are independent given θ_o, and the θ_o are independent draws from one law G. G is a fact about the sort: part of α in the table of THEORY 4.2.

**[Def]** The *profile* of an object at an occurrence is its record so far: (a, b), the number of times each value was shown.

**Proposition 7 [P✓, check S7].**

1. The probability that an object with profile (a, b) shows the first value next is r(a, b) = E[θ^{a+1}(1−θ)^b] / E[θ^a(1−θ)^b], a function of the profile and of G only.
2. The λ-continuum is the case G = Beta(λ/2, λ/2) **[K]**. For G outside that family no single λ need reproduce r.
3. Among the objects that reached profile (a, b) and occurred again, the share that showed the first value converges to r(a, b) as their number M_{ab} grows, with error of order M_{ab}^{−1/2}.

*Proof.* (1) is Bayes's rule for an object whose record is (a, b); the order of the record does not matter by exchangeability. (3) Objects are independent given G and each contributes one draw with probability r(a, b). (2) The check exhibits a G with two kinds of object, nearly constant and free, for which the best λ misses r by 0.25 on small profiles. ∎

**What it says.** λ was a parameter only because the estimate was made one object at a time. The rule r(a, b) is a universal fact of the sort, it is read off by counting, and the counting is over all objects of the sort at once. The regress stops there: the table of profile counts is filled by the whole corpus, so whatever prior is put on *its* cells matters by 1/M_{ab}, and where M_{ab} is small the profile is large and r(a, b) is close to a/(a + b) anyway. In the check, the profile rule costs 0.6482 bits per fact against 0.6477 for the exact rule and 0.7091 for the best single λ.

This is Robbins's empirical Bayes, and at a = 0 it is the Good–Turing estimate **[K]**. It is derived here from the theory's one symmetry and is not an addition to it. It is the form in which "the pattern of exceptions is the next level's corpus" (THEORY 10.6) is true: the profile is a determinable of the object, objects with the same profile form a class, and that class has a record. The learner has no other estimator. Every probability it outputs is a count in a table indexed by profiles.

---

## 8. Ordered texts

THEORY 4.2 applied Aldous–Hoover to hidden sorts and noted that positions are ordered, not exchangeable.

**Proposition 8.1 (inside one text nothing is forced) [P].** Every law on strings of length n is the law of a text-world with one hidden sort whose law is invariant under permuting the hidden objects.

*Proof.* Hidden objects have no names. Given the positions, an object is the set of positions that name it, and a permutation of objects does not change any such set. So the invariance holds for every law. ∎

So there is no representation theorem for a single rigid text beyond what an added symmetry buys. Two symmetries are available.

**Proposition 8.2 (names of signs) [P✓, check S8].** A law on Vⁿ is invariant under relabelling the signs iff it is produced by drawing an equality pattern (the text-fact (Pos, <, ≈) of THEORY 5.1) from an arbitrary law Π, and then naming its k blocks by a uniformly random injection into V:

> P(x) = Π(pattern of x) / (|V|·(|V| − 1) ⋯ (|V| − k + 1)).

*Proof.* The orbit of x under relabelling is the set of strings with its pattern, and it has that many members. ∎

The second factor is THEORY Proposition 5.1 at level zero: each new name costs the logarithm of the number of names still free. The first factor is unconstrained. So this symmetry reduces the text to the text-fact and says nothing about the text-fact.

**Theorem 8.3 (the corpus) [P, given K].** Let a corpus be a family of texts d = 1, 2, … and let o = 1, 2, … index the objects of one universal sort. Let E_{d,o} be the set of positions of text d that name object o, together with what is shown there. Suppose the law of the array (E_{d,o}) is unchanged by permuting the texts and, separately, by permuting the objects. Then there are independent uniform variables α, (ξ_d), (η_o), (ζ_{d,o}) and a fixed function f with

> E_{d,o} = f(α, ξ_d, η_o, ζ_{d,o}).

*Proof.* The entries take values in a Borel space (finite sets of positions with finitely many values). This is the representation of separately exchangeable arrays (Aldous, Hoover; Kallenberg for general entries) **[K]**. ∎

Positions sit inside the entries. They are never permuted, so their rigidity costs nothing.

**[Reading]** The theorem gives four kinds of independent fact where THEORY 4.2 had three.

| Fact | Attached to | In a learner |
|---|---|---|
| α | the corpus | laws: what holds of all texts and objects |
| η_o | a universal object | the lexicon: what this object is in every text |
| ξ_d | a text | what this text is about |
| ζ_{d,o} | an object in a text | in-context facts: how this object figures here |

THEORY's "particulars" were ξ and ζ together, and η was folded into α. The split matters for the learner: a record that persists across texts estimates η, a record that starts empty in each text estimates ζ, and the two are different objects even when they have the same key. The learner keeps both (ARCHITECTURE §2.4).

**Limits.** The theorem is for infinite arrays. For a finite corpus it can only hold approximately; the known bounds of that kind are for sequences **[K]**. Exchangeability of texts is a fact about how a corpus was assembled and fails for a corpus in temporal order. As with Aldous–Hoover, f is not unique.

---

## 9. The rank of text

WITNESSES 8.1 made everything turn on one number: the tree rank of the relation between positions and the tests the tower generates.

**9.1 That number is bounded by construction [P✓].** By the lemma of §6.2, the value tests on d determinables have rank at most d. The literal tests "the sign j back is a", for j ≤ J, are J determinables: rank at most J, for any text. A form with ten sorts whose keys conjoin at most three determinables generates relations of rank at most three. So the tree rank of a text relative to a tower is never large. It is at most the number of determinables in play, and it does not distinguish a tame language from a wild one.

Two things follow.

- The bound of Theorem 1 is true and useless at the sizes that matter: (ln n/ε)^J classes for J = 8 exceeds any corpus. A determinable with a hundred thousand values has rank 1. Rank bounds the *selection* problem, which is what WITNESSES asked of it. It does not bound the number of classes at a useful level, and it does not measure how much text there must be to fill their records.
- The quantity that governs cost is the one in WITNESSES Theorem 3: the rank r of the *dependence*, since candidates are tested in m^O(r) passes. In a form whose read-out is a list of sorts (rank 1 over sorts), r is the number of determinables a key conjoins.

**9.2 The measurement [M].** The learner of ARCHITECTURE.md was run on text8 with its keys limited to r determinables. Records were counted on one million characters; the figures are bits per character on a held-out half million, static and dynamic as defined there.

RANK_TABLE

**9.3 How analysed is the picture? [M]** In a fully analysed form every fact is either repeated or new and uniform (THEORY Proposition 5.1). For the two-valued facts of the learned picture that means a probability near 0 or 1, or near ½. The distribution of the probability the learned form gave to the fact that occurred, over the held-out facts:

ANALYSED_TABLE

**Status.** "Is language tame?" (THEORY 10.3) asked for bounded VC dimension after analysis. §9.1 shows that the tree-rank version of the question has a trivial answer. The version that is left is quantitative: how fast do classes fill, at keys of the length text needs? That is a learning curve, and ARCHITECTURE §5 reports the only two points of it measured here.

---

## 10. What is still open

1. **Uniqueness of the tower with recency** (§2.3). The operator is not monotone. A proof would need a different potential, or a normal form for recency criteria.
2. **Mixtures with cancellation** (§5.4). A real gap. The first thing to find out is whether any corpus exhibits it.
3. **Margins along paths** (§4). Composition of ε-good classes loses ε per link. No better bound is known to me under stability alone.
4. **Abstraction on real text** (ARCHITECTURE §6). The rule finds classes of signs and learned keys where the residue demands them, and on text8 at ten million characters the residue did not demand classes of words. Whether it does at a hundred times the text, or whether the rule's candidates are too poor to show it, is not settled.
5. **The cost in classes** (§9). Selection is solved and its cost is the rank. The number of records a form needs before they are filled is not bounded by any theorem here.

---

## 11. Sources

**Proved here.** Theorem 3ε and Corollary 3.1; Theorem U and the example of §2.1; the confluence theorem and the recency counterexample of §2.3; Theorem 1″; Proposition 4; §5.1 and §5.2; the lemma of §6.2 and Theorem 6; Proposition 7; Propositions 8.1 and 8.2. Theorem 8.3 is an application of a known theorem. Checks S1–S8 compare each with brute force on small instances, and all pass.

**Cited from memory [K].** Ehrenfeucht and Haussler 1989; Rivest 1987; Hoeffding 1963. Malliaris and Shelah 2014; Lovász and Szegedy 2010; Fox, Pach and Suk 2019. Paige and Tarjan 1987 (partition refinement); Hopcroft 1971. Chow 1961 (threshold functions and their first-order coefficients). The rank and non-negative rank of Euclidean distance matrices (Beasley and Laffey 2009; Hrubeš 2012); Yannakakis 1991. Robbins 1956; Good 1953; Johnson's sufficientness postulate (Zabell 1982). De Finetti; Hewitt and Savage 1955; Aldous 1981; Hoover 1979; Kallenberg 2005; Diaconis and Freedman 1980 (finite exchangeability).

**Measured [M].** §9.2 and §9.3, by the scripts in [experiments/](experiments/). They are measurements on one corpus at one size and are not checks of any theorem.
