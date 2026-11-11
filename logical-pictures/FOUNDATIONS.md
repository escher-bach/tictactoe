# Foundations, again

*A restart. The learner that was in this folder has been removed: it supplied its own objects by hand and then reported scores. This file goes back to the two source notes, THEORY ("Logical pictures") and WITNESSES ("Witness selection"), says what they are, says exactly where the theory was too immature to build on, and proves what can be proved before anything is run on text.* Marks: **[Def]**, **[P]** proved here, **[P✓]** proved and checked by brute force on small cases ([checks/foundations_checks.py](checks/foundations_checks.py)), **[K]** known, cited from memory, **[C]** conjecture, **[Reading]** interpretation. Nothing in this file is a measurement on data.

---

## 0. Where this stands

**The thesis of the two notes, in one sentence.** A corpus has a logically perfect language: a description in terms of objects, their forms and the facts about them, in which everything that is not a new fact is shown by the form. Learning is finding that language from the raw signs, by experience, and nothing may be put in but order and sameness.

**What is established here.**

| | Statement | Status |
|---|---|---|
| 1 | Scores cannot decide the matter. In the limit of text every literal method reaches the entropy rate, so "does not converge to a classical method" is not a claim about asymptotic residue. It is a claim about which objects are identified, and about the text needed (§2.1) | [P, given K] |
| 2 | Tree rank does not keep literal memory out. Recurrence does: a universal is an object with a share of the texts, and there are boundedly many (§2.2, §2.3) | [P✓], [P] |
| 3 | **Kinds are canonical.** "The law is unchanged by exchanging a and b" is an equivalence on signs, and its classes are the coarsest partition under which names are arbitrary. No margin, no choice, no gauge (§3.1) | [P✓] |
| 4 | **The quotient theorem.** A law with kinds is a law on *kinded patterns* (the kind at each position, and which positions carry the same sign) times uniformly chosen names. This is the perfect language at the level of signs (§3.2) | [P✓] |
| 5 | **The pointer form.** Under kinds, the next sign either repeats the sign of an earlier position of its kind or is a uniformly new name of its kind, and which of these depends on the past only through its kinded pattern. Reference to earlier positions is forced by the symmetry; it is not a design choice (§3.3) | [P✓] |
| 6 | **Not classical, at this level.** Where kinds are open, literal records are empty, and a model with kinds but without sameness loses the logarithm of the kind's size at every repeated name (§4) | [P] |
| 7 | **The same form as a transformer's.** The optimal predictor of a law with kinds is a pointer over earlier positions selected by kind, plus a generator of new names: one attention layer with tied input and output codes. And every criterion a text can define in first-order logic with order is one a hard-attention transformer computes, and conversely (§5) | [P] for the form; [K] for the correspondence |

**What is not established, and blocks any experiment.**

Statements 3 to 7 are about the *signs of the text as given*. On raw bytes the signs are 256 bytes, their kinds are few and nearly trivial, and all the interesting objects are composite. The earlier build crossed that gap by writing a word-splitter and calling it a learned criterion. The theory of composite objects is the open problem (§6): I can state what a unit must be and lift every theorem above to units *once the units are given*, and I cannot yet prove that the units of a law are canonical or say how they are found. Until that is a theorem there is nothing honest to run.

So, against the two conditions set for this work:

- **(A) Does not converge to a classical method.** Proved for laws whose kinds are kinds of signs (§4). Open for composite objects.
- **(B) Comparable with transformers.** The form the theory forces is the form transformers have (§5). That is a statement about representation. Nothing is claimed about learning dynamics.

---

## 1. What the two notes are

### 1.1 THEORY

A text is a fact: positions, their order, and which positions carry the same sign (THEORY 5.1). A *picture* of it is a many-sorted structure built over the positions: sorts whose objects are classes of positions under a criterion of identity, determinables of those objects, links between them (THEORY 4.3, 5.2). Two lemmas give the theory its shape. Lemma A: independence and truth-functions exclude nothing, so all the content is in the clauses about objects. Lemma B: in an analysed language the past informs the future only because the future names the same objects.

Read as model theory, a picture is an **interpretation**. The sorts are imaginary sorts of the text structure (quotients by definable equivalences); the tower of THEORY 5.4 is a constructional system, a chain of definitions; and the condition that a criterion be decided strictly before the position makes the definitions formulas of the *past*. The "fully analysed" picture is one whose facts are independent given its form. For exchangeable data such a picture exists: that is the content of the Aldous–Hoover theorem as THEORY 4.2 reads it, with the function f as the picture and its independent arguments as the elementary facts.

So the programme is: **find a past-definable interpretation of the text under which what remains is independent.** The logically perfect language is the language of the interpreted structure.

### 1.2 WITNESSES

WITNESSES asks one question of that programme: how are the defining parameters of a criterion (the witnesses) selected? Its answer is that selection is free when the relation between contexts and tests has finite tree rank (stability), costs m^O(r) for a dependence of rank r, and that recombination is what creates rank.

### 1.3 What neither note contains

Neither says what an *object* is before a criterion for it has been written down. THEORY 5.4 generates criteria by three operations from "objects already in the picture", and at the bottom the only objects are the signs. Everything depends on what the first non-trivial objects are, and the notes leave it to the learning rule: "wherever the residue is not independent, name the object responsible". That sentence is where the earlier build went wrong, in a way the next section makes exact.

---

## 2. Why the build failed, as theory

### 2.1 Residue cannot be the judge

**Proposition 2.1 [P, given K].** Let the source be stationary and ergodic with entropy rate h. The residue per sign of the literal window of length k tends to H_k, the conditional entropy of a sign given the k before it, and H_k decreases to h. No causal predictor of any kind has residue below h.

*Proof.* The first statement is the law of large numbers for counts; H_k ↓ h is the definition of the entropy rate; the last is Shannon's theorem. ∎

So with enough text, literal windows are as good as anything. A rule that admits "the candidate that removes most residue" and prices nothing else will, with enough text, admit literal windows, and it will be right to by its own standard. Two consequences.

- "Does not converge to a classical method" cannot mean "ends with less residue". In the limit there is no less. It can only mean that the learner identifies objects the classical families do not have, and needs less text to do so.
- A learner selected by residue has to be judged by what it identified. The earlier build was judged by bits per character, and that number rewarded exactly the literal objects.

### 2.2 Rank does not keep literal memory out

WITNESSES §7 proposed a rank budget as the capacity control, on the argument that literal offsets recombine and so have high rank.

**Lemma 2.2 [P✓, check F4].** The value tests on J determinables have tree rank at most J, whatever the data. In particular the tests "the sign j back is a", j ≤ J, have rank at most J.

*Proof.* On the branch of a shattered tree where every test holds, each test fixes the value of its determinable and no second test on that determinable can go both ways below it. ∎

A window of eight signs has rank at most eight, and a determinable with a million values has rank one. A rank budget of any useful size admits every literal sort. So the guard the notes relied on does not exist, and the build had none.

### 2.3 What does keep it out: a universal is dense

**Proposition 2.3 [P].** Let the texts of a corpus be exchangeable. Then each object occurs in a fixed share of the texts, and an object whose share is zero never occurs. If a text names at most L objects of a sort, at most L/μ objects of that sort have share μ or more.

*Proof.* For a fixed object the indicators "named in text d" are exchangeable, so by de Finetti they have a limiting frequency and are independent given it. The shares of all objects sum to at most L. ∎

A record is a count over an object's occurrences in different texts, so it estimates something only for an object with a share, and to a precision set by that share times the size of the corpus. A sort that keeps a record for every literal window, most of them seen once, is keeping records that estimate nothing: the earlier build had millions. The capacity control the theory supplies is therefore not a bound on rank. It is this: **a universal is an object that recurs in a fixed share of texts, and there are at most L/μ of those. What does not recur is a particular, and a particular has no record outside its own text.** It can only be reached through a universal, which is the object THEORY 9.2 asked for.

### 2.4 ε-goodness is not identity

WITNESSES makes a class "one object" when no single test divides it by more than a margin. SOLUTIONS §2 showed that this notion has no canonical partition and that a determinable with many rare values is "one object" under it. For a law that is not deterministic it is the wrong notion: two signs can agree on every test up to the margin and be different objects, and two occurrences of one object disagree on every test that is not determined. Identity has to be defined on the law, not on near-unanimity of tests. §3 does that.

### 2.5 The five faults, in one list

1. Objects were supplied (a word-splitter with a menu), because the theory did not say what the first composite objects are.
2. Admission was by residue, which rewards literal objects (2.1).
3. Nothing excluded literal sorts (2.2), and records were free, against 2.3.
4. Classes were formed by near-unanimity of tests (2.4), and only for frequent objects, where a class adds nothing.
5. The only facts were facts about signs, so nothing had to be said about kinds.

Each is a gap in the theory before it is a bug in the code.

---

## 3. Kinds

THEORY 4.1: "an object has no nature beyond its form. Two objects of the same form differ only in being two. The law of the world is invariant under any permutation of objects that preserves form." Take that as the definition instead of as a remark.

### 3.1 Kinds are canonical

Let Σ be the set of signs and P a law on texts over Σ. A permutation of Σ acts on texts sign by sign.

**[Def] Kind.** Signs a and b are *of one kind* when P is unchanged by exchanging them.

**Theorem 3.1 [P✓, check F1].** Being of one kind is an equivalence relation. Let B\* be its partition. Then P is invariant under every permutation that maps each class of B\* to itself, and a partition has that property iff it refines B\*.

*Proof.* Reflexive and symmetric by definition. If exchanging a, b and exchanging b, c each preserve P, so does their composite (a b)(b c)(a b), which exchanges a and c. The permutations that preserve each class of a partition are generated by exchanges within classes, so they preserve P iff every such exchange does, that is, iff every class lies inside a class of B\*. ∎

This is what was missing. The kinds of a law are *defined by the law*. There is one partition, the coarsest under which names are arbitrary. It involves no margin and no choice of tests, so the non-uniqueness of SOLUTIONS §2 does not arise, and it has no gauge of any kind: not a rotation, not even a relabelling, since a partition has no labels.

In the terms of model theory, kinds play the part of types over the empty set on the sort of signs: they are the classes of the exchange relation, and they refine the orbits of the group of all permutations that preserve the law. A law with finitely many kinds on an unbounded stock of signs is the analogue of an ℵ₀-categorical theory: for each length there are finitely many kinded patterns (Ryll-Nardzewski) **[K, Reading]**.

### 3.2 The quotient theorem

**[Def] Kinded pattern.** For a text x and a partition B of the signs, τ_B(x) records, for each position, the kind of its sign, and for each pair of positions whether they carry the same sign. It is the text-fact (Pos, <, ≈) of THEORY 5.1 with one determinable added.

**Theorem 3.2 [P✓, check F2].** P is invariant under the permutations preserving B iff there is a law Q on kinded patterns with

> P(x) = Q(τ_B(x)) / ∏_i |B_i|·(|B_i| − 1) ⋯ (|B_i| − k_i(x) + 1),

where k_i(x) is the number of distinct signs of kind i in x. Q is arbitrary.

*Proof.* The orbit of x is the set of texts with its kinded pattern, and it has that many members: the names of each kind can be chosen injectively in that many ways. ∎

**[Reading]** This is the perfect language at the level of signs, and it is THEORY's Proposition 5.1 without the assumption of exactness. A text says two things. It says its kinded pattern, and the law of that is the whole of what the corpus fixed. And it introduces names, each of which costs the logarithm of the number of names of its kind still free, and nothing else can be said about a name. With one kind this is SOLUTIONS Proposition 8.2. With every sign its own kind it is empty: no symmetry, no analysis. The content of a corpus's form, at this level, is where it lies between the two.

### 3.3 The pointer form

**Theorem 3.3 [P✓, check F3].** Let P be invariant under the permutations preserving B. Given the text before position t, the next sign is distributed as follows. There are numbers q_j ≥ 0, one for each distinct sign already used, and ν_i ≥ 0, one for each kind, summing to one, such that

- with probability q_j the next sign *is the sign at the earlier positions of block j*;
- with probability ν_i it is a sign of kind i not yet used, uniformly among those.

The numbers q and ν depend on the text so far only through its kinded pattern. Conversely any such family of numbers defines a law with kinds B.

*Proof.* The permutations that preserve B and fix every sign already used act transitively on the unused signs of each kind and preserve the conditional law, so unused signs of one kind are equally probable. The probabilities of "repeat block j" and "new of kind i" are ratios of probabilities of prefixes, and by Theorem 3.2 those depend on the kinded pattern only. The converse is Theorem 3.2 read forwards. ∎

**What it says.**

- **Reference is forced.** In a world with kinds, the only thing that can be predicted about *which* sign comes next is that it is the same as an earlier one. So the predictor must point at earlier positions. This is Lemma B made exact: the past informs the future's identity only through re-identification, and here re-identification is a link to a position.
- **The link is keyed on a universal.** Which block is repeated is a function of the kinded pattern. No name enters it. So the object reached is a particular (this text's name) and the criterion that reaches it is stated in kinds: "a particular identified through a universal" (THEORY 9.2) is not one admissible candidate among others. It is the general form.
- **Abstraction and reference are one theorem.** Kinds (3.1) are the abstraction; pointers (3.3) are the in-context learning; and the second is the shape the law must have because of the first.

### 3.4 What the symmetry does not give

Three limits, stated now so they are not discovered later as excuses.

1. **Exactness.** In a real corpus no two signs are exactly exchangeable. "Unchanged up to a margin" is not transitive, so with a margin the canonical partition becomes a hierarchy of finer and finer near-kinds, and Theorem 3.1 holds exactly only at each level of exactness. This is a real loss and I have no theorem that controls it.
2. **Testing.** Whether exchanging a and b preserves P involves statistics of every order. For a source of bounded memory it is decided by statistics of that order **[P]**: a stationary Markov law of order k is determined by its (k+1)-sign marginals, so it is invariant iff they are. In general a learner can only test to some depth, and what it finds is coarser than B\* where the difference is deep.
3. **Q is arbitrary.** The theorems reduce the law to Q and say nothing about Q. All further structure of the corpus is in the law of kinded patterns, and analysing *that* is the rest of the tower.

---

## 4. Not classical, at the level of signs

The families of THEORY 3.3 fail on a law with kinds for reasons that can be proved.

**Proposition 4.1 (literal records are empty) [P].** Let P have a kind with m members, and let c be any string containing a sign of that kind. In a text of length n the expected number of occurrences of c is at most n/m.

*Proof.* The images of c under exchanges within the kind are at least m distinct strings, all with the same expected count, and the counts of all strings of one length sum to at most n. ∎

So when kinds are open (m large against the text), no record keyed on a literal sign of an open kind is ever filled, in any amount of text that does not exhaust the kind. By Proposition 2.3 such keys are not universals. The n-gram, and PPM as far as it learns across texts, are built from exactly these records. This is the regime of the open vocabulary, and it is where those methods are known to lose.

**Proposition 4.2 (kinds without sameness) [P].** Let a predictor depend on the past only through the sequence of kinds, as a class n-gram or an automaton over classes does. On a law with kinds its expected cost is at least that of a predictor that is uniform within each kind, and such a predictor pays log m, beyond the cost of the kind itself, at every sign of a kind with m members, repeated or not. The pointer form pays, beyond the cost of the kind, the logarithm of the names still free for a new name, and −log of the probability of that repeat given its kind for a repeated one. Where the pattern determines the repeat, the difference is log m per repeated name.

*Proof.* On an invariant law the expected cost of a predictor is unchanged when it is conjugated by an exchange within a kind, and cost is convex in the predictor, so averaging over those exchanges does not raise the expected cost. A predictor that sees only kinds is, after averaging, uniform within each kind. ∎

**Proposition 4.3 (sameness without kinds).** A model whose only criterion is the longest literal match can repeat a name only where the literal context of the repeat has occurred before in the text. Where the repeat is selected by kind and the literal context is new, as when a key reappears in another spelling, it has nothing to match. This is THEORY 9.2's separation; it is quoted here, not re-proved.

**Together.** On a law with open kinds and repeats selected by kind, a learner that identifies B\* and the pattern law is not an n-gram, not a class model, not an automaton over classes and not a longest-match model, and each of those pays a cost per text that does not go away with more text. That is condition (A), as a theorem, for signs.

It is a theorem about what is identified. It does not say that a particular procedure identifies it within a particular amount of text. For the kinds of a bounded-memory source that follows from §3.4 point 2 and a count of exchange tests, |Σ|² of them, each one pass. For the pattern law it is the rest of the tower and is not done.

---

## 5. The same form as a transformer

### 5.1 The predictor the symmetry forces is an attention layer

Read Theorem 3.3 as an architecture. The input code of a sign is its kind together with its name. The predictor needs, at each position: a weight for each earlier position (the q_j, shared among the positions of a block); the name at the position chosen, copied to the output; a weight for each kind (the ν_i); and an output code tied to the input code, since "the same sign" is equality of codes. That is one attention layer in which keys and queries are computed from kinds and pattern, values are names, and the read-out is tied: the layer of THEORY's Proposition 5.2 with a link in front of it.

**Corollary 5.1 [P].** For a law with kinds, the Bayes-optimal next-sign predictor is equivariant under exchanges within kinds, and every equivariant predictor has the pointer form. In particular any sequence model that treats two signs alike until its data distinguish them computes, on those signs, a pointer over earlier positions selected by kind.

*Proof.* The optimal predictor is the conditional law, which is equivariant because the law is invariant; equivariance is all the proof of Theorem 3.3 used. ∎

**[Reading]** The copying and induction circuits found in trained transformers **[K]** are this form. The theory does not need to imitate them. It says they are what any optimal predictor of a world with kinds must contain, and it says what their keys are: kinds.

### 5.2 What a text can define

THEORY 8.5, recalled because it is the other half of the comparison **[K]**. A text over finitely many signs is a finite linear order with unary predicates. Its first-order definable properties of a position are those of linear temporal logic with "since" (Kamp), and those are exactly what masked hard-attention transformers compute (Yang, Chiang and Angluin). Each use of "since" is one recency link; each Boolean combination is one conjunction. Counting quantifiers correspond to averaging attention. So the tower of THEORY 5.4, restricted to finite value sets, and the hard-attention transformer define the same criteria. And linear orders with unary predicates are tame in the strongest useful sense: they do not have the independence property (they are dp-minimal, Simon), so each fixed criterion has bounded VC dimension uniformly over all texts. THEORY 8.7 hoped that the corpus world is tame once orders are adjoined. For the text structure itself that is a theorem. What can be wild is the law, not the language in which criteria are written.

### 5.3 What this comparison is and is not

It is a comparison of **representations**: what must be represented (5.1) and what can be defined (5.2). On both counts the logical picture and the transformer coincide, and where they differ it is in the logical picture's favour only in that its kinds are identified exactly (3.1) where a vector code is identified up to a continuous group (THEORY 8.2).

It is not a comparison of **learning**. A transformer finds its criteria by gradient descent. The theory has, so far, exchange tests for kinds and nothing proved for the pattern law. The test of whether a future learner is learning is therefore not a score. It is whether the partition it reports is B\*, checked by exchanging signs within its classes and seeing the law unchanged, and whether its links are keyed on kinds. Those are properties of the representation and can be read off it.

---

## 6. The open problem: composite objects

Everything above is about the signs as given. From raw bytes that is not enough, and this is the place the earlier build cheated.

### 6.1 What lifts for free

**[Def] Unit system.** A set C of non-empty strings such that every text of the corpus is a concatenation of members of C in exactly one way (a code), together with a partition of C into kinds, such that the law is unchanged by exchanging two units of one kind wherever they occur as units.

**Proposition 6.1 [P].** For a unit system, Theorems 3.1 to 3.3 and Propositions 4.1 and 4.2 hold with "sign" replaced by "unit".

*Proof.* Unique factorisation makes the texts the free monoid on C, and the proofs used nothing about signs but that texts are sequences of them. ∎

So if the units were known, the perfect language would be: the kinded pattern of units; pointers to earlier units selected by kind; and new units, whose only content is their spelling. The spelling of a new unit is a short text over the signs with its own kinds and patterns, one level down. That is the shape of the tower from bytes, and it has two levels of the same thing, not a special module.

### 6.2 What is not known

1. **Are the units of a law canonical?** For signs, Theorem 3.1 gives one partition. For units there are two choices at once, the code and its kinds, and I have no theorem that a law has a distinguished unit system: the finest one, or the one whose symmetry accounts for the most. Without that, any procedure that returns units is returning a convention, which is what a menu of boundary classes was.
2. **How are they found?** The natural candidate is substitution: two strings are of one kind when replacing one by the other, everywhere, changes the probability of every text by a constant factor. That is a congruence on strings, the weighted analogue of the syntactic congruence of a language, and it is defined by the law alone. Its relation to unit systems (when its classes come from a code, whether the code is unique) is what has to be worked out. The known results on learning from substitutability are for a class of grammars **[K]** and by THEORY's own standard a tree-shaped grammar is a classical endpoint, so this cannot simply be taken over.
3. **Exactness**, again (3.4, point 1), and now for strings.
4. **The pattern law.** Q is unconstrained. Its analysis is where conjunction, path and the selection results of WITNESSES would re-enter, with density (2.3) in place of the rank budget and identity defined by exchange (3.1) in place of ε-goodness. None of that is written.

### 6.3 The gate

No experiment on text follows from this file. The conditions under which one would:

1. a theorem that the units of a law are determined by the law, or a proof that they are not and a principled statement of what is determined instead;
2. a procedure for them that uses only order, sameness and counts, with a stated cost;
3. the corresponding statement for the pattern law, at least for one non-trivial class of laws.

When those exist, the test is the one in §5.3: read the representation. Exchange what the learner says is of one kind and see whether the law is unchanged.

---

## 7. Sources

**Proved here.** Proposition 2.3; Lemma 2.2; Theorems 3.1, 3.2 and 3.3; the remark on bounded memory in 3.4; Propositions 4.1 and 4.2; Corollary 5.1; Proposition 6.1. Checks F1 to F4 compare Theorems 3.1 to 3.3 and Lemma 2.2 with brute force on laws over five signs, and pass.

**Cited from memory [K].** Shannon; the convergence of conditional entropies to the entropy rate. De Finetti. Ryll-Nardzewski. Kamp's theorem; McNaughton and Papert; Yang, Chiang and Angluin 2024 on masked hard-attention transformers; Yang and Chiang 2024 on counting. Simon 2011 (coloured linear orders are dp-minimal). Induction and copying circuits in transformers (Olsson et al. 2022). Learning from substitutability (Clark and Eyraud 2007).

**Not carried over.** The learner, its experiments, the baselines and all scores have been removed from this folder. They remain in the history of the branch, up to commit `6cb547a`.
