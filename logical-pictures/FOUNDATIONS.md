# Foundations, again

*A restart. The learner that was in this folder has been removed: it supplied its own objects by hand and then reported scores. This file goes back to the two source notes, THEORY ("Logical pictures") and WITNESSES ("Witness selection"), says what they are, says exactly where the theory was too immature to build on, proves what the objects of a law are, and only then builds the learner those theorems license.* Marks: **[Def]**, **[P]** proved here, **[P✓]** proved and checked by brute force on small cases ([checks/foundations_checks.py](checks/foundations_checks.py)), **[K]** known, cited from memory, **[C]** conjecture, **[Reading]** interpretation. The only measurements are in §9, and they are read as representations, not as scores.

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

| 8 | **Memory and states are canonical.** Each past has a least set of names its future depends on; pasts whose futures agree up to renaming form the states of one machine, determined by the law, whose memories have a size and a symmetry (§6) | [P✓] |
| 9 | **Units are canonical in the limit.** Where units recombine freely, they are the basis of the smallest free submonoid containing the texts (§7) | [P, given K] |
| 10 | A learner that uses only order, sameness and counts returns that machine, exactly with exact statistics, and does so on three worlds where every literal record is empty (§8, §9) | [P]; observed |

**What is still not established.** From raw bytes the names are spelled. Statement 9 is about the support of the law and has no finite-sample form yet, real names are not equally likely, and spellings have their own law (§7). So the experiment of §9 is on texts whose signs are already the units, and nothing here licenses one on bytes. §10 lists what is next.

Against the two conditions set for this work:

- **(A) Does not converge to a classical method.** Proved for laws with names (§4, §6.3). What the learner returns, states with memories and their symmetries, is not an object of any classical family.
- **(B) Comparable with transformers.** The form the theory forces is the form transformers have, and the canonical machine is an attention network whose links are fixed by its state (§5, §6.3). That is a statement about representation. Nothing is claimed about learning dynamics.

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

## 6. Memory: the canonical form of a law with names

§3 reduced a law to its law Q on kinded patterns and stopped: "Q is arbitrary". This section says what the objects of Q are. The setting is that of *data words*, and the mathematics is that of nominal sets (Gabbay and Pitts; Bojańczyk, Klin and Lasota) **[K]**; the statements are proved here in the form needed.

**Setting.** The signs are finitely many *closed* signs, each its own kind, and names, in one or more open kinds. Renamings are the permutations that preserve the kinds, and the law is invariant under them. For a past h with positive probability, μ_h is the law of what follows h. Assume each open kind has at least three more names than any text uses.

### 6.1 What a past remembers

**[Def] Support, memory.** A set S of names *supports* μ_h when every renaming that fixes each member of S leaves μ_h unchanged. The names occurring in h always do.

**Theorem 6.1 (least memory) [P✓, check F5].** If S and S′ support μ_h, so does S ∩ S′. So there is a least supporting set, mem(h): the names the future still depends on.

*Proof.* Renamings fixing S ∩ S′ are generated by exchanges (a b) with a, b outside S ∩ S′. If both are outside S, or both outside S′, the exchange preserves μ_h. Otherwise say a ∈ S ∖ S′ and b ∉ S. Take a name c of the same kind outside S, S′, a and b. Then (a b) = (a c)(c b)(a c); (a c) moves nothing in S′ and (c b) moves nothing in S, so each factor preserves μ_h. ∎

**Theorem 6.2 (the pointer form, sharpened) [P✓, check F5].**

1. Given h, all names outside mem(h) are equally likely to come next, whether they occurred in h or not. A forgotten name is exactly as good as a new one.
2. mem(h·a) ⊆ mem(h) ∪ {a}. Memory changes only by taking in the sign just read and by letting names go.

*Proof.* (1) An exchange of two names outside mem(h) preserves μ_h. (2) A renaming that fixes mem(h) and a preserves μ_h and the event "a comes next", hence μ_{h·a}. ∎

So the pointers of Theorem 3.3 do not range over all earlier names. They range over a memory that the law determines, and the law also determines when a name leaves it. The particulars of a picture are these: the members of mem(h). Nothing had to be chosen to say what they are.

### 6.2 What a past is: states up to renaming

**[Def] State.** Two pasts are *in the same state* when some renaming carries the future law of one to that of the other. A state has a memory size, and a group of rearrangements of its memory that leave its law unchanged: its *symmetry*. A memory whose symmetry is everything is a set; one whose symmetry is trivial is a tuple.

**Theorem 6.3 (the canonical machine) [P✓, check F5].**

1. If h and h′ are in the same state by a renaming π, then h·a and h′·π(a) are. If a and b are names of one kind outside mem(h), then h·a and h·b are in the same state.
2. So the states form a deterministic machine. From a state, the sign read is a closed sign, the name in a given place of the memory, or "a name outside the memory" of a given kind; each leads to one state, and says which names are kept. The law is given by the probabilities of these readings at each state.
3. This machine is determined by the law. Any equivariant deterministic machine with a memory of names that generates the law maps onto it, state by state, and its memory contains mem(h) at every past.

*Proof.* (1) μ_{h′·π(a)} is π applied to μ_{h·a}, by the definition of conditioning; for the second claim take π = (a b), which preserves μ_h. (2) is (1) with Theorem 6.2. (3) A machine's state and memory at h determine μ_h equivariantly, so pasts it identifies are in the same state, and its memory supports μ_h, so it contains the least support. ∎

**[Reading]**

- This is the Myhill–Nerode theorem of the theory of nominal sets, for laws instead of languages. In the terms of THEORY's table of classical families, it is the row "automata and causal states" with the entry in its last column, *missing: particulars*, filled in. A causal state is a class of pasts with the same future. A state here is a class of pasts with the same future *up to who is who*, together with the list of who still matters.
- It answers the question the notes left open, "what is an object before a criterion for it has been written": the universals of a law are its states, the particulars are the members of their memories, and both are defined by the law alone.
- The symmetry of a memory is part of the logical form. "The keys that have been bound and not yet asked for" is a set; a stack of open brackets is a tuple. The removed build could represent neither as such.
- A law has a *finite form* when it has finitely many states. That is the condition under which everything below works, and it is the analogue of orbit-finiteness, which in model-theoretic terms is definability over a structure whose automorphism group has finitely many orbits on tuples of each length **[K]**.

### 6.3 Not a classical method

A machine over states alone, with no memory, is a hidden Markov model or an automaton; by Proposition 4.2 it pays log m at every remembered name. A longest-match model has a memory but it is the whole literal past, with no states and no forgetting, so it cannot say which earlier name will be referred to when that is decided by the state and not by the literal context. A machine of states with memories is neither, and the difference is not a matter of degree: Theorem 6.3 says the memory and its symmetry are part of what the law *is*.

Against the attention layer of §5.1: a place in the memory is a link to the position that put the name there. So the canonical machine is a network of the kind §5 described, in which what is attended to is fixed by the state. What the theorem adds is that this network is unique, and that it has a size: the number of states and the size of their memories.

---

## 7. Units

On raw bytes the names are spelled, so the canonical machine is needed over units and not over bytes. §6 of the previous version of this file asked whether the units of a law are determined by it. In the limit of text they are.

**[Def]** A set of strings is a *code* when every string has at most one factorisation into its members; the strings that have one form a *free* submonoid, and the code is its unique basis.

**Theorem 7.1 (units are canonical under recombination) [P, given K].** Let X be the support of the law: the texts with positive probability. The intersection of all free submonoids containing X is free. Its basis U is therefore the coarsest code in which every text factors: every other such code cuts the members of U into pieces. If the law is generated from a code C by a law on sequences of units under which every sequence of units is possible, then U = C.

*Proof.* The intersection of free submonoids of a free monoid is free (Tilson) **[K]**; a free submonoid is finer than another exactly when it contains it. If every sequence over C is possible then X generates the submonoid of all products of members of C, which is free with basis C, and it is contained in every free submonoid containing X. ∎

So clause P3 of the picture theory, that whatever the forms permit is possible, is what makes units canonical: where units recombine freely, the coarsest code that every text respects is the units. A word-splitter is not needed and a menu of boundaries is not needed. The free submonoid is also what a criterion needs: by Schützenberger's characterisation **[K]** it is the closure of the texts under one rule, *if u, uw, wv and v are admissible then so is w*, which is a statement about order and sameness only.

**What this does not give.** The theorem is about the support of the law. A finite corpus is a finite set of texts, usually itself a code, and its own coarsest code is the texts themselves. A procedure for a finite corpus has to estimate which recombinations are possible from those that occurred, and I have no theorem for that. Two further gaps: real names are not equally likely, so exchange holds only up to a weight per name (§3.4); and the spelling of a new name is itself a text one level down, with its own law. Neither is treated.

**Consequence for experiments.** The theory is complete where the units are the signs: texts given as sequences of names and closed signs. It is not complete for bytes. An experiment on bytes would again need units from somewhere, and until there is a finite-sample form of Theorem 7.1 that somewhere would be me.

---

## 8. The learner the theorems license

For texts whose signs are the units, Theorems 6.1 to 6.3 say what is to be found, and each part of it is decided by a count. Code: [nominal/learn.py](nominal/learn.py).

**Supplied.** Order, sameness of sign, counts. Three numbers: the share of texts a sign must recur in to count as a universal, how many signs ahead two futures are compared, and the level of the test.

**The rule.**

1. *Universals and names* (Proposition 2.3). A sign that recurs in a fixed share of the texts is read as itself. Every other sign is a name, and all that is read of it is where it recurs (Theorem 3.2).
2. *Forgetting* (Theorem 6.1). After a transition, a remembered name is dropped if it is never referred to again. Under the pattern law the only thing a text can show about a name is where it recurs, so this is exactly the test of whether it is in the least support.
3. *Identity* (Theorem 6.3). The new past is the same state as an existing one if some matching of their memories makes the laws of their next D signs agree, by a test of homogeneity. Otherwise it is a new state.
4. Repeat, in order of how often an undefined transition is met, until none is met often.

**Proposition 8.1 [P].** With exact statistics, on a law with N states, the rule with D ≥ N − 1 returns the canonical machine.

*Proof.* Every past it examines is assigned its least memory by step 2 and Theorem 6.1. Two states of a deterministic machine with N states whose laws agree on the next N − 1 signs agree altogether (the classical refinement argument), so step 3 identifies two pasts exactly when they are in the same state. The transitions are then those of Theorem 6.3. ∎

With counts in place of exact statistics the rule errs with the probability of its tests. I have not worked out the rate.

**What this is, plainly.** It is state merging, the method of the automata row, carried out up to renaming and with a memory. What the classical method lacks is exactly what the lift supplies. Learning automata over names is known as a problem with queries to a teacher **[K]**; here it is done from text, with forgetting decided by counts, and the object returned is fixed in advance by a theorem.

---

## 9. What it learns

Three worlds, each generated by a machine with a memory of names ([nominal/worlds.py](nominal/worlds.py)). Names are drawn from a pool of a thousand million, so that in 200,000 signs about 80,000 distinct signs occur and almost none occurs in two texts: the regime of Proposition 4.1, where no literal record is ever filled. The learner sees the texts and nothing else.

The test is not a score. It is whether the representation is the generating one ([nominal/run.py](nominal/run.py), output in [nominal/OUTPUT.txt](nominal/OUTPUT.txt)): on unseen texts, is each learned state one state of the generator and conversely, is the learned memory at every position exactly the set of names the generator's future depends on, and does the memory have the right symmetry.

| World | Generator | Learned | States correspond one to one | Memory equals the generator's | Symmetry found |
|---|---|---|---|---|---|
| **Bind and recall.** Two keys are bound to values, then asked for in either order | 10 states, memory up to 4 | 10 states, memories 0,1,2,3,4,4,3,2,1,0 | yes | 40,000 of 40,000 positions | exchanging the two pairs, in the two states where both are pending: the memory is a set of pairs |
| **Stack.** Names are opened and closed in reverse order, to depth 3 | 4 states, memory = depth | 4 states, memories 0,1,2,3 | yes | 42,660 of 42,660 | none: the memory is a tuple |
| **Overwrite.** A name is set, a later one replaces it, a closed sign asks for the current one | 3 states, memory up to 1 | 3 states, memories 0,1,1 | yes | 26,639 of 26,639 | – |

Reading the learned machines in the output: in bind and recall, the state after the mark reads "the name in place 0" or "the name in place 2", each half the time, and both lead to the same state with three names kept; that is "either key, and then its value, and the key is no longer needed". In overwrite, reading a new name while holding one leads back to the same state with one name kept: the old one has been dropped, because it is never referred to again. The universals were found by recurrence: the mark, the question and the end of text, and nothing else.

**What this shows and what it does not.** It shows that the objects of §6, states, memories and their symmetries, are learnable from text by counts, exactly, in worlds where every literal method is empty by construction. It is one small experiment on worlds with finite form, small memories and no noise beyond the randomness of the law. It does not show anything about natural text, where names are spelled, recur with unequal weights, and the form is not known to be finite.

---

## 10. Where this stands now

| Condition | Status |
|---|---|
| The objects of a law are defined by the law | Kinds (Theorem 3.1), memory (6.1) and states (6.3): yes. Units: yes in the limit of text, under recombination (7.1) |
| (A) Not a classical method | Proved for laws with names (§4, §6.3). The learner is state merging up to renaming, and what it returns is not an object of any classical family |
| (B) Comparable with transformers | The canonical machine is an attention network whose links are fixed by its state (§5, §6.3). A statement about representation |
| Learnable from order, sameness and counts | For texts over names with a finite form: yes, with exact statistics (8.1), and observed (§9) |
| From raw bytes | No. Needs a finite-sample form of Theorem 7.1, weighted names, and the law of spellings |

**Next, in order.**

1. A finite-sample theorem for units: which recombinations must a corpus contain for the closure of §7 to return the units, and how is "possible" estimated from "occurred".
2. Names with weights: the exchange relation up to a factor per name, and the form of Theorems 3.2 and 6.2 under it.
3. Noise and scale for the learner of §8: the rate of its errors, and laws whose machines are large.
4. Kinds of closed signs and of states: abstraction above the machine. Two states with the same law up to a renaming of *closed* signs are of one kind in the sense of §3, and that is where a tower would start.

---

## 11. Sources

**Proved here.** Proposition 2.3; Lemma 2.2; Theorems 3.1, 3.2 and 3.3; Propositions 4.1 and 4.2; Corollary 5.1; Theorems 6.1, 6.2 and 6.3; Proposition 8.1. Theorem 7.1 is a consequence of a known theorem. Checks F1 to F5 compare Theorems 3.1 to 3.3, Lemma 2.2 and Theorems 6.1 to 6.3 with brute force on small laws, and pass.

**Cited from memory [K].** Shannon; the convergence of conditional entropies to the entropy rate. De Finetti. Ryll-Nardzewski. Kamp's theorem; McNaughton and Papert; Yang, Chiang and Angluin 2024 on masked hard-attention transformers; Yang and Chiang 2024 on counting. Simon 2011 (coloured linear orders are dp-minimal). Induction and copying circuits in transformers (Olsson et al. 2022). Gabbay and Pitts (nominal sets, least supports); Kaminski and Francez (automata with a finite memory of names); Bojańczyk, Klin and Lasota 2014 (automata theory in nominal sets, the Myhill–Nerode theorem there); Moerman et al. 2017 (learning nominal automata with queries). Tilson 1972 (the intersection of free submonoids is free); Schützenberger (free submonoids are the stable ones); Berstel, Perrin and Reutenauer, *Codes and Automata*.

**Measured.** §9 only: three synthetic worlds, read by whether the generating machine was recovered.

**Not carried over.** The earlier learner, its experiments, the baselines and all scores have been removed from this folder. They remain in the history of the branch.
