# What a text is a picture of

*Why the theory so far reduces to context models with caches, and the formalisation that does not. Theory only: definitions, the propositions that can be proved now, and what must be proved before a learner. Marks: **[Def]**, **[P]** proved here, **[K]** known, **[Reading]** interpretation, **[Open]**.*

## 1. The error

In the Tractatus a proposition is a picture *of a situation*. There are two facts: the sign-fact and the fact depicted. The elements of the first stand for objects of the second, and that the elements are arranged as they are says that the objects are so arranged (2.15, 3.1432). In the *Aufbau* the basis is elementary experiences and one relation among them, and what is constructed from it is a world: qualities, places, things. In both, **what is represented is not the representation.**

THEORY says this in LP1 and LP2: a world 𝕎, and a picture "with a map into 𝕎". Then it defines everything on the text alone. A text is the fact (Pos, <, ≈) (THEORY 5.1); a sort is a quotient of the positions by a criterion on what precedes them (5.2); a fact is the value of the next sign (5.5). WITNESSES and everything I built inherited that. The world is the string and the objects are classes of its positions.

**Proposition 1 (the self-picture) [P].** Every law on strings has a picture in the sense of THEORY 5: the sort whose objects are the classes of positions with the same k preceding signs, with the next sign as its determinable. Its residue tends to the entropy rate as k grows.

*Proof.* The criterion is decided before the position; the residue is the k-th conditional entropy. ∎

So a string is always a picture of itself, and that picture is the context model. The only relations a string bears to itself are its repetitions, which is why recency gave caches and nothing else. The builds did not collapse by accident. They found the picture the formalisation asked for. Nothing in the definitions says the picture must be of anything *else*, and that is what is fundamentally wrong.

Two symptoms are now explained.

- **No kinds** (READINESS 3.1). Kinds are sorts of objects *of the world*. A class of positions under value tests cannot be one.
- **No objects of variable extent** (READINESS 3.2). A name is as long as it is. Its extent is not a fact about positions; it is a fact about which stretch of signs stands for one object.

## 2. Text, world, projection

**[Def] World.** A many-sorted relational structure: objects, and for each relation symbol R with a signature of sorts (s₁, …, s_k), the tuples for which it holds. A tuple in a relation is a *state of affairs* (clause P2).

**[Def] Form of depiction.** For each R, a *template*: a string of closed signs with k slots, one per constituent. Closed signs name nothing. They are how the arrangement is shown (4.0312).

**[Def] Text.** A sequence of states of affairs of one world, each written by its template with the names of its constituents in the slots. Names are arbitrary: a text's objects are named by an injection chosen uniformly (THEORY 4.1).

**[Def] Picture law.** A law on texts given by: a law on worlds that is exchangeable in the objects of each sort; a rule for which state of affairs is stated next, depending only on the world and on what has been stated, and not on names; a form of depiction.

The picture of a text, at any point, is the structure of what has been stated so far: its objects and the states of affairs among them. It is a finite relational structure with a map into the world, which is LP2. It is not a state and not a class of positions. Probability is what THEORY 4.4 said: the share of worlds containing the picture so far in which the continuation is also true, weighted by the rule of statement.

What the earlier theorems become in this setting:

| Earlier result | Here |
|---|---|
| Kinds as exchange classes (FOUNDATIONS 3.1) | The sorts of the world. Names are exchangeable exactly within a sort |
| Pointer form (3.3) | A slot is filled by an object already in the picture or by a new one |
| Least memory (6.1) | The objects about which something can still be stated |
| Canonical state (6.3) | The isomorphism type of the picture restricted to those objects. This is why states multiply (10.2): they enumerate pictures |
| Aldous–Hoover (THEORY 4.2) | Applies to the world, where objects are exchangeable, and not to positions |

## 3. The two gaps, as definitions

### 3.1 Kinds: the form of an object is the slots it can fill

"If I know an object I also know all its possible occurrences in states of affairs" (2.0123). A slot is a pair (template, place).

**[Def]** Two slots are *akin* when some object of some text fills both. A *sort* is a class of the equivalence this generates.

**Proposition 2 [P].** In a picture law in which every object of a sort can fill every slot of that sort with positive probability, the sorts of the definition are the sorts of the world, and two names are exchangeable under the law iff the slots they fill are of one sort.

*Proof.* A slot's fillers all have its world sort, so akin slots have one sort; by the hypothesis two slots of one sort share a filler with positive probability. Exchange within a sort preserves the law by exchangeability of the world, and across sorts it produces texts of probability zero. ∎

This is the rule for tabulation that was missing, and it is Carnap's quasi-analysis applied where it belongs: slots are the elements, "shares a filler" is the similarity, and a sort is a similarity circle. It classes *slots*, not values, so Proposition 1 of READINESS does not apply: a sort has as many frequent members as it likes. It needs no margin where a name fills a slot or does not. And it is decided by sameness of sign alone.

### 3.2 Units: a statement, and a name

There are two kinds of unit and the earlier work confused them.

*A statement* is one instance of a template. Replace every name in a text by a slot mark; call the result its *skeleton*. Skeletons are strings over the closed signs and one mark.

**Proposition 3 [P, given K].** If the rule of statement allows every sequence of templates, the templates are the basis of the smallest free submonoid containing the skeletons of the texts.

*Proof.* FOUNDATIONS Theorem 7.1 (Tilson), applied to skeletons. ∎

The same theorem was useless on raw text because a finite corpus of texts is its own coarsest code. Skeletons are different in kind: they are over a small closed alphabet, the same ones recur in every text, and the closure in question, *if u, uw, wv and v occur then w is a unit*, is over strings that are seen thousands of times. That is where a finite-sample statement can be hoped for. **[Open]**

*A name* is what stands in a slot. With atomic names there is nothing to find: by FOUNDATIONS Proposition 2.3 a closed sign recurs across texts and a name does not. With spelled names, a name is a stretch that varies where its surroundings do not: the slot is defined by the template around it. So the extent of a name is fixed by the statement it occurs in, and not by any property of its own signs. **[Open]** for the case where names recur with unequal weights, which is the case of real words.

### 3.3 Why this is not the self-picture

The self-picture has one sort (positions), one relation (succession) and no independence: its facts determine one another. Clauses P3 and P4 are what exclude it. An analysis is a form of depiction under which the world's states of affairs on disjoint sets of objects are independent and recombine freely. Lemma A's empty solution (independent bits that are about nothing) is excluded by the other half: each state of affairs must be written by one contiguous template with a slot for each constituent, so the picture has "exactly as many distinguishable parts" as what it depicts (4.04).

**The identification problem [Open].** Given a law on texts, is the form of depiction (templates, sorts) under which it is a picture law with independent states of affairs unique, up to renaming relations and sorts? Propositions 2 and 3 give uniqueness of sorts and templates under free recombination. What is not proved is uniqueness of the *world's* law, and that the independent form is the only one: that no coarser or finer system of templates also yields independence.

## 4. On paper, against the four requirements

| Requirement (THEORY 3.3) | A learner of picture laws |
|---|---|
| Learned criteria of identity | Three, of different types: when two stretches are the same template (a relation); when two slots are of one sort; when two names are the same object |
| Both scopes | The form of depiction and the law of worlds are universal. A text's objects and its states of affairs are particular |
| Criteria over constructed objects | Sorts are defined over slots, slots over templates, the law over sorts |
| Several objects per position | A statement names a relation and each of its constituents; the picture holds all that has been stated |

The representation is a relational structure. It is factored by construction: its size is the number of states of affairs stated, not the number of their combinations. Its nearest relatives are relational models (stochastic block models and their infinite versions) and semantic-parser induction, not the n-gram, PPM, automaton or tree-grammar rows. Unlike a tree grammar it is re-entrant: one object stands in many states of affairs. Unlike inductive logic programming it enumerates no clauses: templates come from the closure of Proposition 3 and sorts from shared fillers.

It passes on paper. That is a weaker thing than passing, and the first build also looked right before it was run.

## 5. What must be proved before a learner

1. **Templates from finitely many skeletons** (3.2): how much recombination a corpus must contain for the closure to return the templates, and a test for "occurs" by counts.
2. **Independence** (3.3): that under a picture law the future factorises over the connected parts of the picture, so that the law is learned per kind of state of affairs. This is the decomposition FOUNDATIONS 10.3 asked for; here it should follow from exchangeability of the world, with the rule of statement as the only obstacle.
3. **Identification** (3.3).
4. **Identity of objects beyond sameness of name.** In text one object has many names. The criterion has to be THEORY's M1: two names are one object when identifying them makes stated facts repeat. Not treated.
5. **Names with weights and spelled names** (3.2).

The order is 2, 1, 3. Item 2 decides whether the representation is really factored; if it fails, this is a state machine again and should be dropped.

## 6. What this does not claim

It does not claim that natural text is a picture law. Natural text does not come with one template per state of affairs, and most of what it states is not elementary. The claim is narrower: the picture theory is a theory of this kind of object, the earlier formalisation was not, and that difference is why the earlier formalisation gave context models.
