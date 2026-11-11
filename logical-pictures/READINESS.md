# Is the theory ready for a learner? Not yet

*Written before building anything. The rule of WITNESSES §10 (later version, with Theorems 1″, 3′, 7, 8 and §8.3) is instantiated on raw text on paper, and the result is tested against THEORY's four requirements for not being classical. It fails, for two reasons that are gaps in the theory and not in an implementation. Marks: **[P]** proved here, **[Reading]** interpretation.*

## 1. The rule of §10, written out on bytes

What the notes supply: positions, their order, sameness of sign; the operations path, conjunction and recency (THEORY 5.4); any-split for the classes of a sort, peeling for dependence, one partition per sort, a rank budget (WITNESSES §10).

1. **Level 0.** The only sort is Pos, the only determinable is the sign, the only link is adjacency. The tests on a position are "the sign at link j is a".
2. **Coordinates.** One partition per link (§8.3). Any-split over the value tests of one link gives: one class per frequent sign, and one class for all the rest.
3. **Dependence.** Peel the next sign at rank r over those tests. This is a tree of rank r over literal offsets: by WITNESSES §12 itself, a decision-tree n-gram.
4. **Keys.** On the J links the peel chose, any-split gives at most (ln n/ε)ᴶ classes of positions (Corollary 8.5): the frequent windows of J signs, and a remainder.
5. **New links.** For each partition, "the most recent earlier position in my class" and "the position after it". With step 4's partition this is: what followed the last time this window occurred.
6. Repeat from 2 with the new links.

## 2. Tested against the four requirements

| Requirement (THEORY 3.3) | The rule on bytes |
|---|---|
| Learned criteria | The links are chosen by peeling. Yes |
| Both scopes | Recency gives particulars. Yes |
| Criteria over constructed objects | The constructed objects are frequent literal windows. Formally yes; in content every criterion is "the same literal window" |
| Several objects per position | A tuple of coordinates. Yes |

It passes the letter of the test and is, in content, a variable-length literal context model with caches keyed on literal windows: the n-gram and PPM rows together. That is what my first build was, and it is what its audit measured. Removing the hashing of conjunctions (the "multiplying out" of §8.3) would not have changed which objects exist.

## 3. Why: two gaps in the theory

### 3.1 The rule cannot form a kind

**Proposition 1 [P].** Let a class of positions be ε-good for the value tests of a determinable A. Then either one value of A holds on more than 1 − ε of the class, or every value of A holds on less than ε of it. No ε-good class has two values of A each on at least ε of its members.

*Proof.* A value held by a share between ε and 1 − ε ε-splits the class. If one value has share above 1 − ε, the others share less than ε. ∎

So over value tests, a class is either pinned to one value or diffuse over rare ones. "The previous sign is a digit" or "the previous word is a determiner", a class with several frequent members, is not an ε-good class and no run of any-split produces it. The generic type of such a set says "equal to none of them" (WITNESSES §7's own example). Abstraction in the ordinary sense is not something the rule fails to find. It is something the rule's notion of object excludes, as long as the tests are value tests.

Kinds enter THEORY through move M3, tabulation: "recode values so that a family of classes becomes one determinable". WITNESSES treats W1 to W3 and gives no rule for M3. **That rule is missing.**

A candidate exists inside the theorems already proved. Apply any-split not to positions but to the *values* of a determinable, with tests "this value is attested next to that sample value". That relation is 0/1 by construction, Theorems 1, 1″ and 7 apply to it as they stand, and its good classes are sets of values with one form in the sense of clause P1. Whether it is stable enough, what "attested" must mean as a count, and how the kinds of one level become the tests of the next, are not worked out. When I tried something like it on word types it gave digits, determiners and modals, but I had supplied the words.

### 3.2 The rule has no objects longer than a fixed window

Every object the three operations produce is a class of positions defined by signs at fixed link-distances. A word, a number, a name are runs of variable length. The notes contain no operation that yields them, and each time I added one (a boundary class chosen from a menu, a spelling test) it was a fact about the corpus supplied by hand. FOUNDATIONS §7 gives units in the limit of text and no finite-sample rule.

### 3.3 And one condition I cannot check on paper

The rule needs relations that are near 0 or 1 at single-digit rank (WITNESSES §8.4). For literal next-sign tests on text they are not: most next signs are not nearly determined by any class. The theory's answer is that they become so "after analysis", that is, once the tests are about kinds. That depends on 3.1.

## 4. Verdict

**I am not ready to build the learner.** Built today from the notes as they stand, it is a literal context model with caches, for the reasons in §3, and I can say so without running it.

What has to exist first, in order:

1. **A rule for M3**, with a theorem: which sets of values are kinds, canonical for those with a margin (the analogue of Theorem 7), and at what cost.
2. **Objects of variable extent**, defined from order and sameness, with a finite-sample statement.
3. Then, and only then, the tower of §10 over tests about kinds, checked on paper against the four requirements before any code.

`nominal/` in this folder is the state-machine learner of FOUNDATIONS §8. It is classical (FOUNDATIONS §10) and is not a step towards the above.
