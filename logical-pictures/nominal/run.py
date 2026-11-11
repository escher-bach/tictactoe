"""Learn the machine of each world and read what was learned.

There is no score here.  The test is whether the representation is the right one:
  - does each learned state correspond to exactly one state of the generating machine,
    and each generating state to one learned state;
  - is the learned memory, at every position of unseen texts, exactly the set of names
    the generating machine's future depends on;
  - does the learned memory have the right symmetry (a set of pairs, a stack).
"""
import sys
from collections import Counter, defaultdict

sys.path.insert(0, __file__.rsplit("/", 2)[0])
from nominal.learn import learn, symmetries  # noqa: E402
from nominal.worlds import END, WORLDS  # noqa: E402

for name, gen in WORLDS.items():
    train = gen(20000, 1)
    test = gen(4000, 2)
    texts = [t for t, _ in train]
    print(f"=== {name}: {len(texts)} texts, {sum(map(len, texts))} signs, "
          f"{len({x for t in texts for x in t})} distinct signs")
    M, stats = learn(texts)
    print(f"universals found: {sorted(M.closed)}   (END = {END})")
    print(f"learned: {len(M.arity)} states, memory sizes {M.arity}")
    for s in range(len(M.arity)):
        tot = sum(stats[s].values())
        nxt = Counter()
        for pat, k in stats[s].items():
            nxt[pat[0] if pat[0][0] != "n" else ("n",)] += k
        row = ", ".join(f"{'end' if sym == ('c', END) else sym[0] + str(sym[1]) if len(sym) > 1 else 'new'}"
                        f" {k / tot:.2f}->{M.trans[s].get(sym, ('?',))[0]}" for sym, k in sorted(nxt.items(), key=lambda z: -z[1]))
        sym = symmetries(M, texts, s, D=16) if M.arity[s] > 1 else []   # to the end of the text
        print(f"  state {s} [memory {M.arity[s]}{', unchanged by ' + str(sym) if sym else ''}]: {row}")
    # compare with the generating machine on unseen texts
    to_true, to_learned = defaultdict(Counter), defaultdict(Counter)
    mem_ok = n = undefined = 0
    for text, truth in test:
        path, stop = M.run(text)
        undefined += stop is not None
        for (s, mem), (label, live) in zip(path, truth):
            to_true[s][label] += 1
            to_learned[label][s] += 1
            mem_ok += set(mem) == live
            n += 1
    f1 = all(len(c) == 1 for c in to_true.values())
    f2 = all(len(c) == 1 for c in to_learned.values())
    print(f"unseen texts: {n} positions; learned state -> generating state is a function: {f1}; "
          f"its inverse is a function: {f2}; states of the generator: {len(to_learned)}")
    print(f"memory equals the generator's live names at {mem_ok}/{n} positions; "
          f"texts on which the machine was undefined: {undefined}/{len(test)}")
    for s, c in sorted(to_true.items()):
        print(f"    state {s} = {dict(c)}")
