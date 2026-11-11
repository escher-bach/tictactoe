"""Small worlds in which the non-classical objects of THEORY 9 are the whole story.

respelled  THEORY 9.2: a particular identified through a universal.  Each episode binds
           keys to values ("k v") and later asks for them with the key spelled differently
           ("K ?").  The literal context of a question never preceded its answer.
analogy    THEORY 4.4: structure without names.  Each episode is "a b b a ;" for a fresh
           pair of signs.  The rule mentions no sign, only which positions carry the same one.
"""
import numpy as np


def respelled(n_episodes, seed, n_keys=12, n_vals=8, per_episode=4):
    """Signs: lower keys 0..n_keys-1, upper keys n_keys..2n_keys-1, values, then ';'.
    Returns the text and a mask of the answer positions (ideal cost: 0 bits each)."""
    rng = np.random.default_rng(seed)
    v0 = 2 * n_keys
    sep = v0 + n_vals
    out, ans = [], []
    for _ in range(n_episodes):
        keys = rng.choice(n_keys, per_episode, replace=False)
        vals = rng.integers(0, n_vals, per_episode)
        for k, v in zip(keys, vals):
            out += [k, v0 + v]
            ans += [0, 0]
        for i in rng.permutation(per_episode):
            out += [n_keys + keys[i], v0 + vals[i]]
            ans += [0, 1]
        out.append(sep)
        ans.append(0)
    return np.array(out, np.int64), np.array(ans, bool), sep + 1


def analogy(n_episodes, seed, signs):
    """Episodes "a b b a ;" with a != b drawn from `signs`.  The separator is sign 0.
    Mask marks the two determined positions (second b, final a)."""
    rng = np.random.default_rng(seed)
    out, det = [], []
    signs = np.asarray(signs)
    for _ in range(n_episodes):
        a, b = rng.choice(signs, 2, replace=False)
        out += [a, b, b, a, 0]
        det += [0, 0, 1, 1, 0]
    return np.array(out, np.int64), np.array(det, bool)
