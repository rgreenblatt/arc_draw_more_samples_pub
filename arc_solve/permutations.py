import itertools
import numpy as np

def permutation_list(n: int):
    assert n > 1
    gen = np.random.default_rng(n)
    if n <= 7:
        out = list(itertools.permutations(range(n)))
        out.remove(tuple(range(n)))
        gen.shuffle(out)
        return out

    out = [
        tuple(int(y) for y in x)
        for x in np.array([gen.permutation(n) for _ in range(3000)]).tolist()
    ]
    try:
        out.remove(tuple(range(n)))
    except ValueError:
        ...
    return out

all_permutation_indices = {n: permutation_list(n) for n in range(2, 11)}
