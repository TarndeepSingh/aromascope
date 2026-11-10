"""Leakage-resistant splitting. Molecules sharing a group (scaffold, or flat structure for
acyclic molecules) always land in the same partition."""
from __future__ import annotations

from collections import defaultdict

import numpy as np


def scaffold_split(groups, fracs=(0.8, 0.1, 0.1), seed: int = 0) -> np.ndarray:
    """Grouped train/val/test split.

    Groups larger than half the validation size always go to train (so a handful of huge
    scaffolds cannot dominate val/test); remaining groups are shuffled with a fixed seed and
    used to fill test, then val. Deterministic for a given seed.
    """
    groups = list(groups)
    n = len(groups)
    by_group: dict = defaultdict(list)
    for i, g in enumerate(groups):
        by_group[g].append(i)
    n_val, n_test = int(round(fracs[1] * n)), int(round(fracs[2] * n))

    big = [s for s in by_group.values() if len(s) > max(1, n_val // 2)]
    small = [s for s in by_group.values() if len(s) <= max(1, n_val // 2)]
    rng = np.random.RandomState(seed)
    rng.shuffle(small)

    assign = np.array(["train"] * n, dtype=object)
    n_t = n_v = 0
    for s in small:
        if n_t + len(s) <= n_test:
            assign[s] = "test"
            n_t += len(s)
        elif n_v + len(s) <= n_val:
            assign[s] = "val"
            n_v += len(s)
    return assign


def random_split(n: int, fracs=(0.8, 0.1, 0.1), seed: int = 0) -> np.ndarray:
    rng = np.random.RandomState(seed)
    perm = rng.permutation(n)
    n_val, n_test = int(round(fracs[1] * n)), int(round(fracs[2] * n))
    assign = np.array(["train"] * n, dtype=object)
    assign[perm[:n_test]] = "test"
    assign[perm[n_test : n_test + n_val]] = "val"
    return assign


def check_no_group_leak(groups, assign) -> int:
    """Return the number of groups that appear in more than one partition (should be 0)."""
    seen: dict = defaultdict(set)
    for g, a in zip(groups, assign):
        seen[g].add(a)
    return sum(1 for v in seen.values() if len(v) > 1)
