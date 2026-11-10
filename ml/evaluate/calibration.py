"""Per-label Platt scaling fit on validation scores."""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def fit_platt(P_val: np.ndarray, Y_val: np.ndarray, names: list[str], min_pos: int = 10) -> dict[str, dict[str, float]]:
    """Fit a sigmoid on logit(score) for each label with at least `min_pos` validation positives."""
    out: dict[str, dict[str, float]] = {}
    for j, name in enumerate(names):
        y = Y_val[:, j]
        if y.sum() < min_pos or y.sum() == len(y):
            continue
        lr = LogisticRegression(C=1e4, solver="lbfgs", max_iter=500)
        lr.fit(logit(P_val[:, [j]]), y)
        out[name] = {"a": float(lr.coef_[0, 0]), "b": float(lr.intercept_[0])}
    return out
