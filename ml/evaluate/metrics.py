"""Metrics for multi-label descriptor prediction. Labels with no positives are skipped, never
scored as zero."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve


def per_label_ap(Y: np.ndarray, P: np.ndarray) -> np.ndarray:
    """Average precision per label; NaN where the label has no positives (or only positives)."""
    out = np.full(Y.shape[1], np.nan)
    for j in range(Y.shape[1]):
        pos = Y[:, j].sum()
        if 0 < pos < len(Y):
            out[j] = average_precision_score(Y[:, j], P[:, j])
    return out


def macro_ap(Y: np.ndarray, P: np.ndarray) -> float:
    return float(np.nanmean(per_label_ap(Y, P)))


def micro_ap(Y: np.ndarray, P: np.ndarray) -> float:
    keep = (Y.sum(axis=0) > 0) & (Y.sum(axis=0) < len(Y))
    return float(average_precision_score(Y[:, keep].ravel(), P[:, keep].ravel()))


def top_k_hit(Y: np.ndarray, P: np.ndarray, k: int = 3) -> float:
    """Share of molecules (with at least one true label) whose top-k contains a true label."""
    rows = np.where(Y.sum(axis=1) > 0)[0]
    top = np.argsort(-P[rows], axis=1)[:, :k]
    hits = [Y[r, top[i]].any() for i, r in enumerate(rows)]
    return float(np.mean(hits))


def best_f1_threshold(y: np.ndarray, p: np.ndarray) -> tuple[float, float]:
    """Threshold maximizing F1 on the given data; returns (threshold, f1)."""
    if y.sum() == 0:
        return 0.5, 0.0
    prec, rec, thr = precision_recall_curve(y, p)
    f1 = 2 * prec * rec / np.clip(prec + rec, 1e-12, None)
    f1 = f1[:-1]
    if len(f1) == 0:
        return 0.5, 0.0
    i = int(np.argmax(f1))
    return float(thr[i]), float(f1[i])


def f1_at(y: np.ndarray, p: np.ndarray, thr: float) -> float:
    pred = p >= thr
    tp = float((pred & (y == 1)).sum())
    fp = float((pred & (y == 0)).sum())
    fn = float(((~pred) & (y == 1)).sum())
    return 0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn)


def bootstrap_macro_ap(Y, P, n_boot: int = 200, seed: int = 0) -> tuple[float, float]:
    """95% bootstrap interval for macro AP, resampling molecules."""
    rng = np.random.RandomState(seed)
    n = len(Y)
    vals = []
    for _ in range(n_boot):
        idx = rng.randint(0, n, n)
        v = per_label_ap(Y[idx], P[idx])
        if np.isfinite(v).any():
            vals.append(np.nanmean(v))
    if not vals:
        return float("nan"), float("nan")
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def brier(Y: np.ndarray, P: np.ndarray) -> float:
    return float(np.mean((P - Y) ** 2))


def ece(Y: np.ndarray, P: np.ndarray, bins: int = 10) -> float:
    """Pooled expected calibration error over all molecule-label pairs, using quantile bins
    because positives are rare."""
    y, p = Y.ravel(), P.ravel()
    edges = np.unique(np.quantile(p, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return float(abs(p.mean() - y.mean()))
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, len(edges) - 2)
    total = 0.0
    for b in np.unique(idx):
        m = idx == b
        total += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(total)


def reliability_tier(lift: float | None, test_support: int, min_support: int = 5) -> str:
    """Plain-language tier shown beside each descriptor. Based on held-out lift over prevalence."""
    if lift is None or test_support < min_support:
        return "unrated"
    if lift >= 4:
        return "strong"
    if lift >= 2:
        return "moderate"
    return "weak"


def clean(obj):
    """Recursively replace NaN/inf with None so reports are valid JSON for browsers and the API."""
    if isinstance(obj, dict):
        return {k: clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [clean(v) for v in obj]
    if isinstance(obj, (np.floating, float)):
        return None if not np.isfinite(obj) else float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    return obj
