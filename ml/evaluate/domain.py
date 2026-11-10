"""Applicability-domain thresholds chosen against validation error versus coverage."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score


def max_tanimoto(query_bits: np.ndarray, ref_bits: np.ndarray, exclude_self: bool = False) -> np.ndarray:
    """Max Tanimoto similarity of each query row to the reference set (both 0/1 matrices)."""
    q = query_bits.astype(np.float32)
    r = ref_bits.astype(np.float32)
    inter = q @ r.T
    union = q.sum(1)[:, None] + r.sum(1)[None, :] - inter
    sim = np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)
    if exclude_self:
        np.fill_diagonal(sim, 0)
    return sim.max(axis=1)


def per_molecule_ap(Y: np.ndarray, P: np.ndarray) -> np.ndarray:
    out = np.full(len(Y), np.nan)
    for i in range(len(Y)):
        if 0 < Y[i].sum() < Y.shape[1]:
            out[i] = average_precision_score(Y[i], P[i])
    return out


BINS = [0.0, 0.3, 0.4, 0.5, 0.6, 0.8, 1.01]


def coverage_table(sim: np.ndarray, ap: np.ndarray) -> list[dict]:
    rows = []
    for lo, hi in zip(BINS[:-1], BINS[1:]):
        m = (sim >= lo) & (sim < hi)
        vals = ap[m & np.isfinite(ap)]
        rows.append(
            {
                "similarity_range": [lo, min(hi, 1.0)],
                "n_molecules": int(m.sum()),
                "mean_per_molecule_ap": float(vals.mean()) if len(vals) else None,
            }
        )
    return rows


def choose_thresholds(sim: np.ndarray, ap: np.ndarray, floor: float = 0.30) -> dict:
    """tau_in: lowest similarity at which molecules at or above it score at least 10% better
    (per-molecule AP) than the validation average, clamped to [0.35, 0.70]. This is a
    documented heuristic; the full error-vs-coverage table is in the evaluation report."""
    ok = np.isfinite(ap)
    overall = float(np.nanmean(ap[ok])) if ok.any() else 0.0
    chosen = 0.5
    for t in np.arange(0.35, 0.701, 0.05):
        sel = ok & (sim >= t)
        if sel.sum() >= 10 and ap[sel].mean() >= 1.10 * overall:
            chosen = float(round(t, 2))
            break
    else:
        chosen = 0.5
    return {"tau_in_domain": chosen, "tau_out_of_domain": floor, "validation_mean_ap": overall}
