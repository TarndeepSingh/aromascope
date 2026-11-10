"""Inference, score semantics, applicability domain, evidence and explanations."""
from __future__ import annotations

from typing import Any

import numpy as np

from . import chemistry as chem
from .bundle import Release

CALIBRATED = "calibrated_probability"
RAW = "model_score"


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def raw_scores(model: Any, kind: str, X: np.ndarray) -> np.ndarray:
    """Return an [n, n_trainable] matrix of uncalibrated 0-1 scores for either model family."""
    proba = model.predict_proba(X)
    if isinstance(proba, list):  # multi-output forests return one [n, 2] array per label
        return np.column_stack([p[:, 1] if p.shape[1] > 1 else np.zeros(len(p)) for p in proba])
    return np.asarray(proba)


def apply_calibration(scores: np.ndarray, names: list[str], calibration: dict[str, dict[str, float]]):
    """Apply per-label Platt scaling where a validated calibrator exists.

    Returns (values, is_calibrated_mask) with the same shape as `scores`.
    """
    out = scores.copy()
    mask = np.zeros(scores.shape[1], dtype=bool)
    for j, name in enumerate(names):
        c = calibration.get(name)
        if c:
            out[:, j] = _sigmoid(c["a"] * _logit(scores[:, j]) + c["b"])
            mask[j] = True
    return out, mask


class Predictor:
    def __init__(self, release: Release):
        self.r = release
        self._trainable_names = [release.label_names[i] for i in release.trainable_idx]
        self._by_name = {l["name"]: l for l in release.labels}

    # ---- features -----------------------------------------------------------------------
    def _features(self, std: chem.StandardizedMolecule) -> tuple[np.ndarray, np.ndarray]:
        bits = chem.morgan_bits(std.mol)
        phys = chem.physchem(std.mol)
        x = chem.feature_vector(bits, phys, self.r.feat_mean, self.r.feat_std)
        return x, bits

    # ---- applicability --------------------------------------------------------------------
    def neighbors(self, bits: np.ndarray, k: int = 5) -> list[tuple[int, float]]:
        q = bits.astype(np.float32)
        inter = self.r.ref_bits @ q
        union = self.r.ref_bit_counts + q.sum() - inter
        sim = np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)
        order = np.argsort(-sim)[:k]
        return [(int(i), float(sim[i])) for i in order]

    def applicability(self, std: chem.StandardizedMolecule, max_sim: float) -> dict[str, Any]:
        d = self.r.domain
        t_in, t_out = d["tau_in_domain"], d["tau_out_of_domain"]
        mw_lo, mw_hi = d["mol_weight_range"]
        outside_range = not (mw_lo <= std.mol_weight <= mw_hi)
        if outside_range or max_sim < t_out:
            status = "out_of_domain"
            why = (
                f"Molecular weight {std.mol_weight:.0f} is outside the training range ({mw_lo:.0f}-{mw_hi:.0f})."
                if outside_range
                else f"The closest training molecule has Tanimoto similarity {max_sim:.2f}, below the {t_out:.2f} floor."
            )
        elif max_sim < t_in:
            status = "limited_evidence"
            why = (
                f"The closest training molecule has similarity {max_sim:.2f}; the release treats "
                f"{t_in:.2f} or higher as well covered."
            )
        else:
            status = "in_domain"
            why = f"The closest training molecule has similarity {max_sim:.2f} (well covered at {t_in:.2f}+)."
        return {
            "status": status,
            "basis": "Morgan-fingerprint Tanimoto similarity to the nearest training molecule, plus training molecular-weight range",
            "max_similarity": round(max_sim, 4),
            "thresholds": {"in_domain": t_in, "out_of_domain": t_out},
            "explanation": why,
            "caveat": "Fingerprint similarity measures structural proximity under one representation. It does not guarantee similar odor or a correct prediction.",
        }

    # ---- main entry -----------------------------------------------------------------------
    def predict(self, std: chem.StandardizedMolecule, top_k: int = 15, explain: str | None = None) -> dict[str, Any]:
        x, bits = self._features(std)
        neigh = self.neighbors(bits, k=5)
        max_sim = neigh[0][1] if neigh else 0.0
        applic = self.applicability(std, max_sim)

        raw = raw_scores(self.r.model, self.r.model_kind, x[None, :])
        values, cal_mask = apply_calibration(raw, self._trainable_names, self.r.calibration)
        order = np.argsort(-values[0])
        abstained = applic["status"] == "out_of_domain"

        predictions = []
        if not abstained:
            for j in order[:top_k]:
                meta = self._by_name[self._trainable_names[j]]
                predictions.append(
                    {
                        "descriptor": meta["name"],
                        "value": round(float(values[0, j]), 4),
                        "score_type": CALIBRATED if cal_mask[j] else RAW,
                        "reliability": meta.get("reliability", "unrated"),
                        "train_support": meta.get("train_support"),
                        "test_average_precision": meta.get("test_ap"),
                    }
                )

        return {
            "abstained": abstained,
            "predictions": predictions,
            "applicability": applic,
            "evidence": self._evidence(neigh),
            "explanation": self._explain(std, x, bits, explain, predictions) if explain and not abstained else None,
            "projection": self._project(bits),
        }

    # ---- evidence -------------------------------------------------------------------------
    def _evidence(self, neigh: list[tuple[int, float]]) -> dict[str, Any]:
        from rdkit import Chem

        items = []
        for idx, sim in neigh:
            smi = self.r.ref_smiles[idx]
            m = Chem.MolFromSmiles(smi)
            item: dict[str, Any] = {
                "smiles": smi,
                "similarity": round(sim, 4),
                "svg": chem.render_svg(m, size=(200, 150)) if m is not None else None,
            }
            if self.r.ref_labels is not None:
                item["labels"] = self.r.ref_labels.get(smi, [])
            items.append(item)
        ledger = self.r.ledger.get("sources", [{}])[0]
        return {
            "method": "Nearest training molecules by Morgan-fingerprint Tanimoto similarity",
            "labels_included": self.r.ref_labels is not None,
            "labels_note": None
            if self.r.ref_labels is not None
            else "Reference labels are withheld in this release pending a data-rights review (see Model & Data).",
            "attribution": ledger.get("attribution", ""),
            "neighbors": items,
        }

    # ---- map projection ---------------------------------------------------------------------
    def _project(self, bits: np.ndarray) -> dict[str, float]:
        xy = (bits.astype(np.float32) - self.r.proj_mean) @ self.r.proj_components.T
        return {"x": round(float(xy[0]), 4), "y": round(float(xy[1]), 4)}

    # ---- explanation (linear models only) -----------------------------------------------------
    def _explain(self, std, x, bits, descriptor, predictions) -> dict[str, Any]:
        base = {"descriptor": descriptor, "available": False, "svg": None}
        if descriptor not in self._trainable_names:
            return {**base, "note": "Unknown descriptor."}
        if self.r.model_kind != "logreg":
            return {**base, "note": "Substructure attribution is only available for the linear model."}
        try:
            j = self._trainable_names.index(descriptor)
            coef = self.r.model.estimators_[j].coef_[0][: chem.N_BITS]
            env = chem.morgan_bit_atoms(std.mol)
            atom_w: dict[int, float] = {}
            for bit, atom_sets in env.items():
                c = float(coef[bit])
                if c <= 0:
                    continue
                for atoms in atom_sets:
                    for a in atoms:
                        atom_w[a] = atom_w.get(a, 0.0) + c / max(len(atoms), 1)
            if not atom_w:
                return {**base, "note": "No substructure pushed this descriptor up for this molecule."}
            top = max(atom_w.values())
            weights = {a: w / top for a, w in atom_w.items()}
            return {
                "descriptor": descriptor,
                "available": True,
                "svg": chem.render_svg(std.mol, highlights=weights),
                "note": "Glow marks atoms in substructures with positive model weight for this descriptor. These are model associations, not causal chemistry.",
            }
        except Exception:
            return {**base, "note": "Explanation could not be generated for this molecule."}
