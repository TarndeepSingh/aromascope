"""Offline training, evaluation and release assembly.

Usage:
    python -m ml.train.train --version 0.1.0

Pipeline: load processed data -> scaffold split -> Stage 0 prevalence baseline, Stage 1 regularized
logistic regression on Morgan+physchem, Stage 2 random forest -> choose by validation macro AP
(with a size gate) -> per-label calibration -> applicability-domain thresholds -> frozen test
evaluation -> release bundle with checksum manifest. Training is never triggered by an API request.
"""
from __future__ import annotations

import argparse
import datetime as dt
import io
import json
import subprocess
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from rdkit import Chem
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier

from aromascope import __version__ as CODE_VERSION
from aromascope import chemistry as chem
from aromascope.bundle import REQUIRED_FILES, write_manifest
from aromascope.predictor import apply_calibration, raw_scores
from ml.evaluate import calibration as cal
from ml.evaluate import domain as dom
from ml.evaluate import metrics as M
from ml.evaluate import splits as S

MAX_PROMOTABLE_MB = 40  # larger candidates are benchmarked but not shipped
RF_MARGIN = 1.02  # RF must beat the linear model by 2% on validation macro AP to be promoted


def log(msg: str) -> None:
    print(f"[train] {msg}", flush=True)


def git_rev() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def featurize_frame(smiles: list[str]):
    bits, phys = [], []
    for s in smiles:
        mol = Chem.MolFromSmiles(s)
        bits.append(chem.morgan_bits(mol))
        phys.append(chem.physchem(mol))
    return np.vstack(bits), np.vstack(phys)


def fit_logreg(X, Y, C, n_jobs):
    est = OneVsRestClassifier(LogisticRegression(C=C, solver="liblinear", max_iter=300), n_jobs=n_jobs)
    return est.fit(X, Y)


def fit_forest(X, Y, seed, n_jobs):
    return RandomForestClassifier(
        n_estimators=60, min_samples_leaf=5, max_depth=14, max_features="sqrt", n_jobs=n_jobs, random_state=seed
    ).fit(X, Y)


def size_mb(obj) -> float:
    buf = io.BytesIO()
    joblib.dump(obj, buf, compress=3)
    return buf.tell() / 1e6


def run(args) -> Path:
    processed = Path(args.processed)
    mols = pd.read_csv(processed / "molecules.csv")
    Ydf = pd.read_csv(processed / "labels.csv").set_index("mol_id").loc[mols["mol_id"]]
    ledger = json.loads((processed / "ledger.json").read_text())
    cfg = json.loads(Path(args.config).read_text())
    names_all = list(Ydf.columns)
    Y_all = Ydf.values.astype(int)
    log(f"{len(mols)} molecules, {len(names_all)} descriptors")

    bits, phys = featurize_frame(mols["canonical_smiles"].tolist())
    assign = S.scaffold_split(mols["group"].tolist(), seed=args.seed)
    leaks = S.check_no_group_leak(mols["group"].tolist(), assign)
    assert leaks == 0, f"{leaks} scaffold groups cross partitions"
    tr, va, te = (assign == "train"), (assign == "val"), (assign == "test")
    log(f"split sizes train/val/test = {tr.sum()}/{va.sum()}/{te.sum()} (group leaks: {leaks})")

    mean, std = phys[tr].mean(axis=0), phys[tr].std(axis=0)
    std[std == 0] = 1.0
    X = chem.feature_vector(bits, phys, mean, std)

    pos_train = Y_all[tr].sum(axis=0)
    trainable = np.where(pos_train >= args.min_train_pos)[0]
    names = [names_all[i] for i in trainable]
    Y = Y_all[:, trainable]
    log(f"{len(trainable)}/{len(names_all)} descriptors have >= {args.min_train_pos} training positives")

    report: dict = {"models": {}}
    # ---- Stage 0: prevalence baseline ------------------------------------------------------------
    prev = Y[tr].mean(axis=0)
    P0_val, P0_test = np.tile(prev, (va.sum(), 1)), np.tile(prev, (te.sum(), 1))
    report["models"]["prevalence"] = {
        "val_macro_ap": M.macro_ap(Y[va], P0_val),
        "test_macro_ap": M.macro_ap(Y[te], P0_test),
        "note": "Predicts each descriptor's training prevalence for every molecule.",
    }

    # ---- Stage 1: regularized logistic regression -----------------------------------------------------
    best = None
    for C in [0.03, 0.1, 0.3, 1.0]:
        m = fit_logreg(X[tr], Y[tr], C, args.n_jobs)
        v = M.macro_ap(Y[va], raw_scores(m, "logreg", X[va]))
        log(f"logreg C={C}: val macro AP {v:.4f}")
        if best is None or v > best[0]:
            best = (v, C, m)
    lr_val, lr_C, lr_model = best
    report["models"]["logreg"] = {"val_macro_ap": lr_val, "C": lr_C, "size_mb": round(size_mb(lr_model), 2)}

    # ---- Stage 2: random forest (benchmark; promoted only if clearly better and small) -------------
    chosen_kind, chosen_model, chosen_val = "logreg", lr_model, lr_val
    if not args.skip_rf:
        rf = fit_forest(X[tr], Y[tr], args.seed, args.n_jobs)
        rf_val = M.macro_ap(Y[va], raw_scores(rf, "rf", X[va]))
        rf_mb = size_mb(rf)
        promoted = rf_val > lr_val * RF_MARGIN and rf_mb <= MAX_PROMOTABLE_MB
        report["models"]["random_forest"] = {
            "val_macro_ap": rf_val,
            "size_mb": round(rf_mb, 2),
            "promoted": bool(promoted),
            "promotion_rule": f"val macro AP > {RF_MARGIN}x logistic regression and size <= {MAX_PROMOTABLE_MB} MB",
        }
        log(f"random forest: val macro AP {rf_val:.4f}, {rf_mb:.1f} MB, promoted={promoted}")
        if promoted:
            chosen_kind, chosen_model, chosen_val = "rf", rf, rf_val
    report["chosen_model"] = chosen_kind

    # ---- Calibration (fit on validation only) -------------------------------------------------------
    R_val = raw_scores(chosen_model, chosen_kind, X[va])
    R_test = raw_scores(chosen_model, chosen_kind, X[te])
    calib = cal.fit_platt(R_val, Y[va], names, min_pos=args.min_cal_pos)
    C_test, cmask = apply_calibration(R_test, names, calib)
    ece_raw, ece_cal = M.ece(Y[te], R_test), M.ece(Y[te], C_test)
    brier_raw, brier_cal = M.brier(Y[te], R_test), M.brier(Y[te], C_test)
    gate_cal = bool(len(calib) > 0 and brier_cal <= brier_raw and ece_cal <= ece_raw)
    if not gate_cal:
        log("calibration did not improve held-out Brier and ECE; shipping raw model scores")
        calib = {}
        C_test, cmask = R_test, np.zeros(len(names), dtype=bool)
    report["calibration"] = {
        "method": "per-descriptor Platt scaling on logit(score), fit on validation",
        "labels_calibrated": int(len(calib)),
        "brier_raw": brier_raw,
        "brier_calibrated": brier_cal,
        "ece_raw": ece_raw,
        "ece_calibrated": ece_cal,
        "gate_passed": gate_cal,
        "meaning": "Calibrated value = estimated probability that the descriptor is listed for a molecule like this in the reference dataset (not a perceptual probability).",
    }
    C_val, _ = apply_calibration(R_val, names, calib)

    # ---- Frozen test evaluation ---------------------------------------------------------------------------
    lo, hi = M.bootstrap_macro_ap(Y[te], C_test, n_boot=args.n_boot, seed=args.seed)
    ap_test = M.per_label_ap(Y[te], C_test)
    report["test"] = {
        "n_molecules": int(te.sum()),
        "macro_ap": M.macro_ap(Y[te], C_test),
        "macro_ap_ci95": [lo, hi],
        "micro_ap": M.micro_ap(Y[te], C_test),
        "top3_hit_rate": M.top_k_hit(Y[te], C_test, 3),
        "baseline_macro_ap": report["models"]["prevalence"]["test_macro_ap"],
    }

    # Secondary comparison: random (leaky) split of the same data, to show how much scaffold
    # grouping matters. Never used for model selection.
    rs = S.random_split(len(mols), seed=args.seed)
    rm = fit_logreg(X[rs == "train"], Y[rs == "train"], lr_C, args.n_jobs) if chosen_kind == "logreg" else None
    if rm is not None:
        report["random_split_comparison"] = {
            "test_macro_ap_random_split": M.macro_ap(Y[rs == "test"], raw_scores(rm, "logreg", X[rs == "test"])),
            "test_macro_ap_scaffold_split_raw": M.macro_ap(Y[te], R_test),
            "note": "A random split lets close analogs cross the boundary and is a secondary comparison only.",
        }

    # ---- Per-label table, thresholds, reliability ---------------------------------------------------------
    labels_meta = []
    for k, i_all in enumerate(range(len(names_all))):
        name = names_all[i_all]
        if i_all in set(trainable.tolist()):
            j = int(np.where(trainable == i_all)[0][0])
            te_pos = int(Y[te][:, j].sum())
            prevalence = te_pos / max(te.sum(), 1)
            ap = float(ap_test[j]) if np.isfinite(ap_test[j]) else None
            lift = (ap / prevalence) if (ap is not None and prevalence > 0) else None
            thr, _ = M.best_f1_threshold(Y[va][:, j], C_val[:, j])
            labels_meta.append(
                {
                    "name": name,
                    "trainable": True,
                    "train_support": int(Y[tr][:, j].sum()),
                    "val_support": int(Y[va][:, j].sum()),
                    "test_support": te_pos,
                    "test_prevalence": prevalence,
                    "test_ap": ap,
                    "lift_over_prevalence": lift,
                    "test_f1_at_val_threshold": M.f1_at(Y[te][:, j], C_test[:, j], thr),
                    "val_f1_threshold": thr,
                    "calibrated": bool(name in calib),
                    "reliability": M.reliability_tier(lift, te_pos),
                }
            )
        else:
            labels_meta.append(
                {"name": name, "trainable": False, "train_support": int(Y_all[tr][:, i_all].sum()), "reliability": "unrated"}
            )
    report["per_label"] = labels_meta
    report["reliability_rule"] = "strong: held-out AP >= 4x prevalence; moderate: >= 2x; weak: < 2x; unrated: < 5 test positives"

    # ---- Applicability domain ------------------------------------------------------------------------------
    sim_val = dom.max_tanimoto(bits[va], bits[tr])
    sim_test = dom.max_tanimoto(bits[te], bits[tr])
    ap_mol_val = dom.per_molecule_ap(Y[va], C_val)
    ap_mol_test = dom.per_molecule_ap(Y[te], C_test)
    thr = dom.choose_thresholds(sim_val, ap_mol_val)
    thr["mol_weight_range"] = [float(phys[tr][:, 0].min()), float(phys[tr][:, 0].max())]
    thr["coverage_table_validation"] = dom.coverage_table(sim_val, ap_mol_val)
    thr["coverage_table_test"] = dom.coverage_table(sim_test, ap_mol_test)
    thr["share_test_in_domain"] = float((sim_test >= thr["tau_in_domain"]).mean())
    thr["share_test_out_of_domain"] = float((sim_test < thr["tau_out_of_domain"]).mean())
    report["domain"] = {k: v for k, v in thr.items()}
    log(f"domain thresholds: in>={thr['tau_in_domain']}, out<{thr['tau_out_of_domain']}")

    # ---- Odor map (structure projection; coordinates are NOT a perceptual space) --------------------------
    pca = PCA(n_components=2, random_state=args.seed).fit(bits[tr].astype(np.float32))
    xy = pca.transform(bits.astype(np.float32))
    R_all = raw_scores(chosen_model, chosen_kind, X)
    P_all, _ = apply_calibration(R_all, names, calib)
    top_idx = P_all.argmax(axis=1)
    map_data = {
        "projection": "PCA of Morgan fingerprints (structure projection, not a perceptual map)",
        "explained_variance": [float(v) for v in pca.explained_variance_ratio_],
        "descriptors": names,
        "points": [
            {
                "i": int(i),
                "smiles": mols["canonical_smiles"].iloc[i],
                "x": round(float(xy[i, 0]), 3),
                "y": round(float(xy[i, 1]), 3),
                "split": str(assign[i]),
                "top": int(top_idx[i]),
                "top_score": round(float(P_all[i, top_idx[i]]), 3),
            }
            for i in range(len(mols))
        ],
    }

    # ---- Assemble release -------------------------------------------------------------------------------------
    out = Path(args.out) / args.version
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {"estimator": chosen_model, "kind": chosen_kind, "trainable_idx": trainable.tolist()},
        out / "model.joblib",
        compress=3,
    )
    np.savez(out / "features.npz", mean=mean, std=std)
    np.savez_compressed(out / "reference.npz", packed=np.packbits(bits[tr], axis=1), n_bits=chem.N_BITS)
    ref_smiles = mols["canonical_smiles"][tr].tolist()
    (out / "reference_smiles.json").write_text(json.dumps(ref_smiles))
    if cfg.get("redistribution_approved"):
        lab_map = {
            s: [names_all[i] for i in np.where(Y_all[k] > 0)[0]]
            for k, s in zip(np.where(tr)[0], ref_smiles)
        }
        (out / "reference_labels.json").write_text(json.dumps(lab_map))
    (out / "labels.json").write_text(json.dumps(M.clean(labels_meta), indent=1))
    (out / "calibration.json").write_text(json.dumps(calib))
    (out / "domain.json").write_text(json.dumps(M.clean(thr)))
    (out / "map.json").write_text(json.dumps(map_data, separators=(",", ":")))
    np.savez(out / "projection.npz", mean=bits[tr].astype(np.float32).mean(axis=0), components=pca.components_.astype(np.float32))
    (out / "ledger.json").write_text(json.dumps(ledger, indent=2))
    report = M.clean(report)
    (out / "evaluation_report.json").write_text(json.dumps(report, indent=2))
    (out / "model_card.md").write_text(model_card(args.version, report, ledger, chosen_kind, len(mols), tr, va, te, cfg))

    snapshot = f"leffingwell@pyrfume:{ledger['sources'][0].get('pyrfume_commit', 'unknown')[:8]}"
    write_manifest(
        out,
        {
            "id": "aromascope-odor-profile",
            "version": args.version,
            "created_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
            "code_version": CODE_VERSION,
            "code_revision": git_rev(),
            "data_snapshot": snapshot,
            "featurizer_version": chem.FEATURIZER_VERSION,
            "policy_version": chem.POLICY_VERSION,
            "model_kind": chosen_kind,
            "split": {"method": "scaffold-grouped", "seed": args.seed, "train": int(tr.sum()), "val": int(va.sum()), "test": int(te.sum())},
            "redistribution_approved": bool(cfg.get("redistribution_approved", False)),
            "data_hashes": ledger["sources"][0].get("files", {}),
        },
    )
    (Path(args.out) / "CURRENT").write_text(args.version)
    log(f"release written to {out}")

    if args.mlflow:
        try:
            import mlflow

            with mlflow.start_run(run_name=f"release-{args.version}"):
                mlflow.log_params({"model": chosen_kind, "seed": args.seed, "logreg_C": lr_C})
                mlflow.log_metrics({"test_macro_ap": report["test"]["macro_ap"], "val_macro_ap": chosen_val})
                mlflow.log_artifacts(str(out))
        except Exception as exc:  # tracking is optional and must never break a release
            log(f"MLflow logging skipped: {exc}")
    return out


def model_card(version, report, ledger, kind, n, tr, va, te, cfg) -> str:
    t = report["test"]
    src = ledger["sources"][0]
    strong = [l["name"] for l in report["per_label"] if l.get("reliability") == "strong"]
    weak = [l["name"] for l in report["per_label"] if l.get("reliability") == "weak"]
    c = report["calibration"]
    d = report["domain"]
    return f"""# AromaScope model card, release {version}

**Task.** Multi-label prediction of odor descriptors (binary) from one molecular structure (SMILES).
**Output.** A ranked list of descriptors. Values are {'calibrated probabilities' if c['gate_passed'] else 'uncalibrated model scores'} that the descriptor is listed for a molecule like this in the reference dataset.
These are model-based predictions, **not** sensory measurements and **not** a room-air detector.

## Model
- Family: `{kind}` ({'regularized logistic regression, one per descriptor' if kind == 'logreg' else 'random forest'}), baseline-first selection.
- Features: Morgan fingerprint (radius 2, 2048 bits) + 8 standardized physicochemical descriptors. Stereochemistry is **not** used by the features.
- Selection rule: validation macro average precision; the random forest is promoted only if it beats the linear model by {int((RF_MARGIN-1)*100)}% and fits the size budget.

## Data
- Source: {src['name']}, DOI {src['doi']}; Pyrfume commit `{src.get('pyrfume_commit','unknown')[:10]}`.
- {n} molecules after standardization. Scaffold-grouped split: train {int(tr.sum())}, validation {int(va.sum())}, test {int(te.sum())}.
- Label handling: the source lists positive descriptors only; unlisted descriptors are unknown but are treated as negative in training (positive-unlabeled setting). Reported precision is therefore a conservative estimate for rare descriptors and probabilities describe *dataset listing*, not perception.
- Rights: **{src['rights']['dataset_terms']}**  Redistribution of labels approved: `{cfg.get('redistribution_approved', False)}`.

## Held-out results (frozen test set, {t['n_molecules']} molecules)
- Macro average precision: **{t['macro_ap']:.3f}** (95% bootstrap CI {(t['macro_ap_ci95'][0] or 0):.3f} to {(t['macro_ap_ci95'][1] or 0):.3f}) versus **{t['baseline_macro_ap']:.3f}** for the prevalence baseline.
- Micro average precision: {t['micro_ap']:.3f}. Top-3 hit rate: {t['top3_hit_rate']:.3f}.
- Descriptors with strong held-out performance: {', '.join(strong) or 'none'}.
- Descriptors with weak performance (treat with caution): {', '.join(weak) or 'none'}.

## Calibration
- {c['method']}; {c['labels_calibrated']} descriptors calibrated. Test Brier {c['brier_raw']:.4f} to {c['brier_calibrated']:.4f}; ECE {c['ece_raw']:.4f} to {c['ece_calibrated']:.4f}. Gate passed: {c['gate_passed']}.

## Applicability domain
- Nearest-neighbour Tanimoto similarity to training molecules. Well covered at >= {d['tau_in_domain']}; below {d['tau_out_of_domain']} or outside the training molecular-weight range the model abstains.

## Limitations
- Predictions apply to this reference vocabulary and source, not to every person's perception.
- Structure alone omits purity, concentration, conditions, mixtures and individual differences.
- Similarity does not guarantee sensory similarity; the model can be wrong.
- Substructure highlights are model associations, not causal mechanisms.
- No air sampling is involved. Identifying an unknown airborne odor is out of scope.
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed", default="data/processed")
    ap.add_argument("--out", default="artifacts/releases")
    ap.add_argument("--version", default="0.1.0")
    ap.add_argument("--config", default="ml/config.json")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min-train-pos", type=int, default=5)
    ap.add_argument("--min-cal-pos", type=int, default=10)
    ap.add_argument("--n-boot", type=int, default=200)
    ap.add_argument("--n-jobs", type=int, default=-1)
    ap.add_argument("--skip-rf", action="store_true")
    ap.add_argument("--mlflow", action="store_true")
    run(ap.parse_args())


if __name__ == "__main__":
    main()
