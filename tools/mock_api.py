"""DEV-ONLY mock of the AromaScope API contract. Returns deterministic SAMPLE data, not model output.

Use it to work on the web client without RDKit or a trained release:
    python tools/mock_api.py            # serves the API and the web client on http://localhost:8080
The web client shows a "Demo mode" banner whenever it talks to this server.
"""
import hashlib
import json
import math
import os
import random
import sys
from pathlib import Path

from starlette.applications import Starlette
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import JSONResponse
from starlette.routing import Mount, Route
from starlette.staticfiles import StaticFiles

NAMES = ["fruity", "green", "sweet", "floral", "fatty", "herbal", "waxy", "sulfurous", "woody", "citrus", "spicy", "minty", "balsamic",
         "creamy", "nutty", "earthy", "roasted", "vanilla", "almond", "alcoholic", "ethereal", "pungent", "rose", "apple", "pineapple", "musty", "phenolic", "caramellic"]
LIMITS = ["Predicted from structure only; this is not a sensory measurement and not an air sensor.",
          "Applies to the reference vocabulary and source dataset, not to every person's perception.",
          "Purity, concentration, temperature and mixtures are not modelled.",
          "Structural similarity does not guarantee similar odor. The model can be wrong.",
          "Model features ignore stereochemistry: enantiomers receive identical predictions."]
RNG = random.Random(7)
LABELS = []
for n in NAMES:
    tr = RNG.randint(40, 900); ap = round(RNG.uniform(0.12, 0.72), 3); prev = round(RNG.uniform(0.02, 0.2), 3); lift = ap / prev
    LABELS.append({"name": n, "trainable": True, "train_support": tr, "val_support": tr // 8, "test_support": tr // 8, "test_prevalence": prev, "test_ap": ap, "lift_over_prevalence": lift,
                   "reliability": "strong" if lift >= 4 else "moderate" if lift >= 2 else "weak", "calibrated": True})
BY = {l["name"]: l for l in LABELS}
ANCHORS = {"O=Cc1ccccc1": ["almond", "sweet", "phenolic"], "COc1cc(C=O)ccc1O": ["vanilla", "sweet", "balsamic", "creamy", "caramellic"], "CCCC(=O)OCC": ["fruity", "pineapple", "sweet", "apple", "ethereal"]}


def h(s):
    return int(hashlib.sha256(s.encode()).hexdigest(), 16)


def hexagon(extra=""):
    pts = " ".join(f"{210 + 78 * math.cos(math.pi / 3 * i - math.pi / 6):.1f},{150 + 78 * math.sin(math.pi / 3 * i - math.pi / 6):.1f}" for i in range(6))
    return f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 420 300"><polygon points="{pts}" fill="none" stroke="#e8edf6" stroke-width="3" stroke-linejoin="round"/>{extra}<text x="210" y="290" text-anchor="middle" fill="#8793a6" font-size="12">mock depiction</text></svg>'


def profile(smiles):
    r = random.Random(h(smiles))
    order = list(NAMES); r.shuffle(order)
    if smiles in ANCHORS:
        order = ANCHORS[smiles] + [n for n in order if n not in ANCHORS[smiles]]
    vals = sorted([r.uniform(0.05, 0.28) for _ in order], reverse=True)
    vals[0] = r.uniform(0.55, 0.82)
    vals[1] = r.uniform(0.38, 0.52)
    vals[2] = r.uniform(0.28, 0.4)
    vals.sort(reverse=True)
    return list(zip(order, vals))


async def predictions(request):
    body = await request.json()
    s = (body.get("smiles") or "").strip()
    rid = "mock" + hashlib.md5(s.encode()).hexdigest()[:8]
    if not s:
        return JSONResponse({"error": {"code": "empty_input", "message": "Enter a SMILES string.", "hint": "Try CCO (ethanol).", "retriable": False}, "request_id": rid}, status_code=422)
    if "!" in s or s.count("(") != s.count(")") or s.startswith("xx"):
        return JSONResponse({"error": {"code": "invalid_smiles", "message": "This is not a valid SMILES string.", "hint": "Check bracket pairs, ring-closure digits and aromatic atom valences.", "retriable": False}, "request_id": rid}, status_code=422)
    if "." in s:
        return JSONResponse({"error": {"code": "unsupported_mixture", "message": "Multiple components (a mixture or salt) were found.", "hint": "Submit the parent molecule only.", "retriable": False}, "request_id": rid}, status_code=422)
    status = "out_of_domain" if "I" in s else "limited_evidence" if "Cl" in s or "Br" in s else "in_domain"
    sim = {"out_of_domain": 0.18, "limited_evidence": 0.41}.get(status, 0.5 + (h(s) % 40) / 100)
    top_k = int(body.get("top_k") or 15)
    prof = profile(s)[:top_k]
    preds = [{"descriptor": n, "value": round(v, 4), "score_type": "calibrated_probability" if i % 5 != 4 else "model_score", "reliability": BY[n]["reliability"],
              "train_support": BY[n]["train_support"], "test_average_precision": BY[n]["test_ap"]} for i, (n, v) in enumerate(prof)]
    ex = None
    if body.get("explain"):
        ex = {"descriptor": body["explain"], "available": True, "svg": hexagon('<circle cx="210" cy="72" r="34" fill="rgba(66,215,197,.45)"/><circle cx="278" cy="111" r="26" fill="rgba(66,215,197,.3)"/>'),
              "note": "MOCK highlight. Real releases mark atoms in substructures with positive model weight; these are associations, not causal chemistry."}
    pr = random.Random(h(s) + 1)
    nb = [{"smiles": ["CCCCC(=O)OCC", "CC(=O)OCCC(C)C", "CCCCCC(=O)OCC", "CCOC(=O)C=C", "CCCC(=O)OC"][i], "similarity": round(max(sim - 0.07 * i, 0.12), 3), "svg": hexagon()} for i in range(5)]
    return JSONResponse({
        "request_id": rid,
        "molecule": {"input_smiles": s, "canonical_smiles": s, "formula": "C?H?O?", "mol_weight": 100 + (h(s) % 200), "heavy_atoms": 8 + len(s) // 3, "unspecified_stereo": "@" not in s and "C(C)C" in s,
                     "changes": [], "structure_svg": hexagon(), "projection": {"x": round(pr.uniform(-2, 2), 3), "y": round(pr.uniform(-1.5, 1.5), 3)}},
        "model": {"id": "mock-dev", "version": "0.0.0-mock", "data_snapshot": "mock", "featurizer_version": "mock", "policy_version": "mock", "kind": "logreg"},
        "task": TASK, "abstained": status == "out_of_domain", "predictions": [] if status == "out_of_domain" else preds,
        "applicability": {"status": status, "basis": "Mock similarity", "max_similarity": sim, "thresholds": {"in_domain": 0.5, "out_of_domain": 0.3},
                          "explanation": {"in_domain": f"The closest training molecule has similarity {sim:.2f} (well covered at 0.50+).", "limited_evidence": f"The closest training molecule has similarity {sim:.2f}; the release treats 0.50 or higher as well covered.",
                                          "out_of_domain": f"The closest training molecule has Tanimoto similarity {sim:.2f}, below the 0.30 floor."}[status],
                          "caveat": "Fingerprint similarity measures structural proximity under one representation. It does not guarantee similar odor or a correct prediction."},
        "evidence": {"method": "Nearest training molecules by Morgan-fingerprint Tanimoto similarity", "labels_included": False,
                     "labels_note": "Reference labels are withheld in this release pending a data-rights review (see Model & Data).", "attribution": "Mock data. No real source.", "neighbors": nb},
        "explanation": ex, "limitations": LIMITS})


TASK = {"target_type": "multilabel_descriptor", "description": "Probability that each descriptor is listed for a molecule like this in the reference dataset.", "scale": "0 to 1", "descriptors": len(NAMES), "reference_population": "MOCK sample data"}


async def model(request):
    cov = [{"similarity_range": [lo, hi], "n_molecules": n, "mean_per_molecule_ap": ap} for (lo, hi, n, ap) in [(0, .3, 18, .21), (.3, .4, 64, .28), (.4, .5, 92, .36), (.5, .6, 71, .44), (.6, .8, 60, .52), (.8, 1, 22, .68)]]
    ev = {"models": {"prevalence": {"val_macro_ap": .09, "test_macro_ap": .09}, "logreg": {"val_macro_ap": .31, "C": .3}, "random_forest": {"val_macro_ap": .30, "promoted": False}},
          "chosen_model": "logreg", "test": {"n_molecules": 327, "macro_ap": .318, "macro_ap_ci95": [.29, .345], "micro_ap": .402, "top3_hit_rate": .83, "baseline_macro_ap": .09},
          "random_split_comparison": {"test_macro_ap_random_split": .36, "test_macro_ap_scaffold_split_raw": .31},
          "calibration": {"meaning": "Calibrated value = estimated probability that the descriptor is listed for a molecule like this in the reference dataset.", "labels_calibrated": 24, "brier_raw": .0451, "brier_calibrated": .0398, "ece_raw": .031, "ece_calibrated": .008, "gate_passed": True},
          "domain": {"tau_in_domain": .5, "tau_out_of_domain": .3, "share_test_in_domain": .49, "coverage_table_test": cov},
          "reliability_rule": "strong: held-out AP >= 4x prevalence; moderate: >= 2x; weak: < 2x; unrated: < 5 test positives"}
    return JSONResponse({"model": {"id": "mock-dev", "version": "0.0.0-mock", "kind": "logreg", "created_at": "mock", "code_revision": "mock", "data_snapshot": "mock", "featurizer_version": "mock", "policy_version": "mock",
                                   "split": {"method": "scaffold-grouped", "train": 2815, "val": 352, "test": 327}, "checksums": {"model.joblib": "0" * 64}},
                         "task": TASK, "labels": LABELS, "evaluation": ev,
                         "ledger": {"sources": [{"name": "MOCK dataset", "doi": "n/a", "pyrfume_commit": "mock", "rights": {"dataset_terms": "Mock data; no real rights.", "redistribution_approved": False}}],
                                    "chemistry_coverage": {"records_in": 3522, "parse_or_policy_failures": 7, "duplicate_records_merged": 12, "molecules_out": 3494}, "label_handling": {"missingness": "Mock note on missing labels.", "aggregation": "No rater-level data."}},
                         "model_card_markdown": "# MOCK model card\nThis is sample text from the development mock server.", "limitations": LIMITS})


async def omap(request):
    r = random.Random(3); pts = []
    centers = [(-2.2, 1.0), (1.8, 1.4), (0.2, -1.6), (2.6, -0.8), (-1.0, -0.4), (0.4, 0.6)]
    for i in range(900):
        c = r.randrange(len(centers)); cx, cy = centers[c]
        pts.append({"i": i, "smiles": "C" * (2 + i % 9) + "O", "x": round(r.gauss(cx, .55), 3), "y": round(r.gauss(cy, .5), 3), "split": r.choice(["train"] * 8 + ["val", "test"]), "top": [0, 1, 2, 3, 4, 5][c] if r.random() > .25 else r.randrange(len(NAMES)), "top_score": round(r.uniform(.3, .8), 3)})
    return JSONResponse({"projection": "PCA of Morgan fingerprints (structure projection, not a perceptual map)", "explained_variance": [.11, .07], "descriptors": NAMES, "points": pts})


async def ready(request):
    return JSONResponse({"status": "ready", "model_version": "mock-0.0.0", "warnings": []})


routes = [Route("/health/ready", ready), Route("/health/live", ready), Route("/api/v1/predictions", predictions, methods=["POST"]),
          Route("/api/v1/model", model), Route("/api/v1/map", omap)]
web = Path(__file__).resolve().parent.parent / "apps" / "web"
routes.append(Mount("/", StaticFiles(directory=str(web), html=True)))
app = Starlette(routes=routes)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=int(os.environ.get("PORT", "8080")), log_level="warning")
