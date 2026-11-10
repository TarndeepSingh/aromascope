"""End to end on a tiny synthetic dataset: build -> train -> release -> API. Labels here come from
SMARTS rules so the model has real signal; this is a pipeline smoke test, not a scientific result."""
import json
import types

import pandas as pd
import pytest

pytest.importorskip("rdkit")
pytest.importorskip("fastapi")
from rdkit import Chem  # noqa: E402

RULES = {
    "fruity": "[CX3](=O)[OX2][#6]",
    "aldehydic": "[CX3H1](=O)[#6]",
    "alcoholic": "[CX4][OX2H]",
    "sour": "[CX3](=O)[OX2H1]",
    "sulfurous": "[SX2H]",
    "green": "[CX3]=[CX3]",
    "aromatic": "c",
}


def synthetic_smiles():
    out = []
    for a in ["C", "CC", "CCC", "CCCC", "CCCCC", "CC(C)C"]:
        for b in ["C", "CC", "CCC", "CCCC", "CCCCC", "CC(C)C"]:
            out.append(f"{a}C(=O)O{b}")
    out += [f"{'C' * n}C=O" for n in range(1, 9)] + [f"{'C' * n}O" for n in range(1, 10)]
    out += [f"{'C' * n}C(=O)O" for n in range(1, 8)] + [f"{'C' * n}S" for n in range(1, 8)]
    out += [f"C=C{'C' * n}" for n in range(1, 8)] + [f"C=CC{'C' * n}O" for n in range(1, 6)]
    rings = ["c1ccccc1{}", "C1CCCCC1{}", "c1ccncc1{}", "c1ccoc1{}", "c1ccsc1{}", "C1CCCC1{}"]
    out += [r.format(sub) for r in rings for sub in ["C", "CC", "CCO", "C=O", "CC(=O)OC", "O", "CCS", "C(=O)O"]]
    return list(dict.fromkeys(out))


@pytest.fixture(scope="module")
def release(tmp_path_factory):
    from ml.data.build_dataset import build
    from ml.train.train import run

    root = tmp_path_factory.mktemp("e2e")
    smiles = synthetic_smiles()
    pats = {k: Chem.MolFromSmarts(v) for k, v in RULES.items()}
    ids = [str(i) for i in range(len(smiles))]
    labels = pd.DataFrame({k: [int(Chem.MolFromSmiles(s).HasSubstructMatch(p)) for s in smiles] for k, p in pats.items()}, index=ids)
    mols = pd.DataFrame({"src_id": ids, "raw_smiles": smiles, "name": smiles, "cas": None})
    source = {"name": "synthetic", "doi": "n/a", "pyrfume_commit": "test", "task": "synthetic rules", "attribution": "test",
              "rights": {"dataset_terms": "synthetic", "redistribution_approved": False}, "files": {}}
    proc = root / "processed"
    build(mols, labels, proc, source)
    cfg = root / "config.json"
    cfg.write_text(json.dumps({"redistribution_approved": False}))
    args = types.SimpleNamespace(processed=str(proc), out=str(root / "releases"), version="0.0.1", config=str(cfg), seed=0, min_train_pos=3,
                                 min_cal_pos=3, n_boot=10, n_jobs=1, skip_rf=False, mlflow=False)
    return run(args), root / "releases"


@pytest.fixture()
def client(release, monkeypatch):
    from fastapi.testclient import TestClient
    from services.api.app import main

    path, _ = release
    monkeypatch.setenv("AROMASCOPE_RELEASE_DIR", str(path))
    monkeypatch.setenv("AROMASCOPE_RATE_LIMIT", "0")
    with TestClient(main.app) as c:
        yield c


def test_release_has_all_files_and_checksums(release):
    from aromascope.bundle import REQUIRED_FILES, verify

    path, _ = release
    manifest, _warn = verify(path)
    assert all(f in manifest["files"] for f in REQUIRED_FILES)


def test_evaluation_report_is_valid_json_without_nan(release):
    path, _ = release
    json.loads((path / "evaluation_report.json").read_text(), parse_constant=lambda c: (_ for _ in ()).throw(ValueError(c)))


def test_health_and_model(client):
    assert client.get("/health/live").status_code == 200
    assert client.get("/health/ready").json()["status"] == "ready"
    m = client.get("/api/v1/model").json()
    assert m["model"]["checksums"] and m["evaluation"]["test"]["n_molecules"] > 0 and m["limitations"]


def test_prediction_contract(client):
    r = client.post("/api/v1/predictions", json={"smiles": "CCCC(=O)OCC"})
    assert r.status_code == 200
    j = r.json()
    for key in ["molecule", "model", "task", "predictions", "applicability", "evidence", "limitations", "abstained"]:
        assert key in j
    assert j["applicability"]["status"] in {"in_domain", "limited_evidence", "out_of_domain"}
    assert j["molecule"]["structure_svg"].lstrip().startswith(("<?xml", "<svg"))
    assert all(p["score_type"] in {"calibrated_probability", "model_score"} for p in j["predictions"])
    assert j["evidence"]["labels_included"] is False
    assert "SMILES" not in json.dumps(client.get("/health/ready").json())


def test_explanation(client):
    r = client.post("/api/v1/predictions", json={"smiles": "CCCC(=O)OCC", "explain": "fruity"}).json()
    assert r["explanation"] is None or "available" in r["explanation"]


@pytest.mark.parametrize("body,status,code", [
    ({"smiles": "C(C"}, 422, "invalid_smiles"),
    ({"smiles": "CCO.O"}, 422, "unsupported_mixture"),
    ({"smiles": ""}, 422, "empty_input"),
    ({"smiles": "C" * 500}, 413, "input_too_large"),
    ({"nope": 1}, 422, "invalid_request"),
])
def test_typed_errors_never_fabricate(client, body, status, code):
    r = client.post("/api/v1/predictions", json=body)
    assert r.status_code == status and r.json()["error"]["code"] == code and r.json()["request_id"]


def test_tampered_bundle_fails_readiness(release, monkeypatch, tmp_path):
    import shutil
    from fastapi.testclient import TestClient
    from services.api.app import main

    path, _ = release
    bad = tmp_path / "bad"
    shutil.copytree(path, bad)
    (bad / "labels.json").write_text("[]")
    monkeypatch.setenv("AROMASCOPE_RELEASE_DIR", str(bad))
    with TestClient(main.app) as c:
        assert c.get("/health/ready").status_code == 503
        assert c.post("/api/v1/predictions", json={"smiles": "CCO"}).status_code == 503


def test_odor_map(client):
    m = client.get("/api/v1/map").json()
    assert m["points"] and "descriptors" in m
