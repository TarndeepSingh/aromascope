"""Tests that need neither RDKit nor FastAPI."""
import json
import numpy as np
import pytest

from ml.evaluate import calibration, domain, metrics, splits


def test_scaffold_split_has_no_group_leak_and_is_deterministic():
    groups = [f"s{i % 40}" for i in range(500)]
    a = splits.scaffold_split(groups, seed=3)
    assert splits.check_no_group_leak(groups, a) == 0
    assert (a == splits.scaffold_split(groups, seed=3)).all()
    assert {"train", "val", "test"} <= set(a)


def test_metrics_skip_labels_without_positives():
    Y = np.array([[1, 0], [0, 0], [1, 0], [0, 0]])
    P = np.array([[0.9, 0.1], [0.1, 0.2], [0.8, 0.3], [0.2, 0.1]])
    ap = metrics.per_label_ap(Y, P)
    assert ap[0] == pytest.approx(1.0) and np.isnan(ap[1])
    assert metrics.macro_ap(Y, P) == pytest.approx(1.0)


def test_clean_replaces_nan_so_json_is_valid():
    out = metrics.clean({"a": float("nan"), "b": [np.float64("inf"), 1.5], "c": np.int64(3)})
    assert out == {"a": None, "b": [None, 1.5], "c": 3}
    json.dumps(out, allow_nan=False)


def test_reliability_tiers():
    assert metrics.reliability_tier(5, 10) == "strong"
    assert metrics.reliability_tier(2.5, 10) == "moderate"
    assert metrics.reliability_tier(1.2, 10) == "weak"
    assert metrics.reliability_tier(9, 2) == "unrated"


def test_platt_only_for_labels_with_enough_positives():
    rng = np.random.RandomState(0)
    Y = (rng.rand(200, 2) < [0.3, 0.02]).astype(int)
    P = np.clip(Y * 0.3 + rng.rand(200, 2) * 0.6, 0, 1)
    cal = calibration.fit_platt(P, Y, ["common", "rare"], min_pos=10)
    assert "common" in cal and "rare" not in cal


def test_max_tanimoto_identical_rows_is_one():
    bits = (np.random.RandomState(1).rand(20, 64) < 0.3).astype(np.uint8)
    assert domain.max_tanimoto(bits[:5], bits).min() == pytest.approx(1.0)


def test_sha256_file_and_manifest_mismatch(tmp_path):
    pytest.importorskip("rdkit")
    from aromascope import bundle

    f = tmp_path / "model.joblib"
    f.write_bytes(b"abc")
    assert bundle.sha256_file(f) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    with pytest.raises(bundle.BundleError):
        bundle.verify(tmp_path)  # no manifest
