"""Release bundle: writing, checksum verification and loading.

A release is one immutable folder holding everything needed to serve and audit a model:
weights, feature scaler, label ontology, calibration, domain thresholds, reference index,
evaluation report, model card, dataset ledger and a manifest with SHA-256 checksums.
The API refuses to start serving unless the manifest validates.
"""
from __future__ import annotations

import hashlib
import json
import platform
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

MANIFEST = "manifest.json"
REQUIRED_FILES = [
    "model.joblib",
    "features.npz",
    "reference.npz",
    "reference_smiles.json",
    "labels.json",
    "calibration.json",
    "domain.json",
    "map.json",
    "projection.npz",
    "evaluation_report.json",
    "model_card.md",
    "ledger.json",
]
OPTIONAL_FILES = ["reference_labels.json"]


class BundleError(RuntimeError):
    """The release bundle is missing, tampered with, or incompatible with this runtime."""


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def runtime_versions() -> dict[str, str]:
    import sklearn

    versions = {"python": platform.python_version(), "numpy": np.__version__, "scikit_learn": sklearn.__version__}
    try:
        from rdkit import rdBase

        versions["rdkit"] = rdBase.rdkitVersion
    except Exception:  # pragma: no cover
        versions["rdkit"] = "unavailable"
    return versions


def write_manifest(release_dir: Path, info: dict[str, Any]) -> dict[str, Any]:
    files = {}
    for name in REQUIRED_FILES + OPTIONAL_FILES:
        p = release_dir / name
        if p.exists():
            files[name] = sha256_file(p)
    missing = [n for n in REQUIRED_FILES if n not in files]
    if missing:
        raise BundleError(f"Cannot write manifest; missing files: {missing}")
    manifest = {**info, "files": files, "runtime": runtime_versions()}
    (release_dir / MANIFEST).write_text(json.dumps(manifest, indent=2, sort_keys=True))
    return manifest


def verify(release_dir: Path) -> tuple[dict[str, Any], list[str]]:
    """Return (manifest, warnings). Raises BundleError on any integrity or compatibility failure."""
    release_dir = Path(release_dir)
    mpath = release_dir / MANIFEST
    if not mpath.exists():
        raise BundleError(f"No {MANIFEST} in {release_dir}")
    try:
        manifest = json.loads(mpath.read_text())
    except json.JSONDecodeError as exc:
        raise BundleError(f"Manifest is not valid JSON: {exc}") from exc

    files = manifest.get("files", {})
    for name in REQUIRED_FILES:
        if name not in files:
            raise BundleError(f"Manifest does not list required file {name}")
    for name, expected in files.items():
        p = release_dir / name
        if not p.exists():
            raise BundleError(f"Listed file is missing: {name}")
        if sha256_file(p) != expected:
            raise BundleError(f"Checksum mismatch for {name}; the bundle was modified or corrupted")

    warnings: list[str] = []
    current = runtime_versions()
    built = manifest.get("runtime", {})
    # Pickled estimators are only safe to load under the same scikit-learn feature release.
    if built.get("scikit_learn", "").split(".")[:2] != current["scikit_learn"].split(".")[:2]:
        raise BundleError(
            f"scikit-learn {built.get('scikit_learn')} built this bundle; runtime has {current['scikit_learn']}. "
            "Install the pinned requirements.txt."
        )
    if built.get("rdkit") and built.get("rdkit") != current.get("rdkit"):
        warnings.append(f"RDKit version differs (bundle {built.get('rdkit')}, runtime {current.get('rdkit')}).")

    from .chemistry import FEATURIZER_VERSION, POLICY_VERSION

    if manifest.get("featurizer_version") != FEATURIZER_VERSION or manifest.get("policy_version") != POLICY_VERSION:
        raise BundleError("Bundle was built with a different featurizer or chemistry policy than this code.")
    return manifest, warnings


@dataclass
class Release:
    path: Path
    manifest: dict[str, Any]
    warnings: list[str]
    model: Any
    model_kind: str
    trainable_idx: np.ndarray  # positions in the full vocabulary that the model scores
    labels: list[dict[str, Any]]
    calibration: dict[str, dict[str, float]]
    feat_mean: np.ndarray
    feat_std: np.ndarray
    ref_bits: np.ndarray  # float32 [N, 2048]
    ref_bit_counts: np.ndarray
    ref_smiles: list[str]
    ref_labels: dict[str, list[str]] | None
    domain: dict[str, Any]
    map_data: dict[str, Any]
    proj_mean: np.ndarray
    proj_components: np.ndarray
    evaluation: dict[str, Any]
    model_card: str
    ledger: dict[str, Any]
    label_names: list[str] = field(default_factory=list)


def load_release(release_dir: Path | str) -> Release:
    import joblib

    release_dir = Path(release_dir)
    manifest, warnings = verify(release_dir)

    def jload(name: str) -> Any:
        return json.loads((release_dir / name).read_text())

    payload = joblib.load(release_dir / "model.joblib")
    feats = np.load(release_dir / "features.npz")
    ref = np.load(release_dir / "reference.npz")
    proj = np.load(release_dir / "projection.npz")
    n_bits = int(ref["n_bits"])
    bits = np.unpackbits(ref["packed"], axis=1)[:, :n_bits].astype(np.float32)

    ref_labels = None
    if (release_dir / "reference_labels.json").exists() and manifest.get("redistribution_approved"):
        ref_labels = jload("reference_labels.json")

    labels = jload("labels.json")
    return Release(
        path=release_dir,
        manifest=manifest,
        warnings=warnings,
        model=payload["estimator"],
        model_kind=payload["kind"],
        trainable_idx=np.asarray(payload["trainable_idx"], dtype=int),
        labels=labels,
        calibration=jload("calibration.json"),
        feat_mean=feats["mean"],
        feat_std=feats["std"],
        ref_bits=bits,
        ref_bit_counts=bits.sum(axis=1),
        ref_smiles=jload("reference_smiles.json"),
        ref_labels=ref_labels,
        domain=jload("domain.json"),
        map_data=jload("map.json"),
        proj_mean=proj["mean"],
        proj_components=proj["components"],
        evaluation=jload("evaluation_report.json"),
        model_card=(release_dir / "model_card.md").read_text(),
        ledger=jload("ledger.json"),
        label_names=[l["name"] for l in labels],
    )


def resolve_release_dir(root: Path | str | None = None) -> Path:
    """Pick the release folder: AROMASCOPE_RELEASE_DIR, else artifacts/releases/CURRENT."""
    import os

    explicit = os.environ.get("AROMASCOPE_RELEASE_DIR")
    if explicit:
        return Path(explicit)
    base = Path(root or os.environ.get("AROMASCOPE_ARTIFACTS", "artifacts/releases"))
    current = base / "CURRENT"
    if not current.exists():
        raise BundleError(f"No release found: {current} does not exist. Run `make train` first.")
    return base / current.read_text().strip()
