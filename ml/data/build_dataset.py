"""Build the processed dataset and dataset ledger from the Leffingwell data in Pyrfume.

Usage (from the repo root):
    python -m ml.data.build_dataset --fetch

Raw source files are never modified. Everything derived is written to data/processed/.
Missing descriptors are *unknown*, not negative; the training stage documents how it treats them.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import subprocess
from pathlib import Path

import pandas as pd

PYRFUME_URL = "https://github.com/pyrfume/pyrfume-data.git"
SOURCE_DIR = "leffingwell"


def fetch(dest: Path) -> None:
    if (dest / SOURCE_DIR / "behavior.csv").exists():
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not (dest / ".git").exists():
        subprocess.run(
            ["git", "clone", "--depth", "1", "--filter=blob:none", "--sparse", PYRFUME_URL, str(dest)], check=True
        )
    subprocess.run(["git", "-C", str(dest), "sparse-checkout", "set", SOURCE_DIR], check=True)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def load_leffingwell(root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (molecules[id, smiles, cas, name], labels[id x descriptors] as 0/1)."""
    d = root / SOURCE_DIR
    beh = pd.read_csv(d / "behavior.csv", index_col=0)
    mol = pd.read_csv(d / "molecules.csv", index_col=0)
    stim = pd.read_csv(d / "stimuli.csv", index_col=0)
    common = beh.index.intersection(mol.index)
    if len(common) == 0:
        raise RuntimeError("behavior.csv and molecules.csv share no ids; the Pyrfume layout may have changed")
    beh = beh.loc[common]
    mols = pd.DataFrame(
        {
            "src_id": common.astype(str),
            "raw_smiles": mol.loc[common, "IsomericSMILES"].values,
            "name": mol.loc[common, "name"].values,
            "cas": stim["cas"].reindex(common).values if "cas" in stim.columns else None,
        }
    )
    labels = (beh.fillna(0) > 0).astype(int)
    labels.index = mols["src_id"]
    return mols, labels


def git_commit(root: Path) -> str:
    try:
        return subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def build(mols: pd.DataFrame, labels: pd.DataFrame, outdir: Path, source: dict) -> dict:
    """Standardize, deduplicate and write processed tables plus the ledger."""
    from aromascope import chemistry as chem  # imported lazily so loaders are testable without RDKit

    outdir.mkdir(parents=True, exist_ok=True)
    rows, failures = [], []
    for _, r in mols.iterrows():
        try:
            s = chem.standardize(r["raw_smiles"])
        except chem.ChemistryError as exc:
            failures.append({"src_id": r["src_id"], "raw_smiles": r["raw_smiles"], "code": exc.code})
            continue
        scaf = chem.murcko_scaffold(s.mol)
        rows.append(
            {
                "src_id": r["src_id"],
                "canonical_smiles": s.canonical_smiles,
                "flat_smiles": s.flat_smiles,
                "scaffold": scaf,
                # acyclic molecules have no scaffold; group by flat structure so stereoisomers stay together
                "group": scaf if scaf else "acyclic:" + s.flat_smiles,
                "cas": r["cas"],
                "name": r["name"],
                "raw_smiles": r["raw_smiles"],
                "standardization_notes": "; ".join(s.changes),
            }
        )
    std = pd.DataFrame(rows)

    # Resolve duplicates on isomeric canonical SMILES: union the labels (a descriptor reported
    # by any duplicate record is kept), and count what was merged.
    lab = labels.loc[std["src_id"]].reset_index(drop=True)
    lab["canonical_smiles"] = std["canonical_smiles"].values
    merged_labels = lab.groupby("canonical_smiles", sort=False).max()
    first = std.drop_duplicates("canonical_smiles", keep="first").set_index("canonical_smiles")
    n_dupes = len(std) - len(first)
    first = first.loc[merged_labels.index].reset_index()
    first.insert(0, "mol_id", [f"M{i:05d}" for i in range(len(first))])
    merged_labels = merged_labels.reset_index(drop=True)
    merged_labels.insert(0, "mol_id", first["mol_id"].values)

    first.to_csv(outdir / "molecules.csv", index=False)
    merged_labels.to_csv(outdir / "labels.csv", index=False)

    n_pos = merged_labels.drop(columns="mol_id").sum()
    ledger = {
        "built_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "policy_version": chem.POLICY_VERSION,
        "sources": [source],
        "chemistry_coverage": {
            "records_in": int(len(mols)),
            "parse_or_policy_failures": int(len(failures)),
            "failure_codes": pd.Series([f["code"] for f in failures], dtype=str).value_counts().to_dict(),
            "duplicate_records_merged": int(n_dupes),
            "molecules_out": int(len(first)),
            "stereo_unspecified_note": "Stereochemistry is preserved in canonical SMILES; model features ignore it.",
        },
        "label_handling": {
            "descriptors": int(merged_labels.shape[1] - 1),
            "raw_vs_canonical": "Descriptor vocabulary is used as published (no terms merged).",
            "missingness": "Source lists positive descriptors only. An unlisted descriptor is UNKNOWN; training treats it as negative "
            "(positive-unlabeled setting) and the model card states this limitation.",
            "aggregation": "No rater-level data is available in this source; there is no dispersion or rater count.",
            "min_positives": int(n_pos.min()),
            "max_positives": int(n_pos.max()),
        },
    }
    (outdir / "ledger.json").write_text(json.dumps(ledger, indent=2))
    if failures:
        pd.DataFrame(failures).to_csv(outdir / "rejected.csv", index=False)
    return ledger


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pyrfume-dir", default="data/raw/pyrfume-data")
    ap.add_argument("--out", default="data/processed")
    ap.add_argument("--fetch", action="store_true", help="sparse-clone Pyrfume if not present")
    ap.add_argument("--config", default="ml/config.json")
    ap.add_argument("--load-only", action="store_true", help="only load and report shapes (no RDKit needed)")
    args = ap.parse_args()

    root = Path(args.pyrfume_dir)
    if args.fetch:
        fetch(root)
    mols, labels = load_leffingwell(root)
    print(f"Loaded {len(mols)} molecules, {labels.shape[1]} descriptors")
    if args.load_only:
        return

    cfg = json.loads(Path(args.config).read_text())
    d = root / SOURCE_DIR
    source = {
        "name": "Leffingwell Odor Dataset (via Pyrfume)",
        "doi": "10.5281/zenodo.4085098",
        "url": "https://github.com/pyrfume/pyrfume-data/tree/main/leffingwell",
        "retrieved": dt.date.today().isoformat(),
        "pyrfume_commit": git_commit(root),
        "maintainer": "Pyrfume Project; original data John Leffingwell, cleaned by Sanchez-Lengeling et al.",
        "files": {f: sha256(d / f) for f in ["behavior.csv", "molecules.csv", "stimuli.csv"]},
        "rights": {
            "pyrfume_repository_license": "MIT",
            "dataset_terms": "The dataset manifest says use restrictions are in a LICENSE file from Leffingwell/Google; that file was "
            "not found in the retrieved checkout. UNRESOLVED: review before redistributing labels.",
            "redistribution_approved": bool(cfg.get("redistribution_approved", False)),
        },
        "task": "Multi-label descriptor assignment (binary). Reference population: expert-compiled flavor and fragrance descriptors, not a rater panel.",
        "attribution": "Odor labels: Leffingwell dataset, cleaned by Sanchez-Lengeling et al., distributed by the Pyrfume project.",
    }
    ledger = build(mols, labels, Path(args.out), source)
    print(json.dumps(ledger["chemistry_coverage"], indent=2))


if __name__ == "__main__":
    main()
