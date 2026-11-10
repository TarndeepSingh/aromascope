"""Chemistry policy, featurization and depiction.

This module is the single implementation shared by offline training and online serving,
so the two cannot drift apart. Changing anything here that alters features or the
accepted-structure policy requires bumping POLICY_VERSION / FEATURIZER_VERSION, which
invalidates older release bundles on purpose.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from rdkit import Chem
from rdkit.Chem import Crippen, Descriptors, Lipinski, rdFingerprintGenerator, rdMolDescriptors
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem.Scaffolds import MurckoScaffold

POLICY_VERSION = "chem-policy-1"
FEATURIZER_VERSION = "morgan2x2048+physchem8-v1"

MAX_INPUT_CHARS = 400
MIN_HEAVY_ATOMS = 2
MAX_HEAVY_ATOMS = 60
# H, C, N, O, F, P, S, Cl, Br, I
ALLOWED_ATOMIC_NUMBERS = {1, 6, 7, 8, 9, 15, 16, 17, 35, 53}
N_BITS = 2048
MORGAN_RADIUS = 2
PHYSCHEM_NAMES = [
    "mol_weight",
    "crippen_logp",
    "tpsa",
    "h_donors",
    "h_acceptors",
    "rotatable_bonds",
    "ring_count",
    "heavy_atoms",
]


class ChemistryError(ValueError):
    """A typed, user-explainable input problem. Never produces a fabricated prediction."""

    def __init__(self, code: str, message: str, hint: str = ""):
        super().__init__(message)
        self.code = code
        self.message = message
        self.hint = hint


@dataclass
class StandardizedMolecule:
    mol: Chem.Mol
    input_smiles: str
    canonical_smiles: str
    flat_smiles: str  # canonical without stereo; used to keep stereoisomers in one split group
    formula: str
    mol_weight: float
    heavy_atoms: int
    unspecified_stereo: bool
    changes: list[str] = field(default_factory=list)


def standardize(raw: str | None) -> StandardizedMolecule:
    """Parse and standardize one molecule under the versioned policy.

    Policy (documented in docs/dataset_card.md):
    - exactly one connected component; mixtures and salts are rejected, never stripped
    - formal charges are neutralized where chemically possible, and this is reported
    - stereochemistry is preserved in the canonical SMILES
    - tautomers are NOT canonicalized
    """
    if raw is None or not str(raw).strip():
        raise ChemistryError("empty_input", "Enter a SMILES string.", "Try CCO (ethanol).")
    text = str(raw).strip()
    if len(text) > MAX_INPUT_CHARS:
        raise ChemistryError(
            "input_too_large",
            f"Input is longer than {MAX_INPUT_CHARS} characters.",
            "AromaScope handles single small molecules, not polymers or long records.",
        )
    if any(ch.isspace() for ch in text):
        raise ChemistryError(
            "invalid_smiles",
            "SMILES cannot contain spaces.",
            "Remove spaces or any trailing name/identifier after the structure.",
        )

    mol = Chem.MolFromSmiles(text)
    if mol is None:
        raise ChemistryError(
            "invalid_smiles",
            "This is not a valid SMILES string.",
            "Check bracket pairs, ring-closure digits and aromatic atom valences.",
        )

    if len(Chem.GetMolFrags(mol)) > 1:
        raise ChemistryError(
            "unsupported_mixture",
            "Multiple components (a mixture or salt) were found.",
            "AromaScope predicts one molecule at a time. Submit the parent molecule only.",
        )

    changes: list[str] = []
    neutral = rdMolStandardize.Uncharger().uncharge(mol)
    if Chem.MolToSmiles(neutral) != Chem.MolToSmiles(mol):
        changes.append("Formal charges were neutralized.")
        mol = neutral
    Chem.SanitizeMol(mol)

    heavy = mol.GetNumHeavyAtoms()
    if heavy < MIN_HEAVY_ATOMS:
        raise ChemistryError(
            "unsupported_structure",
            "The molecule is too small to profile.",
            f"At least {MIN_HEAVY_ATOMS} heavy atoms are needed.",
        )
    if heavy > MAX_HEAVY_ATOMS:
        raise ChemistryError(
            "unsupported_structure",
            f"The molecule has {heavy} heavy atoms; the limit is {MAX_HEAVY_ATOMS}.",
            "The reference data covers small, volatile-range molecules.",
        )
    bad = sorted({a.GetSymbol() for a in mol.GetAtoms() if a.GetAtomicNum() not in ALLOWED_ATOMIC_NUMBERS})
    if bad:
        raise ChemistryError(
            "unsupported_structure",
            f"Unsupported element(s): {', '.join(bad)}.",
            "Supported elements: H, C, N, O, F, P, S, Cl, Br, I.",
        )
    if not any(a.GetAtomicNum() == 6 for a in mol.GetAtoms()):
        raise ChemistryError(
            "unsupported_structure", "The molecule contains no carbon.", "Only organic molecules are supported."
        )

    canonical = Chem.MolToSmiles(mol, isomericSmiles=True)
    flat = Chem.MolToSmiles(mol, isomericSmiles=False)
    centers = Chem.FindMolChiralCenters(mol, includeUnassigned=True, useLegacyImplementation=False)
    return StandardizedMolecule(
        mol=mol,
        input_smiles=text,
        canonical_smiles=canonical,
        flat_smiles=flat,
        formula=rdMolDescriptors.CalcMolFormula(mol),
        mol_weight=float(Descriptors.MolWt(mol)),
        heavy_atoms=heavy,
        unspecified_stereo=any(label == "?" for _, label in centers),
        changes=changes,
    )


def murcko_scaffold(mol: Chem.Mol) -> str:
    """Bemis-Murcko scaffold SMILES without stereo; empty string for acyclic molecules."""
    try:
        return MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
    except Exception:  # pragma: no cover - defensive
        return ""


def morgan_bits(mol: Chem.Mol) -> np.ndarray:
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=MORGAN_RADIUS, fpSize=N_BITS)
    return np.asarray(gen.GetFingerprintAsNumPy(mol), dtype=np.uint8)


def physchem(mol: Chem.Mol) -> np.ndarray:
    return np.array(
        [
            Descriptors.MolWt(mol),
            Crippen.MolLogP(mol),
            rdMolDescriptors.CalcTPSA(mol),
            Lipinski.NumHDonors(mol),
            Lipinski.NumHAcceptors(mol),
            Lipinski.NumRotatableBonds(mol),
            rdMolDescriptors.CalcNumRings(mol),
            mol.GetNumHeavyAtoms(),
        ],
        dtype=np.float64,
    )


def feature_vector(bits: np.ndarray, phys: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """Final model input: 2048 Morgan bits followed by 8 standardized physchem values.

    Works on a single molecule (1-D inputs) or a batch (2-D inputs).
    """
    z = (phys - mean) / std
    return np.concatenate([bits.astype(np.float32), z.astype(np.float32)], axis=-1)


def _environment_atoms(mol: Chem.Mol, atom_idx: int, radius: int) -> set[int]:
    if radius == 0:
        return {atom_idx}
    atoms = {atom_idx}
    for bond_idx in Chem.FindAtomEnvironmentOfRadiusN(mol, radius, atom_idx):
        bond = mol.GetBondWithIdx(bond_idx)
        atoms.add(bond.GetBeginAtomIdx())
        atoms.add(bond.GetEndAtomIdx())
    return atoms


def morgan_bit_atoms(mol: Chem.Mol) -> dict[int, list[set[int]]]:
    """Map each set Morgan bit to the atom sets of the environments that set it."""
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=MORGAN_RADIUS, fpSize=N_BITS)
    out = rdFingerprintGenerator.AdditionalOutput()
    out.AllocateBitInfoMap()
    gen.GetFingerprint(mol, additionalOutput=out)
    return {
        int(bit): [_environment_atoms(mol, int(a), int(r)) for a, r in envs]
        for bit, envs in out.GetBitInfoMap().items()
    }


def render_svg(
    mol: Chem.Mol,
    size: tuple[int, int] = (420, 300),
    highlights: dict[int, float] | None = None,
) -> str:
    """Depict a molecule as SVG on a transparent background in a dark-mode palette.

    `highlights` maps atom index -> weight in [0, 1] and draws a teal glow.
    """
    drawer = rdMolDraw2D.MolDraw2DSVG(size[0], size[1])
    opts = drawer.drawOptions()
    try:
        rdMolDraw2D.SetDarkMode(opts)
    except Exception:  # pragma: no cover - older RDKit
        pass
    opts.clearBackground = False
    opts.bondLineWidth = 2
    opts.padding = 0.12
    kwargs = {}
    if highlights:
        atoms = [i for i, w in highlights.items() if w > 0.05]
        kwargs["highlightAtoms"] = atoms
        kwargs["highlightAtomColors"] = {i: (0.26, 0.84, 0.77, min(0.85, 0.25 + 0.6 * highlights[i])) for i in atoms}
        kwargs["highlightBonds"] = []
    drawer.DrawMolecule(mol, **kwargs)
    drawer.FinishDrawing()
    return drawer.GetDrawingText()
