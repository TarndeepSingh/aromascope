import pytest

pytest.importorskip("rdkit")
from aromascope import chemistry as chem  # noqa: E402


def test_canonicalization_and_stereo_preserved():
    s = chem.standardize("OC(C)CC")  # unspecified centre
    assert s.unspecified_stereo
    r = chem.standardize("C[C@H](O)CC")
    assert "@" in r.canonical_smiles and not r.unspecified_stereo


@pytest.mark.parametrize(
    "text,code",
    [
        ("", "empty_input"),
        ("C(C", "invalid_smiles"),
        ("CCO CCC", "invalid_smiles"),
        ("CCO.O", "unsupported_mixture"),
        ("[Na+].[Cl-]", "unsupported_mixture"),
        ("[Si](C)(C)C", "unsupported_structure"),
        ("C", "unsupported_structure"),
        ("C" * 401, "input_too_large"),
    ],
)
def test_rejections_are_typed(text, code):
    with pytest.raises(chem.ChemistryError) as e:
        chem.standardize(text)
    assert e.value.code == code and e.value.hint


def test_charge_neutralization_is_reported_not_silent():
    s = chem.standardize("CC(=O)[O-]")
    assert s.canonical_smiles == "CC(=O)O" and s.changes


def test_features_shape_and_determinism():
    s = chem.standardize("CCCC(=O)OCC")
    bits, phys = chem.morgan_bits(s.mol), chem.physchem(s.mol)
    assert bits.shape == (chem.N_BITS,) and phys.shape == (len(chem.PHYSCHEM_NAMES),)
    assert (bits == chem.morgan_bits(chem.standardize("CCOC(=O)CCC").mol)).all()


def test_svg_renders():
    s = chem.standardize("c1ccccc1O")
    assert chem.render_svg(s.mol).lstrip().startswith(("<?xml", "<svg"))
    assert "svg" in chem.render_svg(s.mol, highlights={0: 1.0})
