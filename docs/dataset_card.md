# Dataset card: Leffingwell odor data via Pyrfume

| Field | Value |
|---|---|
| Source | Leffingwell Odor Dataset, DOI 10.5281/zenodo.4085098 (John Leffingwell; cleaned by Sanchez-Lengeling et al.) |
| Distribution | Pyrfume project, `pyrfume/pyrfume-data`, folder `leffingwell` (fetched at build time, never committed here) |
| Size | 3,522 records, 113 binary descriptors (before standardization and de-duplication) |
| Population / task | Expert-compiled flavor and fragrance descriptors. **Not** a rater panel; no concentration, protocol or rater data |
| Missing labels | Only positive descriptors are listed. Unlisted = **unknown**. Training treats unlisted as negative (positive-unlabeled), which understates precision for rare descriptors |
| Rights | Pyrfume repository: MIT. The dataset manifest says use restrictions are in a `LICENSE` file from Leffingwell/Google; **that file was not present in the retrieved checkout**. Status: **unresolved**. |
| Redistribution | Off by default (`ml/config.json`). While off, the API shows neighbour structures and similarity but not their labels |

## Chemistry policy (`chem-policy-1`)
Single connected component (mixtures and salts rejected, not stripped); charges neutralized and reported; stereochemistry kept in canonical SMILES but **not** used by features; tautomers not canonicalized; heavy atoms 2-60; elements H C N O F P S Cl Br I; at least one carbon.

## Splits
Scaffold-grouped 80/10/10 (Bemis-Murcko). Acyclic molecules are grouped by flat structure so stereoisomers stay together. A random split is reported only as a secondary comparison.

## Before you publish a public demo
Read the Leffingwell terms yourself. If label redistribution is permitted, set `redistribution_approved` to `true` in `ml/config.json`, retrain, and the evidence panel will show neighbour labels. If not, leave it `false`.
