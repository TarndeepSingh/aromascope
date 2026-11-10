# AromaScope model card, release 0.1.0

**Task.** Multi-label prediction of odor descriptors (binary) from one molecular structure (SMILES).
**Output.** A ranked list of descriptors. Values are calibrated probabilities that the descriptor is listed for a molecule like this in the reference dataset.
These are model-based predictions, **not** sensory measurements and **not** a room-air detector.

## Model
- Family: `rf` (random forest), baseline-first selection.
- Features: Morgan fingerprint (radius 2, 2048 bits) + 8 standardized physicochemical descriptors. Stereochemistry is **not** used by the features.
- Selection rule: validation macro average precision; the random forest is promoted only if it beats the linear model by 2% and fits the size budget.

## Data
- Source: Leffingwell Odor Dataset (via Pyrfume), DOI 10.5281/zenodo.4085098; Pyrfume commit `8054ea98ed`.
- 3522 molecules after standardization. Scaffold-grouped split: train 2818, validation 352, test 352.
- Label handling: the source lists positive descriptors only; unlisted descriptors are unknown but are treated as negative in training (positive-unlabeled setting). Reported precision is therefore a conservative estimate for rare descriptors and probabilities describe *dataset listing*, not perception.
- Rights: **The dataset manifest says use restrictions are in a LICENSE file from Leffingwell/Google; that file was not found in the retrieved checkout. UNRESOLVED: review before redistributing labels.**  Redistribution of labels approved: `False`.

## Held-out results (frozen test set, 352 molecules)
- Macro average precision: **0.311** (95% bootstrap CI 0.320 to 0.385) versus **0.035** for the prevalence baseline.
- Micro average precision: 0.417. Top-3 hit rate: 0.793.
- Descriptors with strong held-out performance: alcoholic, alliaceous, almond, apple, balsamic, banana, beefy, brandy, burnt, buttery, caramellic, cheesy, citrus, coconut, coffee, cognac, creamy, dairy, ethereal, fatty, fermented, garlic, grape, jasmine, ketonic, meaty, melon, mint, mushroom, nutty, odorless, oily, onion, pear, pineapple, pungent, roasted, rose, rum, savory, solvent, sour, spicy, sulfurous, vegetable, waxy, winey, woody.
- Descriptors with weak performance (treat with caution): fruity, musty, sweet.

## Calibration
- per-descriptor Platt scaling on logit(score), fit on validation; 32 descriptors calibrated. Test Brier 0.0245 to 0.0239; ECE 0.0095 to 0.0042. Gate passed: True.

## Applicability domain
- Nearest-neighbour Tanimoto similarity to training molecules. Well covered at >= 0.5; below 0.3 or outside the training molecular-weight range the model abstains.

## Limitations
- Predictions apply to this reference vocabulary and source, not to every person's perception.
- Structure alone omits purity, concentration, conditions, mixtures and individual differences.
- Similarity does not guarantee sensory similarity; the model can be wrong.
- Substructure highlights are model associations, not causal mechanisms.
- No air sampling is involved. Identifying an unknown airborne odor is out of scope.
