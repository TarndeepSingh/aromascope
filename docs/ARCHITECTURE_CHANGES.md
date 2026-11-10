# What changed from the architecture document, and why

| Area | Document | Built | Reason |
|---|---|---|---|
| Web client | Next.js + TypeScript | Dependency-free static client (ES modules, no build step) | No npm toolchain to break; deploys to any static host or from the same container; every screen was exercised in a real browser. The UI scope (Analyze, Compare, Odor Map, Model & Data, all states, accessibility) is unchanged. |
| Deployment shape | Compose: web + API (+ Postgres) | One container serves API + client; Postgres omitted | Document says persistence is optional; history is off by default, so there is nothing to store. |
| Data decision | Open | Leffingwell via Pyrfume, binary multi-label | Smallest clean, openly hosted descriptor set; fits the "categorical" target path with PR-AUC metrics. DREAM (continuous ratings) is a later extension. |
| Missing labels | "Unknown, not negative" | Unknown in the ledger, treated as negative for training (PU) | Source lists positives only; stated everywhere it matters (model card, UI wording: "listed in the reference data"). |
| Calibration | Only when evaluated | Per-label Platt scaling with a held-out gate; otherwise raw scores are labelled "model score" | Implements the rule "never turn an arbitrary score into a percentage". |
| Rights | Per-source ledger | Label redistribution flag, default off | Rights for the label file are unresolved. |
| Explanations | "Supported explanations" | Atom-level glow for the linear model only | Uses model coefficients; marked as associations, not causal. |
| GNN | Gated comparison | Not built | Gate requires evidence the baseline is insufficient. |
| MLflow | Local tracking | Optional `--mlflow` flag | Not required to reproduce a release. |
| Extra | n/a | `tools/mock_api.py` | Lets the client be developed and demoed without RDKit; the client shows a "Demo mode" banner against it. |
