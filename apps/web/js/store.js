// Tiny shared state so views can hand a molecule to each other (e.g. Analyze -> Odor Map).
export const store = { last: null };
export const EXAMPLES = [
  ["Vanillin", "COc1cc(C=O)ccc1O"],
  ["Ethyl butyrate", "CCCC(=O)OCC"],
  ["Linalool", "CC(C)=CCCC(C)(O)C=C"],
  ["Menthol", "CC(C)[C@@H]1CC[C@@H](C)C[C@H]1O"],
  ["Limonene", "CC1=CC[C@@H](CC1)C(C)=C"],
  ["Indole", "c1ccc2[nH]ccc2c1"],
  ["Benzaldehyde", "O=Cc1ccccc1"],
];
