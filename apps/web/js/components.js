import { h, icon, safeSvg, fmt } from "./dom.js";
import { rose, gauge } from "./charts.js";

export const STATUS = {
  in_domain: { label: "Well covered", icon: "check" },
  limited_evidence: { label: "Limited evidence", icon: "alert" },
  out_of_domain: { label: "Out of domain", icon: "x" },
};

export function statusChip(status) {
  const m = STATUS[status] || STATUS.limited_evidence;
  return h("span", { class: `status ${status}` }, icon(m.icon), m.label);
}

export function svgBox(cls, svgText, fallback) {
  const box = h("div", { class: cls });
  const node = safeSvg(svgText);
  if (node) box.append(node); else box.append(h("p", { class: "muted small", style: "padding:14px" }, fallback || "No depiction available."));
  return box;
}

export function lensCard(res, { title = "Structure", extra } = {}) {
  const m = res.molecule;
  const lens = svgBox("lens", m.structure_svg, "Structure could not be drawn.");
  lens.setAttribute("role", "img");
  lens.setAttribute("aria-label", `Two-dimensional structure of ${m.formula}, canonical SMILES ${m.canonical_smiles}`);
  const smiles = h("span", { class: "mono" }, m.canonical_smiles);
  const copy = h("button", { class: "icon-btn", type: "button", "aria-label": "Copy canonical SMILES",
    onclick: async () => { try { await navigator.clipboard.writeText(m.canonical_smiles); copy.title = "Copied"; } catch { /* clipboard unavailable */ } } }, icon("copy"));
  return h("section", { class: "glass", "aria-label": title },
    h("header", {}, h("h2", {}, title), h("span", { class: "eyebrow" }, m.formula)),
    lens,
    h("div", { class: "facts" },
      h("div", {}, h("b", {}, fmt(m.mol_weight, 1)), "mol. weight"),
      h("div", {}, h("b", {}, String(m.heavy_atoms)), "heavy atoms"),
      h("div", {}, h("b", {}, m.unspecified_stereo ? "unspecified" : "specified / none"), "stereo")),
    h("div", { class: "copyrow" }, smiles, copy),
    ...(m.changes || []).map((c) => h("p", { class: "notice" }, "Note: " + c)),
    m.unspecified_stereo ? h("p", { class: "hint" }, "This structure has stereocentres without defined configuration. The model ignores stereochemistry either way.") : null,
    extra);
}

export function signature(res) {
  if (res.abstained || !res.predictions.length) return null;
  const top = res.predictions.slice(0, 3).map((p) => p.descriptor);
  const em = (t) => h("em", {}, t);
  return h("p", { class: "signature" }, "Predicted to read as ", em(top[0]), top[1] ? [", ", em(top[1])] : null, top[2] ? [" and ", em(top[2])] : null, ".");
}

export function rankList(res, { onExplain, explaining } = {}) {
  const ol = h("ol", { class: "rank", "aria-label": "Ranked descriptors" });
  res.predictions.forEach((p, i) => {
    const raw = p.score_type !== "calibrated_probability";
    const level = { strong: 3, moderate: 2, weak: 1, unrated: 0 }[p.reliability] ?? 0;
    const rel = h("span", { class: `rel ${p.reliability}`, title: `Held-out reliability: ${p.reliability}`, role: "img", "aria-label": `Held-out reliability ${p.reliability}` },
      ...[1, 2, 3].map((k) => h("i", { class: k <= level ? "on" : "" })));
    const val = raw ? p.value.toFixed(2) : `${(p.value * 100).toFixed(p.value < 0.1 ? 1 : 0)}%`;
    const btn = onExplain ? h("button", { class: "icon-btn", type: "button", "aria-pressed": String(explaining === p.descriptor),
      "aria-label": `Show which atoms push the model toward ${p.descriptor}`, title: "Show where on the structure", onclick: () => onExplain(p.descriptor) }, icon("eye")) : null;
    ol.append(h("li", {},
      h("span", { class: "n" }, String(i + 1)),
      h("span", {}, p.descriptor),
      h("span", { class: `bar${raw ? " raw" : ""}`, role: "presentation" }, h("i", { style: `width:${Math.max(1.5, p.value * 100)}%` })),
      h("span", { class: "row-actions" }, h("span", { class: "v" }, val), rel, btn)));
  });
  return ol;
}

export function profileCard(res, opts = {}) {
  const kinds = new Set(res.predictions.map((p) => p.score_type));
  const body = res.abstained
    ? h("div", { class: "banner danger", role: "status" }, icon("x"),
        h("div", {}, h("strong", {}, "No profile returned."), "This molecule is outside what this release was trained on, so AromaScope abstains instead of guessing. ", res.applicability.explanation))
    : h("div", { class: "pred-wrap" }, rose(res.predictions), rankList(res, opts));
  const legend = res.abstained ? null : h("div", { class: "legend" },
    kinds.has("calibrated_probability") ? h("span", {}, "Calibrated probability (listed in reference data)") : null,
    kinds.has("model_score") ? h("span", { class: "raw" }, "Model score (ranks, not a probability)") : null,
    h("span", { style: "display:inline-flex;gap:6px;align-items:center" }, h("span", { class: "rel", style: "display:inline-flex" }, h("i", { class: "on" }), h("i", { class: "on" }), h("i", {})), "held-out reliability"));
  return h("section", { class: "glass", "aria-label": "Predicted odor profile" },
    h("header", {}, h("h2", {}, "Predicted odor profile"), statusChip(res.applicability.status)),
    signature(res), body, legend);
}

export function limitedBanner(res) {
  const a = res.applicability;
  if (a.status === "in_domain") return null;
  const cls = a.status === "out_of_domain" ? "danger" : "caution";
  return h("div", { class: `banner ${cls}`, role: "status" }, icon(a.status === "out_of_domain" ? "x" : "alert"),
    h("div", {}, h("strong", {}, a.status === "out_of_domain" ? "Out of domain" : "Limited evidence"),
      a.explanation, " ", a.status === "limited_evidence" ? "Treat the ranking as a lead to check, not as a result." : ""));
}

export function reliabilityCard(res) {
  const a = res.applicability;
  const top = res.predictions[0];
  const ev = res.evidence;
  const nbSim = ev.neighbors.length ? ev.neighbors[0].similarity : 0;
  const sig = (title, ...kids) => h("div", { class: "signal" }, h("h3", {}, title), ...kids);
  return h("section", { class: "glass", "aria-label": "Reliability" },
    h("header", {}, h("h2", {}, "How far to trust this"), h("span", { class: "eyebrow" }, "four separate signals")),
    h("div", { class: "signals" },
      sig("Applicability domain", gauge(a), h("p", {}, statusChip(a.status)), h("p", { class: "small muted" }, a.caveat)),
      sig("Evidence strength",
        top ? h("p", {}, h("b", {}, top.descriptor), `: held-out reliability is `, h("b", {}, top.reliability), top.test_average_precision != null ? ` (average precision ${top.test_average_precision.toFixed(2)})` : "", ".") : h("p", { class: "muted" }, "No descriptors returned."),
        h("p", { class: "small muted" }, `Closest reference molecule: similarity ${nbSim.toFixed(2)}. ${top && top.train_support != null ? top.train_support + " training molecules carry this descriptor." : ""}`)),
      h("div", { class: "signal na" }, h("h3", {}, "Prediction uncertainty"), h("p", { class: "val" }, "No interval is shown."), h("p", { class: "small muted" }, "Intervals are only displayed when they have been validated. This release has not validated them, so none are implied.")),
      h("div", { class: "signal na" }, h("h3", {}, "Human variability"), h("p", { class: "val" }, "Not available."), h("p", { class: "small muted" }, "The source lists expert-compiled descriptors without rater-level data, so disagreement between people cannot be shown."))));
}

export function evidenceCard(res, onAnalyze) {
  const ev = res.evidence;
  const grid = h("div", { class: "neighbors" });
  ev.neighbors.forEach((n) => {
    grid.append(h("div", { class: "nb" },
      svgBox("thumb", n.svg, ""),
      h("div", { class: "sim" }, n.similarity.toFixed(2), " ", h("small", {}, "similarity")),
      h("span", { class: "mono small" }, n.smiles),
      n.labels ? h("div", { class: "tags" }, ...n.labels.slice(0, 6).map((l) => h("span", { class: "tag" }, l))) : null,
      onAnalyze ? h("button", { class: "btn ghost", type: "button", onclick: () => onAnalyze(n.smiles) }, "Analyze this") : null));
  });
  return h("section", { class: "glass", "aria-label": "Evidence" },
    h("header", {}, h("h2", {}, "Why this prediction"), h("span", { class: "eyebrow" }, "nearest reference molecules")),
    h("p", { class: "muted small", style: "margin-top:0" }, ev.method + ". Structural closeness, not proof of similar odor."),
    ev.labels_note ? h("p", { class: "notice" }, ev.labels_note) : null, grid,
    ev.attribution ? h("p", { class: "hint" }, ev.attribution) : null);
}

export function explainerCard(res) {
  const kinds = new Set(res.predictions.map((p) => p.score_type));
  return h("section", { class: "glass", "aria-label": "Score explainer" },
    h("header", {}, h("h2", {}, "Reading the numbers")),
    h("ul", { class: "plain", style: "margin-top:0" },
      kinds.has("calibrated_probability") ? h("li", {}, h("b", {}, "Percent values"), " are calibrated probabilities: among molecules scored like this, about that share are listed with the descriptor in the reference dataset. They are not the chance a person will perceive it."): null,
      kinds.has("model_score") ? h("li", {}, h("b", {}, "Hatched bars (0 to 1)"), " are model scores. They rank descriptors for this molecule but have not been calibrated, so do not read them as percentages.") : null,
      h("li", {}, h("b", {}, "Reliability pips"), " compare the model with a prevalence baseline on held-out molecules the model never saw. Three pips means at least four times better than chance for that descriptor."),
      h("li", {}, "A missing descriptor in the reference data is unknown, not a confirmed absence.")),
    h("details", {}, h("summary", {}, "Limitations of this prediction"), h("ul", { class: "plain" }, ...res.limitations.map((l) => h("li", {}, l)))));
}
