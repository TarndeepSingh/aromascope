import { h, clear, icon, fmt } from "./dom.js";
import { getModel } from "./api.js";

export async function renderModel(root) {
  clear(root).append(h("div", { class: "hero" }, h("div", { class: "tag" }, h("span", { class: "eyebrow" }, "Model & Data")),
    h("h1", {}, "What this model is, and ", h("em", {}, "isn't"), "."),
    h("p", { class: "lede" }, "Provenance, split, held-out results and limits for the release currently being served. Every number here comes from the release's own evaluation report.")),
    h("p", { class: "muted", role: "status" }, "Loading…"));
  let d;
  try { d = await getModel(); } catch (e) {
    root.querySelector("[role=status]").replaceWith(h("div", { class: "banner caution", role: "alert" }, icon("alert"), h("div", {}, h("strong", {}, "Model information unavailable"), e.message, " ", e.hint || "")));
    return;
  }
  root.querySelector("[role=status]").remove();
  const m = d.model, ev = d.evaluation, t = ev.test, src = d.ledger.sources[0], cal = ev.calibration, dm = ev.domain, cov = d.ledger.chemistry_coverage;
  const stat = (v, l) => h("div", { class: "stat" }, h("b", {}, v), h("span", {}, l));
  const bar = (label, v, lo, hi, color) => h("div", { class: "cmpbar" }, h("span", {}, label),
    h("div", { class: "track", role: "img", "aria-label": `${label}: ${v.toFixed(3)}` }, h("i", { style: `width:${Math.min(100, v * 100)}%;background:${color}` }),
      lo != null ? h("span", { class: "ci", style: `left:${lo * 100}%;width:${(hi - lo) * 100}%` }) : null), h("b", {}, v.toFixed(3)));

  const rows = d.labels.filter((l) => l.trainable).map((l) => ({ ...l }));
  let sortKey = "test_ap", dir = -1;
  const tbody = h("tbody");
  const cols = [["name", "Descriptor"], ["train_support", "Train n"], ["test_support", "Test n"], ["test_ap", "Held-out AP"], ["lift_over_prevalence", "Lift vs chance"], ["reliability", "Reliability"]];
  function fill() {
    rows.sort((a, b) => { const x = a[sortKey], y = b[sortKey]; if (x == null) return 1; if (y == null) return -1; return (x > y ? 1 : x < y ? -1 : 0) * dir; });
    clear(tbody).append(...rows.map((r) => h("tr", {}, h("td", {}, r.name), h("td", {}, String(r.train_support)), h("td", {}, String(r.test_support)),
      h("td", {}, r.test_ap != null ? [h("span", { class: "cellbar", style: `width:${Math.round(r.test_ap * 80)}px` }), r.test_ap.toFixed(2)] : "n/a"), h("td", {}, r.lift_over_prevalence != null ? r.lift_over_prevalence.toFixed(1) + "×" : "n/a"), h("td", {}, r.reliability))));
  }
  const thead = h("thead", {}, h("tr", {}, ...cols.map(([k, l]) => h("th", { "aria-sort": "none" }, h("button", { type: "button", onclick: (e) => { dir = sortKey === k ? -dir : -1; sortKey = k; fill(); } }, l)))));
  fill();

  root.append(
    h("div", { class: "grid" },
      h("section", { class: "glass" }, h("header", {}, h("h2", {}, "Release"), h("span", { class: "eyebrow" }, `${m.id} v${m.version}`)),
        h("div", { class: "stat-row" }, stat(m.kind === "logreg" ? "Logistic regression" : "Random forest", "model family"), stat(String(d.task.descriptors), "descriptors"), stat(`${m.split.train} / ${m.split.val} / ${m.split.test}`, "train / val / test molecules"), stat(m.split.method, "split method")),
        h("p", { class: "small muted" }, d.task.description, " Reference population: ", d.task.reference_population),
        h("details", {}, h("summary", {}, "Build and integrity"), h("ul", { class: "plain" },
          h("li", {}, `Created ${m.created_at}; code revision `, h("span", { class: "mono" }, String(m.code_revision).slice(0, 10))),
          h("li", {}, `Featurizer ${m.featurizer_version}; chemistry policy ${m.policy_version}`),
          h("li", {}, "SHA-256 checksums verified at startup:"), ...Object.entries(m.checksums).map(([f, hs]) => h("li", {}, f, " ", h("span", { class: "hash" }, hs.slice(0, 16) + "…")))))),
      h("section", { class: "glass" }, h("header", {}, h("h2", {}, "Held-out performance"), h("span", { class: "eyebrow" }, `frozen test set · ${t.n_molecules} molecules`)),
        bar("This model (macro AP)", t.macro_ap, t.macro_ap_ci95[0], t.macro_ap_ci95[1], "linear-gradient(90deg,#42d7c5,#9b8cff)"),
        bar("Prevalence baseline", t.baseline_macro_ap, null, null, "#6b7a90"),
        h("p", { class: "small muted" }, `Macro average precision averages per-descriptor results so rare descriptors count equally. Whisker: 95% bootstrap interval. Micro AP ${t.micro_ap.toFixed(3)}; top-3 hit rate ${t.top3_hit_rate.toFixed(3)}.`),
        ev.random_split_comparison ? h("p", { class: "small" }, "Why scaffold grouping matters: on a random split the same model scores ", h("b", {}, ev.random_split_comparison.test_macro_ap_random_split.toFixed(3)), " versus ", h("b", {}, ev.random_split_comparison.test_macro_ap_scaffold_split_raw.toFixed(3)), " when close analogs cannot leak. Only the second number is reported as generalization.") : null,
        h("div", { class: "stat-row sec" }, ...Object.entries(ev.models).map(([k, v]) => stat(fmt(v.val_macro_ap, 3), `${k.replace("_", " ")} · validation macro AP${v.promoted ? " (promoted)" : ""}`)))),
      h("section", { class: "glass" }, h("header", {}, h("h2", {}, "Per-descriptor results"), h("span", { class: "eyebrow" }, "click a heading to sort")),
        h("div", { class: "scroll" }, h("table", {}, thead, tbody)), h("p", { class: "hint" }, ev.reliability_rule)),
      h("div", { class: "grid cols-2" },
        h("section", { class: "glass" }, h("header", {}, h("h2", {}, "Calibration")),
          h("p", { class: "small" }, cal.meaning), h("div", { class: "stat-row" }, stat(`${cal.labels_calibrated}`, "descriptors calibrated"), stat(`${cal.ece_raw.toFixed(3)} → ${cal.ece_calibrated.toFixed(3)}`, "ECE raw → calibrated"), stat(`${cal.brier_raw.toFixed(3)} → ${cal.brier_calibrated.toFixed(3)}`, "Brier raw → calibrated")),
          h("p", { class: "small muted" }, cal.gate_passed ? "The calibration gate passed on held-out data, so probabilities are shown for calibrated descriptors." : "The gate did not pass on held-out data, so only uncalibrated model scores are shown.")),
        h("section", { class: "glass" }, h("header", {}, h("h2", {}, "Applicability domain")),
          h("p", { class: "small" }, `Well covered at similarity ≥ ${dm.tau_in_domain}; the model abstains below ${dm.tau_out_of_domain} or outside the training molecular-weight range. On the test set ${(dm.share_test_in_domain * 100).toFixed(0)}% of molecules fall in the well-covered zone.`),
          h("table", {}, h("thead", {}, h("tr", {}, h("th", {}, "Similarity"), h("th", {}, "Molecules"), h("th", {}, "Mean AP"))),
            h("tbody", {}, ...dm.coverage_table_test.map((r) => h("tr", {}, h("td", {}, `${r.similarity_range[0].toFixed(1)}–${r.similarity_range[1].toFixed(1)}`), h("td", {}, String(r.n_molecules)), h("td", {}, r.mean_per_molecule_ap == null ? "n/a" : r.mean_per_molecule_ap.toFixed(2))))))),
      ),
      h("section", { class: "glass" }, h("header", {}, h("h2", {}, "Data provenance & rights"), h("span", { class: "eyebrow" }, "dataset ledger")),
        h("ul", { class: "plain", style: "margin-top:0" },
          h("li", {}, h("b", {}, "Source: "), src.name, " · DOI ", src.doi, " · Pyrfume commit ", h("span", { class: "mono" }, (src.pyrfume_commit || "").slice(0, 10))),
          h("li", {}, h("b", {}, "Rights: "), src.rights.dataset_terms, ` Label redistribution approved: ${src.rights.redistribution_approved ? "yes" : "no"}.`),
          h("li", {}, h("b", {}, "Coverage: "), `${cov.records_in} records in, ${cov.parse_or_policy_failures} rejected by policy, ${cov.duplicate_records_merged} duplicates merged, ${cov.molecules_out} molecules used.`),
          h("li", {}, h("b", {}, "Labels: "), d.ledger.label_handling.missingness),
          h("li", {}, h("b", {}, "Raters: "), d.ledger.label_handling.aggregation))),
      h("section", { class: "glass" }, h("header", {}, h("h2", {}, "Model card")), h("details", {}, h("summary", {}, "Read the full model card"), h("pre", { class: "card", tabindex: "0" }, d.model_card_markdown))),
      h("section", { class: "glass" }, h("header", {}, h("h2", {}, "Limitations")), h("ul", { class: "plain", style: "margin-top:0" }, ...d.limitations.map((l) => h("li", {}, l))))));
}
