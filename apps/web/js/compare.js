import { h, clear, icon } from "./dom.js";
import { predict, precheck, ApiError } from "./api.js";
import * as C from "./components.js";

export function renderCompare(root, params) {
  clear(root);
  const mk = (id, label, val) => h("div", { style: "flex:1 1 300px" }, h("label", { for: id, class: "eyebrow", style: "display:block;margin-bottom:6px" }, label),
    h("input", { type: "text", id, autocomplete: "off", spellcheck: "false", value: val || "", placeholder: "SMILES", style: "width:100%" }));
  const a = mk("cmp-a", "Molecule A", params.get("a") || "COc1cc(C=O)ccc1O");
  const b = mk("cmp-b", "Molecule B", params.get("b") || "O=Cc1ccccc1");
  const go = h("button", { class: "btn", type: "submit" }, "Compare");
  const out = h("div", { class: "grid", "aria-live": "polite", style: "margin-top:20px" });
  const form = h("form", { class: "field", style: "align-items:flex-end", novalidate: "" }, a, b, go);
  root.append(
    h("div", { class: "hero" }, h("div", { class: "tag" }, h("span", { class: "eyebrow" }, "Compare")),
      h("h1", {}, "Two molecules, ", h("em", {}, "side by side"), "."),
      h("p", { class: "lede" }, "Each profile is predicted independently. Bars are on the same absolute scale; a diamond marks descriptors that both molecules rank highly.")),
    h("section", { class: "glass" }, form), out);

  async function run() {
    const A = a.querySelector("input").value, B = b.querySelector("input").value;
    for (const t of [A, B]) { const p = precheck(t); if (p) { return fail(new ApiError(p)); } }
    go.disabled = true;
    clear(out).append(h("div", { class: "glass", role: "status" }, "Profiling both molecules…", h("div", { class: "progress" }, h("i"))));
    try {
      const [ra, rb] = await Promise.all([predict(A.trim(), { topK: 30 }), predict(B.trim(), { topK: 30 })]);
      show(ra, rb);
    } catch (e) { fail(e instanceof ApiError ? e : new ApiError("Unexpected error.")); } finally { go.disabled = false; }
  }
  function fail(e) {
    clear(out).append(h("div", { class: "banner caution", role: "alert" }, icon("alert"), h("div", {}, h("strong", {}, "Could not compare"), e.message, " ", e.hint || "",
      e.requestId ? h("div", { class: "small muted" }, "Request ID: ", h("span", { class: "mono" }, e.requestId)) : null)));
  }
  function show(ra, rb) {
    const va = new Map(ra.predictions.map((p) => [p.descriptor, p])), vb = new Map(rb.predictions.map((p) => [p.descriptor, p]));
    const topA = new Set(ra.predictions.slice(0, 8).map((p) => p.descriptor)), topB = new Set(rb.predictions.slice(0, 8).map((p) => p.descriptor));
    const names = [...new Set([...topA, ...topB])].sort((x, y) => Math.max(vb.get(y)?.value || 0, va.get(y)?.value || 0) - Math.max(vb.get(x)?.value || 0, va.get(x)?.value || 0));
    const rows = names.map((n) => {
      const pa = va.get(n)?.value || 0, pb = vb.get(n)?.value || 0, shared = topA.has(n) && topB.has(n);
      return h("div", { class: `row${shared ? " shared" : ""}`, role: "group", "aria-label": `${n}: A ${pa.toFixed(2)}, B ${pb.toFixed(2)}${shared ? ", shared" : ""}` },
        h("div", { class: "side a" }, h("i", { style: `width:${pa * 100}%` })), h("div", { class: "lbl" }, n), h("div", { class: "side b" }, h("i", { style: `width:${pb * 100}%` })));
    });
    const overlap = [...topA].filter((n) => topB.has(n));
    clear(out).append(...[
      C.limitedBanner(ra), C.limitedBanner(rb),
      h("div", { class: "cmp-head" }, C.lensCard(ra, { title: "Molecule A" }), C.lensCard(rb, { title: "Molecule B" })),
      h("section", { class: "glass", "aria-label": "Descriptor comparison" },
        h("header", {}, h("h2", {}, "Descriptor comparison"), h("span", { class: "eyebrow" }, `${overlap.length} shared in each top 8`)),
        (ra.abstained || rb.abstained) ? h("p", { class: "notice" }, "One molecule is out of domain, so no profile is available for it and its bars are empty.") : null,
        h("div", { class: "diverge" }, ...rows),
        h("p", { class: "hint" }, "Scores may be calibrated probabilities or uncalibrated model scores; check each molecule's own page before comparing magnitudes across them.")),
      h("div", { class: "cmp-head" }, C.reliabilityCard(ra), C.reliabilityCard(rb))].filter(Boolean));
  }
  form.addEventListener("submit", (e) => { e.preventDefault(); run(); });
  run();
}
