import { h, clear, icon } from "./dom.js";
import { predict, precheck, ApiError } from "./api.js";
import { store, EXAMPLES } from "./store.js";
import * as C from "./components.js";

export function renderAnalyze(root, params) {
  clear(root);
  const input = h("input", { type: "text", id: "smiles", name: "smiles", autocomplete: "off", spellcheck: "false", autocapitalize: "off",
    placeholder: "Paste a SMILES, e.g. COc1cc(C=O)ccc1O", "aria-describedby": "smiles-help", value: params.get("smiles") || "" });
  const go = h("button", { class: "btn", type: "submit" }, "Profile molecule");
  const cancel = h("button", { class: "btn ghost", type: "button", hidden: true }, "Cancel");
  const help = h("p", { class: "hint", id: "smiles-help" }, "One molecule, as SMILES. ", h("kbd", {}, "Ctrl"), " + ", h("kbd", {}, "Enter"), " to run.");
  const out = h("div", { class: "grid", "aria-live": "polite", style: "margin-top:20px" });
  const chips = h("div", { class: "chips", role: "group", "aria-label": "Example molecules" },
    ...EXAMPLES.map(([name, smi]) => h("button", { class: "chip", type: "button", onclick: () => { input.value = smi; run(); } }, name)));
  const form = h("form", { class: "field", novalidate: "" }, input, go, cancel);

  root.append(
    h("div", { class: "hero" },
      h("div", { class: "tag" }, h("span", { class: "eyebrow" }, "Analyze"), h("span", { class: "scope-pill" }, "Sensor-free · predicts from structure")),
      h("h1", {}, "Read a molecule's ", h("em", {}, "odor profile"), " from its structure."),
      h("p", { class: "lede" }, "AromaScope estimates how a reference population might describe a compound's smell. It is a model-based prediction with its evidence and limits shown beside the result. It does not sense air.")),
    h("section", { class: "glass" }, form, chips, help),
    out);

  let controller = null, current = null, explaining = null;

  function emptyState() {
    clear(out).append(h("section", { class: "glass" }, h("h2", {}, "What happens when you run it"),
      h("div", { class: "empty-steps" },
        h("div", {}, h("b", {}, "1. Validate"), "The structure is parsed and standardized. Anything unsupported is explained, never guessed at."),
        h("div", {}, h("b", {}, "2. Profile"), "A model trained on labelled molecules ranks descriptors, with the score type stated."),
        h("div", {}, h("b", {}, "3. Judge"), "Coverage, evidence and nearest molecules show how much to trust the answer."))));
  }

  function showError(e, smiles) {
    input.setAttribute("aria-invalid", "true");
    const retry = e.retriable ? h("button", { class: "btn ghost", type: "button", style: "margin-top:10px", onclick: () => run() }, "Retry") : null;
    clear(out).append(h("div", { class: "banner caution", role: "alert" }, icon("alert"),
      h("div", {}, h("strong", {}, e.code === "network" || e.status >= 500 ? "Service problem" : "Check the structure"), e.message, " ", e.hint,
        e.requestId ? h("div", { class: "small muted", style: "margin-top:4px" }, "Request ID: ", h("span", { class: "mono" }, e.requestId)) : null, retry)));
    input.value = smiles; // preserve input for correction
  }

  function showResult(res) {
    current = res; store.last = { smiles: res.molecule.canonical_smiles, ...res.molecule.projection };
    const lensExtra = h("div", { style: "margin-top:12px;display:flex;gap:8px;flex-wrap:wrap" },
      h("a", { class: "btn ghost", href: `#/map?smiles=${encodeURIComponent(res.molecule.canonical_smiles)}`, style: "text-decoration:none" }, "Show on structure map"),
      h("a", { class: "btn ghost", href: `#/compare?a=${encodeURIComponent(res.molecule.canonical_smiles)}`, style: "text-decoration:none" }, "Compare with another"));
    const lens = C.lensCard(res, { title: "Structure", extra: lensExtra });
    let lensHost = lens;
    const explainNote = h("p", { class: "hint", hidden: true });
    const profile = () => C.profileCard(res, { onExplain: res.abstained ? null : explain, explaining });

    async function explain(descriptor) {
      if (explaining === descriptor) { explaining = null; return rebuild(); }
      explaining = descriptor;
      try {
        const r2 = await predict(res.molecule.input_smiles, { explain: descriptor });
        const ex = r2.explanation;
        if (ex && ex.available) {
          const fresh = C.svgBox("lens", ex.svg, "");
          const old = lensHost.querySelector(".lens");
          old.replaceWith(fresh);
          explainNote.hidden = false; explainNote.textContent = `Glow: atoms in substructures that push the model toward "${descriptor}". ${ex.note}`;
        } else { explainNote.hidden = false; explainNote.textContent = (ex && ex.note) || "No attribution available."; }
      } catch (e) { explainNote.hidden = false; explainNote.textContent = "Could not load the attribution right now."; }
      const prof2 = profile(); prof2.dataset.profile = "";
      out.querySelector("[data-profile]").replaceWith(prof2);
    }
    function rebuild() { showResult(res); }

    lens.append(explainNote);
    const prof = profile(); prof.dataset.profile = "";
    clear(out).append(...[
      C.limitedBanner(res),
      h("div", { class: "grid result-grid" }, lens, h("div", { class: "grid" }, prof, C.reliabilityCard(res))),
      C.evidenceCard(res, (smi) => { input.value = smi; run(); window.scrollTo({ top: 0, behavior: "smooth" }); }),
      C.explainerCard(res),
      h("p", { class: "footer" }, `Model ${res.model.id} v${res.model.version} · data ${res.model.data_snapshot} · features ${res.model.featurizer_version} · request ${res.request_id}`)].filter(Boolean));
    lensHost = lens;
  }

  async function run() {
    const text = input.value;
    input.removeAttribute("aria-invalid");
    const pre = precheck(text);
    if (pre) { showError(new ApiError(pre, { code: "client_validation" }), text); return; }
    if (controller) controller.abort();
    controller = new AbortController(); explaining = null;
    go.disabled = true; cancel.hidden = false;
    clear(out).append(h("div", { class: "glass", role: "status" }, h("span", {}, "Parsing structure and running the model…"), h("div", { class: "progress" }, h("i"))));
    try {
      showResult(await predict(text.trim(), {}, controller.signal));
    } catch (e) {
      if (e.name === "AbortError") { emptyState(); }
      else showError(e instanceof ApiError ? e : new ApiError("Unexpected error."), text);
    } finally { go.disabled = false; cancel.hidden = true; }
  }

  form.addEventListener("submit", (ev) => { ev.preventDefault(); run(); });
  cancel.addEventListener("click", () => controller && controller.abort());
  input.addEventListener("keydown", (ev) => { if ((ev.ctrlKey || ev.metaKey) && ev.key === "Enter") { ev.preventDefault(); run(); } });
  input.addEventListener("input", () => input.removeAttribute("aria-invalid"));
  if (input.value) run(); else emptyState();
}
