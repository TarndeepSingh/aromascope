import { h, clear, icon } from "./dom.js";
import { getMap, predict } from "./api.js";
import { store } from "./store.js";
import * as C from "./components.js";

const PALETTE = ["#42d7c5", "#9b8cff", "#f0b769", "#7fb7ff", "#f29a9a", "#8fe388", "#e58fd9", "#5fd0f2"];
let cache = null;

export async function renderMap(root, params) {
  clear(root).append(h("div", { class: "hero" }, h("div", { class: "tag" }, h("span", { class: "eyebrow" }, "Odor Map")),
    h("h1", {}, "A ", h("em", {}, "structure map"), " of the reference molecules."),
    h("p", { class: "lede" }, "Each point is a known molecule placed by a PCA projection of its fingerprint. This shows chemical similarity, not perceived smell. Colour is the model's top predicted descriptor.")),
    h("p", { class: "muted", role: "status" }, "Loading map…"));
  let data;
  try { data = cache || (cache = await getMap()); } catch (e) {
    root.append(h("div", { class: "banner caution", role: "alert" }, icon("alert"), h("div", {}, h("strong", {}, "Map unavailable"), e.message, " ", e.hint || "")));
    return;
  }
  root.querySelector("[role=status]").remove();
  const pts = data.points, names = data.descriptors;
  const counts = new Map(); pts.forEach((p) => counts.set(p.top, (counts.get(p.top) || 0) + 1));
  const topIdx = [...counts.entries()].sort((a, b) => b[1] - a[1]).slice(0, PALETTE.length).map(([i]) => i);
  const colorOf = new Map(topIdx.map((i, k) => [i, PALETTE[k]]));
  let focus = null, selected = null, query = null;

  const canvas = h("canvas", { tabindex: "0", role: "img", "aria-label": `Structure map of ${pts.length} reference molecules. Use arrow keys to pan, plus and minus to zoom, zero to reset. A list alternative follows the map.` });
  const tip = h("div", { class: "tip", hidden: true });
  const box = h("div", { class: "map-box" }, canvas, tip,
    h("div", { class: "zoom" }, h("button", { class: "btn ghost", type: "button", "aria-label": "Zoom in", onclick: () => zoom(1.35) }, icon("plus")),
      h("button", { class: "btn ghost", type: "button", "aria-label": "Zoom out", onclick: () => zoom(1 / 1.35) }, icon("minus")),
      h("button", { class: "btn ghost", type: "button", "aria-label": "Reset view", onclick: reset }, "Reset")),
    h("div", { class: "map-note" }, `Structure projection · PCA explains ${(100 * data.explained_variance.reduce((a, b) => a + b, 0)).toFixed(0)}% of fingerprint variance`));
  const side = h("div", { class: "grid" });
  const swatches = h("div", { class: "swatches", role: "group", "aria-label": "Filter by top predicted descriptor" });
  const detail = h("div", { "aria-live": "polite" }, h("p", { class: "muted small" }, "Click a point to inspect it."));

  const xs = pts.map((p) => p.x), ys = pts.map((p) => p.y);
  const bx = [Math.min(...xs), Math.max(...xs)], by = [Math.min(...ys), Math.max(...ys)];
  let view = { k: 1, tx: 0, ty: 0 }, W = 0, H = 0, dpr = 1;
  const fit = () => Math.min((W - 60) / (bx[1] - bx[0] || 1), (H - 60) / (by[1] - by[0] || 1));
  const base = { s: 1 };
  const toScreen = (x, y) => [W / 2 + ((x - (bx[0] + bx[1]) / 2) * base.s) * view.k + view.tx, H / 2 - ((y - (by[0] + by[1]) / 2) * base.s) * view.k + view.ty];

  function resize() {
    const r = box.getBoundingClientRect(); dpr = window.devicePixelRatio || 1;
    W = r.width; H = r.height; canvas.width = W * dpr; canvas.height = H * dpr; base.s = fit(); draw();
  }
  function draw() {
    const ctx = canvas.getContext("2d"); ctx.setTransform(dpr, 0, 0, dpr, 0, 0); ctx.clearRect(0, 0, W, H);
    for (const pass of [0, 1]) {
      for (const p of pts) {
        const on = focus == null || p.top === focus; if ((pass === 1) !== on) continue;
        const [x, y] = toScreen(p.x, p.y); if (x < -6 || y < -6 || x > W + 6 || y > H + 6) continue;
        ctx.beginPath(); ctx.arc(x, y, on ? 3.1 : 2, 0, 6.283);
        ctx.fillStyle = colorOf.get(p.top) || "#7c8aa0"; ctx.globalAlpha = on ? (p.split === "train" ? 0.78 : 0.95) : 0.12; ctx.fill();
        if (on && p.split !== "train") { ctx.globalAlpha = 0.9; ctx.strokeStyle = "#f3f6fc"; ctx.lineWidth = 0.8; ctx.stroke(); }
      }
    }
    ctx.globalAlpha = 1;
    const mark = (pt, color, r) => { if (!pt) return; const [x, y] = toScreen(pt.x, pt.y); ctx.beginPath(); ctx.arc(x, y, r, 0, 6.283); ctx.strokeStyle = color; ctx.lineWidth = 2.6; ctx.stroke(); ctx.beginPath(); ctx.arc(x, y, r + 6, 0, 6.283); ctx.strokeStyle = color + "66"; ctx.lineWidth = 1.4; ctx.stroke(); };
    mark(selected, "#f3f6fc", 7); mark(query, "#f0b769", 8);
    if (query) { const [x, y] = toScreen(query.x, query.y); ctx.fillStyle = "#f0b769"; ctx.font = "600 12px system-ui"; ctx.fillText("your molecule", x + 14, y + 4); }
  }
  function reset() { view = { k: 1, tx: 0, ty: 0 }; draw(); }
  function zoom(f, cx = W / 2, cy = H / 2) {
    const k2 = Math.max(0.6, Math.min(40, view.k * f)), r = k2 / view.k;
    view.tx = cx - W / 2 - (cx - W / 2 - view.tx) * r; view.ty = cy - H / 2 - (cy - H / 2 - view.ty) * r; view.k = k2; draw();
  }
  function nearest(mx, my, rad = 9) {
    let best = null, bd = rad * rad;
    for (const p of pts) { if (focus != null && p.top !== focus) continue; const [x, y] = toScreen(p.x, p.y); const d = (x - mx) ** 2 + (y - my) ** 2; if (d < bd) { bd = d; best = p; } }
    return best;
  }
  let drag = null;
  canvas.addEventListener("pointerdown", (e) => { canvas.setPointerCapture(e.pointerId); drag = { x: e.clientX, y: e.clientY, tx: view.tx, ty: view.ty, moved: false }; });
  canvas.addEventListener("pointermove", (e) => {
    const r = canvas.getBoundingClientRect(), mx = e.clientX - r.left, my = e.clientY - r.top;
    if (drag) { const dx = e.clientX - drag.x, dy = e.clientY - drag.y; if (Math.abs(dx) + Math.abs(dy) > 3) drag.moved = true; view.tx = drag.tx + dx; view.ty = drag.ty + dy; draw(); tip.hidden = true; return; }
    const p = nearest(mx, my);
    if (p) { tip.hidden = false; tip.style.left = Math.min(mx + 14, W - 240) + "px"; tip.style.top = my + 14 + "px"; clear(tip).append(h("b", {}, names[p.top]), ` ${p.top_score.toFixed(2)}`, h("div", { class: "mono small" }, p.smiles), h("div", { class: "muted small" }, p.split + " set")); } else tip.hidden = true;
  });
  canvas.addEventListener("pointerup", (e) => {
    const was = drag; drag = null; if (was && was.moved) return;
    const r = canvas.getBoundingClientRect(); const p = nearest(e.clientX - r.left, e.clientY - r.top); if (p) select(p);
  });
  canvas.addEventListener("wheel", (e) => { e.preventDefault(); const r = canvas.getBoundingClientRect(); zoom(e.deltaY < 0 ? 1.18 : 1 / 1.18, e.clientX - r.left, e.clientY - r.top); }, { passive: false });
  canvas.addEventListener("keydown", (e) => {
    const step = 40, m = { ArrowLeft: [step, 0], ArrowRight: [-step, 0], ArrowUp: [0, step], ArrowDown: [0, -step] }[e.key];
    if (m) { view.tx += m[0]; view.ty += m[1]; draw(); e.preventDefault(); }
    else if (e.key === "+" || e.key === "=") zoom(1.3); else if (e.key === "-") zoom(1 / 1.3); else if (e.key === "0") reset();
  });
  new ResizeObserver(resize).observe(box);

  async function select(p) {
    selected = p; draw();
    clear(detail).append(h("p", { class: "muted small" }, "Loading…"));
    try {
      const res = await predict(p.smiles, { topK: 5 });
      clear(detail).append(C.svgBox("lens", res.molecule.structure_svg, ""), h("p", { class: "mono small", style: "margin-top:8px" }, p.smiles),
        h("p", { class: "small" }, `${p.split} set · top: `, h("b", {}, names[p.top])),
        h("ol", { class: "rank", style: "margin-top:8px" }, ...res.predictions.slice(0, 5).map((q) => h("li", { style: "grid-template-columns:1fr auto" }, q.descriptor, h("span", { class: "v" }, q.score_type === "calibrated_probability" ? `${(q.value * 100).toFixed(0)}%` : q.value.toFixed(2))))),
        h("a", { class: "btn ghost", href: `#/analyze?smiles=${encodeURIComponent(p.smiles)}`, style: "text-decoration:none;margin-top:10px;display:inline-block" }, "Open full analysis"));
    } catch (e) { clear(detail).append(h("p", { class: "notice" }, e.message || "Could not load details.")); }
  }

  const setFocus = (i) => { focus = focus === i ? null : i; [...swatches.children].forEach((b) => b.setAttribute("aria-pressed", String(Number(b.dataset.i) === focus))); draw(); renderList(); };
  swatches.append(...topIdx.map((i) => h("button", { type: "button", "data-i": String(i), "aria-pressed": "false", onclick: () => setFocus(i) }, h("i", { style: `background:${colorOf.get(i)}` }), `${names[i]} (${counts.get(i)})`)),
    h("button", { type: "button", disabled: "", style: "cursor:default" }, h("i", { style: "background:#7c8aa0" }), "other descriptors"));
  const search = h("input", { type: "search", placeholder: "Filter list by SMILES", "aria-label": "Filter the list by SMILES", oninput: () => renderList(), style: "width:100%" });
  const listBody = h("tbody");
  function renderList() {
    const q = search.value.trim().toLowerCase();
    const rows = pts.filter((p) => (focus == null || p.top === focus) && (!q || p.smiles.toLowerCase().includes(q))).slice(0, 80);
    clear(listBody).append(...rows.map((p) => h("tr", {}, h("td", {}, h("button", { class: "btn ghost", type: "button", onclick: () => select(p) }, "Inspect")), h("td", { class: "mono" }, p.smiles), h("td", {}, names[p.top]), h("td", {}, p.top_score.toFixed(2)), h("td", {}, p.split))));
  }
  renderList();

  side.append(
    h("section", { class: "glass" }, h("header", {}, h("h2", {}, "Filter"), h("span", { class: "eyebrow" }, `${pts.length} molecules`)), swatches,
      h("p", { class: "hint" }, "White-ringed points were held out from training (validation or test).")),
    h("section", { class: "glass" }, h("header", {}, h("h2", {}, "Selected")), detail));
  root.append(h("div", { class: "map-layout" }, box, side),
    h("section", { class: "glass sec" }, h("details", { open: "" }, h("summary", {}, "Browse as a list (keyboard and screen-reader friendly)"),
      h("div", { style: "margin:10px 0" }, search),
      h("div", { class: "scroll" }, h("table", {}, h("thead", {}, h("tr", {}, h("th", {}, "Action"), h("th", {}, "Structure (SMILES)"), h("th", {}, "Top descriptor"), h("th", {}, "Score"), h("th", {}, "Set"))), listBody)))));
  resize();

  // overlay the user's last molecule, or one passed in the URL
  const want = params.get("smiles");
  try {
    if (want) { const r = await predict(want, { topK: 1 }); query = r.molecule.projection; store.last = { smiles: r.molecule.canonical_smiles, ...query }; }
    else if (store.last) query = { x: store.last.x, y: store.last.y };
    draw();
  } catch { /* overlay is optional */ }
}
