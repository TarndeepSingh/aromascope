import { s } from "./dom.js";

const TAU = Math.PI * 2;
const pt = (cx, cy, r, a) => [cx + r * Math.cos(a), cy + r * Math.sin(a)];
const f = (n) => n.toFixed(2);

/**
 * The odor rose: one petal per descriptor, petal length = the model value on an absolute 0-1 scale
 * (no per-chart normalization, so a weak profile looks weak). Solid petals are calibrated probabilities;
 * hatched petals are uncalibrated model scores.
 */
export function rose(predictions, { n = 12 } = {}) {
  const items = predictions.slice(0, n);
  const W = 380, H = 360, cx = W / 2, cy = H / 2 + 2, r0 = 22, rMax = 132;
  const svg = s("svg", { class: "rose", viewBox: `0 0 ${W} ${H}`, role: "img",
    "aria-label": `Odor rose of the top ${items.length} predicted descriptors: ` + items.map((p) => `${p.descriptor} ${p.value.toFixed(2)}`).join(", ") + ". The ranked list below has the same data." });
  const defs = s("defs", {},
    s("linearGradient", { id: "pg", x1: "0", y1: "0", x2: "1", y2: "1" }, s("stop", { offset: "0", "stop-color": "#42d7c5" }), s("stop", { offset: "1", "stop-color": "#9b8cff" })),
    s("pattern", { id: "hatch", width: "6", height: "6", patternUnits: "userSpaceOnUse", patternTransform: "rotate(45)" },
      s("rect", { width: "6", height: "6", fill: "rgba(155,140,255,.16)" }), s("line", { x1: "0", y1: "0", x2: "0", y2: "6", stroke: "#9b8cff", "stroke-width": "2.4" })));
  svg.append(defs);
  for (const v of [0.25, 0.5, 0.75, 1]) {
    const r = r0 + (rMax - r0) * v;
    svg.append(s("circle", { class: "ring", cx, cy, r }));
    if (v === 0.5 || v === 1) svg.append(s("text", { x: cx + 3, y: cy - r - 3, "font-size": "9" }, v.toFixed(1)));
  }
  const sector = TAU / Math.max(items.length, 1);
  items.forEach((p, i) => {
    const mid = -Math.PI / 2 + i * sector, half = sector * 0.4;
    const a1 = mid - half, a2 = mid + half;
    const r = r0 + (rMax - r0) * Math.max(0.02, Math.min(1, p.value));
    const [x1, y1] = pt(cx, cy, r0, a1), [x2, y2] = pt(cx, cy, r, a1), [x3, y3] = pt(cx, cy, r, a2), [x4, y4] = pt(cx, cy, r0, a2);
    const d = `M${f(x1)} ${f(y1)} L${f(x2)} ${f(y2)} A${f(r)} ${f(r)} 0 0 1 ${f(x3)} ${f(y3)} L${f(x4)} ${f(y4)} A${r0} ${r0} 0 0 0 ${f(x1)} ${f(y1)}Z`;
    const raw = p.score_type !== "calibrated_probability";
    svg.append(s("path", { class: "petal", d, fill: raw ? "url(#hatch)" : "url(#pg)", "fill-opacity": raw ? "1" : String(0.55 + 0.4 * (1 - i / items.length)),
      stroke: raw ? "#9b8cff" : "rgba(255,255,255,.35)", "stroke-width": raw ? "1.4" : "0.8", "stroke-linejoin": "round" }, s("title", {}, `${p.descriptor}: ${p.value.toFixed(2)} (${raw ? "model score" : "calibrated probability"})`)));
    const [lx, ly] = pt(cx, cy, rMax + 10, mid);
    const c = Math.cos(mid);
    const name = p.descriptor.length > 13 ? p.descriptor.slice(0, 12) + "…" : p.descriptor;
    svg.append(s("text", { x: f(lx), y: f(ly + 4), "text-anchor": Math.abs(c) < 0.25 ? "middle" : c > 0 ? "start" : "end", "font-size": i < 3 ? "12.5" : "11", style: i < 3 ? "fill:#f3f6fc;font-weight:600" : "" }, name));
  });
  return svg;
}

/** Semicircular coverage gauge: similarity to the nearest training molecule against the release's two thresholds. */
export function gauge(applic) {
  const W = 260, H = 150, cx = 130, cy = 120, r = 88;
  const sim = Math.max(0, Math.min(1, applic.max_similarity));
  const { in_domain: tIn, out_of_domain: tOut } = applic.thresholds;
  const ang = (t) => Math.PI * (1 - t);
  const at = (t, rr = r) => [cx + rr * Math.cos(ang(t)), cy - rr * Math.sin(ang(t))];
  const arc = (t0, t1) => { const [x0, y0] = at(t0), [x1, y1] = at(t1); return `M${f(x0)} ${f(y0)} A${r} ${r} 0 0 1 ${f(x1)} ${f(y1)}`; };
  const col = { out: "#f29a9a", mid: "#f0b769", ok: "#42d7c5" };
  const svg = s("svg", { class: "gauge", viewBox: `0 0 ${W} ${H}`, role: "img",
    "aria-label": `Coverage gauge. Nearest training molecule similarity ${sim.toFixed(2)}. Below ${tOut} is out of domain, ${tOut} to ${tIn} is limited evidence, ${tIn} or higher is well covered.` });
  for (const [a, b, c] of [[0, tOut, col.out], [tOut, tIn, col.mid], [tIn, 1, col.ok]])
    svg.append(s("path", { d: arc(a, b), fill: "none", stroke: c, "stroke-width": "11", "stroke-opacity": "0.5", "stroke-linecap": "butt" }));
  svg.append(s("path", { d: arc(0, sim), fill: "none", stroke: "#f3f6fc", "stroke-width": "3", "stroke-linecap": "round" }));
  const [mx, my] = at(sim);
  svg.append(s("circle", { cx: f(mx), cy: f(my), r: "7", fill: "#0a1020", stroke: "#f3f6fc", "stroke-width": "2.5" }));
  for (const [t, label] of [[tOut, tOut.toFixed(2)], [tIn, tIn.toFixed(2)]]) {
    const [x1, y1] = at(t, r - 10), [x2, y2] = at(t, r + 10), [tx, ty] = at(t, r + 21);
    svg.append(s("line", { x1: f(x1), y1: f(y1), x2: f(x2), y2: f(y2), stroke: "#aab6c5", "stroke-width": "1.2" }));
    svg.append(s("text", { x: f(tx), y: f(ty + 3), "text-anchor": "middle" }, label));
  }
  svg.append(s("text", { class: "big", x: cx, y: cy - 8, "text-anchor": "middle" }, sim.toFixed(2)));
  svg.append(s("text", { x: cx, y: cy + 10, "text-anchor": "middle" }, "nearest-neighbour similarity"));
  return svg;
}
