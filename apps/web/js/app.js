import { h, icon } from "./dom.js";
import { renderAnalyze } from "./analyze.js";
import { renderCompare } from "./compare.js";
import { renderMap } from "./map.js";
import { renderModel } from "./model.js";
import { API_BASE } from "./api.js";

const routes = { analyze: renderAnalyze, compare: renderCompare, map: renderMap, model: renderModel };
const labels = { analyze: "Analyze", compare: "Compare", map: "Odor Map", model: "Model & Data" };
const root = document.getElementById("view");
const nav = document.getElementById("nav");

for (const k of Object.keys(routes)) nav.append(h("a", { href: `#/${k}`, "data-route": k }, icon(k === "map" ? "map" : k === "model" ? "model" : k), labels[k]));

function parse() {
  const raw = location.hash.replace(/^#\/?/, "");
  const [path, qs] = raw.split("?");
  return { name: routes[path] ? path : "analyze", params: new URLSearchParams(qs || "") };
}
function render() {
  const { name, params } = parse();
  nav.querySelectorAll("a").forEach((a) => (a.dataset.route === name ? a.setAttribute("aria-current", "page") : a.removeAttribute("aria-current")));
  document.title = `${labels[name]} · AromaScope`;
  window.scrollTo(0, 0);
  root.focus({ preventScroll: true });
  routes[name](root, params);
}
window.addEventListener("hashchange", render);
render();

// Flag a demo/mock backend so nobody mistakes sample output for a real model.
fetch(API_BASE + "/health/ready").then((r) => r.json()).then((j) => {
  if (j && /mock/i.test(j.model_version || "")) {
    document.getElementById("demo-banner").hidden = false;
  }
  if (j && j.status === "unavailable") {
    const b = document.getElementById("demo-banner"); b.hidden = false; b.textContent = "The model service is reachable but no release is loaded yet.";
  }
}).catch(() => {});
