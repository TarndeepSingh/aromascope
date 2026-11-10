const SVG_NS = "http://www.w3.org/2000/svg";

function append(el, kids) {
  for (const c of kids.flat(Infinity)) {
    if (c == null || c === false) continue;
    el.append(c.nodeType ? c : document.createTextNode(String(c)));
  }
}

/** Build an HTML element. Strings are always inserted as text, never as markup. */
export function h(tag, attrs = {}, ...kids) {
  const el = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (v == null || v === false) continue;
    if (k === "class") el.className = v;
    else if (k.startsWith("on") && typeof v === "function") el.addEventListener(k.slice(2).toLowerCase(), v);
    else el.setAttribute(k, v === true ? "" : v);
  }
  append(el, kids);
  return el;
}

/** Build an SVG element. */
export function s(tag, attrs = {}, ...kids) {
  const el = document.createElementNS(SVG_NS, tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (v == null || v === false) continue;
    el.setAttribute(k, v);
  }
  append(el, kids);
  return el;
}

/** Parse server-generated SVG and strip anything that could execute. Returns an <svg> node or null. */
export function safeSvg(text) {
  if (!text || typeof text !== "string") return null;
  try {
    const doc = new DOMParser().parseFromString(text, "image/svg+xml");
    const root = doc.documentElement;
    if (!root || root.nodeName.toLowerCase() !== "svg" || doc.querySelector("parsererror")) return null;
    root.querySelectorAll("script, foreignObject, iframe, object, embed, link, style[src]").forEach((n) => n.remove());
    root.querySelectorAll("*").forEach((n) => {
      for (const a of [...n.attributes]) {
        const name = a.name.toLowerCase();
        if (name.startsWith("on") || (name.includes("href") && /^\s*javascript:/i.test(a.value))) n.removeAttribute(a.name);
      }
    });
    const node = document.importNode(root, true);
    node.removeAttribute("width");
    node.removeAttribute("height");
    node.setAttribute("preserveAspectRatio", "xMidYMid meet");
    return node;
  } catch {
    return null;
  }
}

export const ICONS = {
  check: "M4 12l5 5L20 6",
  alert: "M12 3l10 18H2L12 3zM12 10v5M12 18v.01",
  x: "M5 5l14 14M19 5L5 19",
  info: "M12 8v.01M11 12h1v5h1M12 3a9 9 0 100 18 9 9 0 000-18z",
  eye: "M2 12s4-7 10-7 10 7 10 7-4 7-10 7S2 12 2 12zM12 9a3 3 0 100 6 3 3 0 000-6z",
  copy: "M9 9h10v10H9zM5 15V5h10",
  plus: "M12 5v14M5 12h14",
  minus: "M5 12h14",
  home: "M4 12l8-7 8 7M6 10v9h12v-9",
  analyze: "M9 3h6M10 3v6l-5 9a2 2 0 002 3h10a2 2 0 002-3l-5-9V3M8 15h8",
  compare: "M8 3v18M16 3v18M3 8h5M16 16h5",
  map: "M3 6l6-2 6 2 6-2v14l-6 2-6-2-6 2zM9 4v14M15 6v14",
  model: "M12 3l8 4.5v9L12 21l-8-4.5v-9zM12 12l8-4.5M12 12v9M12 12L4 7.5",
};

export function icon(name, cls = "") {
  return s("svg", { viewBox: "0 0 24 24", "aria-hidden": "true", class: cls }, s("path", { d: ICONS[name] }));
}

export function clear(el) {
  while (el.firstChild) el.removeChild(el.firstChild);
  return el;
}

export function fmt(v, d = 2) {
  return v == null || Number.isNaN(v) ? "n/a" : Number(v).toFixed(d);
}
