function resolveBase() {
  let q = null;
  try { q = new URLSearchParams(location.search).get("api"); } catch { /* ignore */ }
  if (q) { try { sessionStorage.setItem("aromascope.api", q); } catch { /* ignore */ } return q.replace(/\/$/, ""); }
  try { const s = sessionStorage.getItem("aromascope.api"); if (s) return s.replace(/\/$/, ""); } catch { /* ignore */ }
  return ((window.AROMASCOPE_CONFIG || {}).apiBase || "").replace(/\/$/, "");
}
export const API_BASE = resolveBase();

export class ApiError extends Error {
  constructor(message, { code = "service_error", hint = "", requestId = "", status = 0, retriable = false } = {}) {
    super(message);
    Object.assign(this, { code, hint, requestId, status, retriable });
  }
}

async function request(path, { method = "GET", body, signal } = {}) {
  let res;
  try {
    res = await fetch(API_BASE + path, {
      method, signal,
      headers: body ? { "Content-Type": "application/json" } : undefined,
      body: body ? JSON.stringify(body) : undefined,
    });
  } catch (e) {
    if (e.name === "AbortError") throw e;
    throw new ApiError("Could not reach the AromaScope service.", { code: "network", hint: "Check your connection or the API address, then retry.", retriable: true });
  }
  let data = null;
  try { data = await res.json(); } catch { /* non-JSON error */ }
  if (!res.ok) {
    const e = (data && data.error) || {};
    throw new ApiError(e.message || `The service returned an error (${res.status}).`, {
      code: e.code || "service_error", hint: e.hint || "", requestId: (data && data.request_id) || res.headers.get("x-request-id") || "",
      status: res.status, retriable: !!e.retriable || res.status >= 500,
    });
  }
  return data;
}

export const predict = (smiles, opts = {}, signal) =>
  request("/api/v1/predictions", { method: "POST", body: { smiles, top_k: opts.topK || 15, explain: opts.explain || null }, signal });
export const getModel = () => request("/api/v1/model");
export const getMap = () => request("/api/v1/map");

/** Light client-side checks so obvious mistakes never need a round trip. Returns an error string or "". */
export function precheck(text) {
  const t = (text || "").trim();
  if (!t) return "Enter a SMILES string, for example CCO for ethanol.";
  if (t.length > 400) return "That is longer than 400 characters. AromaScope profiles one small molecule at a time.";
  if (/\s/.test(t)) return "SMILES cannot contain spaces. Remove any name or text after the structure.";
  for (const [o, c] of [["(", ")"], ["[", "]"]]) {
    if (t.split(o).length !== t.split(c).length) return `Unbalanced ${o}${c} brackets. Check that each opening bracket is closed.`;
  }
  return "";
}
