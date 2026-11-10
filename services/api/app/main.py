"""AromaScope prediction API.

Stateless by default. Loads one immutable, checksum-verified release at startup and reports
ready only if that succeeds. Raw SMILES are never written to logs.
"""
from __future__ import annotations

import logging
import os
import threading
import time
import uuid
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from aromascope import __version__, chemistry as chem
from aromascope.bundle import BundleError, Release, load_release, resolve_release_dir
from aromascope.predictor import Predictor

log = logging.getLogger("aromascope")
logging.basicConfig(level=os.environ.get("AROMASCOPE_LOG_LEVEL", "INFO"))

LIMITATIONS = [
    "Predicted from structure only; this is not a sensory measurement and not an air sensor.",
    "Applies to the reference vocabulary and source dataset, not to every person's perception.",
    "Purity, concentration, temperature and mixtures are not modelled.",
    "Structural similarity does not guarantee similar odor. The model can be wrong.",
    "Model features ignore stereochemistry: enantiomers receive identical predictions.",
]


class PredictionRequest(BaseModel):
    smiles: str = Field(..., description="One molecule as a SMILES string.", max_length=2000)
    top_k: int = Field(15, ge=1, le=50)
    explain: str | None = Field(None, description="Descriptor to attribute to substructures (linear models only).", max_length=64)


class State:
    release: Release | None = None
    predictor: Predictor | None = None
    error: str | None = None


state = State()


def load_into_state() -> None:
    try:
        release = load_release(resolve_release_dir())
        state.release, state.predictor, state.error = release, Predictor(release), None
        log.info("loaded release %s (%s)", release.manifest["version"], release.model_kind)
        for w in release.warnings:
            log.warning(w)
    except Exception as exc:  # readiness must fail closed, never load an arbitrary model
        state.release = state.predictor = None
        state.error = f"{type(exc).__name__}: {exc}"
        log.error("release failed to load: %s", state.error)


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_into_state()
    yield


app = FastAPI(
    title="AromaScope API",
    version=__version__,
    description="Predicts odor descriptors from molecular structure. Not a sensor and not a sensory measurement.",
    lifespan=lifespan,
)

origins = [o.strip() for o in os.environ.get("AROMASCOPE_ORIGINS", "http://localhost:8080,http://127.0.0.1:8080").split(",") if o.strip()]
app.add_middleware(CORSMiddleware, allow_origins=origins, allow_methods=["GET", "POST"], allow_headers=["Content-Type"])

# ---- simple in-memory rate limiter (per client IP, per minute) --------------------------------------------
RATE_LIMIT = int(os.environ.get("AROMASCOPE_RATE_LIMIT", "60"))
_hits: dict[str, deque] = defaultdict(deque)
_lock = threading.Lock()


def rate_limited(ip: str) -> bool:
    if RATE_LIMIT <= 0:
        return False
    now = time.monotonic()
    with _lock:
        q = _hits[ip]
        while q and now - q[0] > 60:
            q.popleft()
        if len(q) >= RATE_LIMIT:
            return True
        q.append(now)
        if len(_hits) > 10_000:  # bound memory
            _hits.clear()
    return False


def error_response(status: int, code: str, message: str, hint: str, request_id: str, retriable: bool = False):
    return JSONResponse(
        status_code=status,
        content={"error": {"code": code, "message": message, "hint": hint, "retriable": retriable}, "request_id": request_id},
        headers={"X-Request-ID": request_id},
    )


@app.middleware("http")
async def request_context(request: Request, call_next):
    rid = request.headers.get("x-request-id") or uuid.uuid4().hex[:12]
    request.state.request_id = rid
    start = time.perf_counter()
    ip = request.client.host if request.client else "unknown"
    if os.environ.get("AROMASCOPE_TRUST_PROXY") == "1":  # behind a TLS-terminating proxy, use the forwarded client IP
        ip = request.headers.get("x-forwarded-for", ip).split(",")[0].strip()
    if request.url.path.startswith("/api/") and rate_limited(ip):
        return error_response(429, "rate_limited", "Too many requests.", "Wait a minute and try again.", rid, True)
    try:
        response = await call_next(request)
    except Exception:
        log.exception("unhandled error rid=%s path=%s", rid, request.url.path)
        return error_response(500, "internal_error", "Something went wrong on the server.", "Retry in a moment.", rid, True)
    ms = (time.perf_counter() - start) * 1000
    # log request id, route, status, timing and model version only; never the input structure
    log.info("rid=%s %s %s %d %.0fms model=%s", rid, request.method, request.url.path, response.status_code, ms,
             state.release.manifest["version"] if state.release else "none")
    response.headers["X-Request-ID"] = rid
    return response


@app.exception_handler(RequestValidationError)
async def validation_handler(request: Request, exc: RequestValidationError):
    rid = getattr(request.state, "request_id", "n/a")
    return error_response(422, "invalid_request", "The request body is not valid.", "Send JSON like {\"smiles\": \"CCO\"}.", rid)


class NotReady(Exception):
    """The model release is not loaded (missing, tampered with, or incompatible)."""


def require_ready() -> Predictor:
    if state.predictor is None:
        raise NotReady()
    return state.predictor


@app.exception_handler(NotReady)
async def not_ready_handler(request: Request, exc: NotReady):
    rid = getattr(request.state, "request_id", "n/a")
    return error_response(503, "model_unavailable", "The model release is not loaded.", "Try again shortly.", rid, True)


# ---- routes ---------------------------------------------------------------------------------------------------
@app.get("/health/live")
def live():
    return {"status": "alive"}


@app.get("/health/ready")
def ready():
    if state.release is None:
        return JSONResponse(status_code=503, content={"status": "unavailable", "reason": state.error})
    m = state.release.manifest
    return {"status": "ready", "model_version": m["version"], "warnings": state.release.warnings}


@app.get("/api/v1/model")
def model_info():
    p = require_ready()
    r = p.r
    m = r.manifest
    return {
        "model": {
            "id": m["id"],
            "version": m["version"],
            "kind": m["model_kind"],
            "created_at": m["created_at"],
            "code_revision": m["code_revision"],
            "data_snapshot": m["data_snapshot"],
            "featurizer_version": m["featurizer_version"],
            "policy_version": m["policy_version"],
            "split": m["split"],
            "checksums": m["files"],
        },
        "task": task_block(r),
        "labels": r.labels,
        "evaluation": r.evaluation,
        "ledger": r.ledger,
        "model_card_markdown": r.model_card,
        "limitations": LIMITATIONS,
    }


def task_block(r: Release) -> dict[str, Any]:
    gate = r.evaluation.get("calibration", {}).get("gate_passed", False)
    return {
        "target_type": "multilabel_descriptor",
        "description": "Probability that each descriptor is listed for a molecule like this in the reference dataset."
        if gate
        else "Uncalibrated model score per descriptor. Ranks descriptors but is not a probability.",
        "scale": "0 to 1",
        "descriptors": len(r.label_names),
        "reference_population": r.ledger["sources"][0]["task"],
    }


@app.get("/api/v1/map")
def odor_map():
    p = require_ready()
    return p.r.map_data


@app.post("/api/v1/predictions")
def predict(body: PredictionRequest, request: Request):
    p = require_ready()
    rid = request.state.request_id
    try:
        std = chem.standardize(body.smiles)
    except chem.ChemistryError as exc:
        status = 413 if exc.code == "input_too_large" else 422
        return error_response(status, exc.code, exc.message, exc.hint, rid)
    result = p.predict(std, top_k=body.top_k, explain=body.explain)
    r = p.r
    m = r.manifest
    return {
        "request_id": rid,
        "molecule": {
            "input_smiles": std.input_smiles,
            "canonical_smiles": std.canonical_smiles,
            "formula": std.formula,
            "mol_weight": round(std.mol_weight, 3),
            "heavy_atoms": std.heavy_atoms,
            "unspecified_stereo": std.unspecified_stereo,
            "changes": std.changes,
            "structure_svg": chem.render_svg(std.mol),
            "projection": result["projection"],
        },
        "model": {
            "id": m["id"],
            "version": m["version"],
            "data_snapshot": m["data_snapshot"],
            "featurizer_version": m["featurizer_version"],
            "policy_version": m["policy_version"],
            "kind": m["model_kind"],
        },
        "task": task_block(r),
        "abstained": result["abstained"],
        "predictions": result["predictions"],
        "applicability": result["applicability"],
        "evidence": result["evidence"],
        "explanation": result["explanation"],
        "limitations": LIMITATIONS,
    }


# ---- optional single-service mode: serve the static web client from the same container ----------------------------
web_dir = os.environ.get("AROMASCOPE_SERVE_WEB")
if web_dir and Path(web_dir).is_dir():
    app.mount("/", StaticFiles(directory=web_dir, html=True), name="web")
