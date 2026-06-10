"""
Verity Scoring API — FastAPI endpoint.

Endpoints:
    POST /score        → real-time fraud scoring (target: < 20ms p99)
    POST /score/batch  → batch scoring
    POST /explain      → full SHAP breakdown for a transaction
    GET  /audit/{txn}  → retrieve audit log entry
    GET  /health       → liveness + readiness probe
    GET  /metrics      → latency stats, decision distribution, drift status

Operational modes:
    NORMAL   — score and enforce decisions
    SHADOW   — score and log decisions but never block (for safe rollouts)
    DEGRADED — ML unavailable; fall back to rule engine only
    FAIL_OPEN  — on any error, return PASS (prioritise availability)
    FAIL_CLOSED — on any error, return BLOCK (prioritise safety)

Run:
    uvicorn api.main:app --host 0.0.0.0 --port 8000 --workers 4
"""

from __future__ import annotations

import os
import time
import traceback
from enum import Enum
from typing import Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Internal imports — adjust paths as needed in your project layout
# ---------------------------------------------------------------------------
try:
    from stream.rules import RuleEngine
    from stream.windows import SlidingWindowAggregator
    from ml.ensemble import EnsembleScorer
    from explainability.shap import KernelSHAP, top_features
    from explainability.registry import ModelRegistry, AuditLog
    from explainability.drift import PSIDriftDetector
    _IMPORTS_OK = True
except ImportError:
    _IMPORTS_OK = False   # allow import during testing without full project

# ---------------------------------------------------------------------------
# Enums & constants
# ---------------------------------------------------------------------------

class OperatingMode(str, Enum):
    NORMAL      = "normal"
    SHADOW      = "shadow"
    DEGRADED    = "degraded"
    FAIL_OPEN   = "fail_open"
    FAIL_CLOSED = "fail_closed"


class Decision(str, Enum):
    BLOCK  = "BLOCK"
    REVIEW = "REVIEW"
    PASS   = "PASS"


BLOCK_THRESHOLD  = 0.90
REVIEW_THRESHOLD = 0.70

CURRENT_MODE = OperatingMode(os.getenv("VERITY_MODE", OperatingMode.NORMAL))
MODEL_VERSION = os.getenv("VERITY_MODEL_VERSION", "ensemble-latest")

# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class TransactionRequest(BaseModel):
    transaction_id: str
    user_id: str
    merchant_id: str
    amount: float = Field(..., gt=0)
    mcc: str
    city: str
    device_id: str
    # Optional pre-computed feature vector (if caller already has it)
    feature_vector: Optional[list[float]] = None
    metadata: Optional[dict] = None


class ScoreResponse(BaseModel):
    transaction_id: str
    risk_score: float
    rule_score: float
    ml_score: float
    decision: str
    shadow_mode: bool
    model_version: str
    latency_ms: float
    explanation: Optional[str] = None  # brief; full explanation via /explain


class ExplainResponse(BaseModel):
    transaction_id: str
    risk_score: float
    decision: str
    baseline: float
    top_features: list[dict]
    explanation: str
    model_version: str


class AuditResponse(BaseModel):
    found: bool
    entries: list[dict]


class HealthResponse(BaseModel):
    status: str          # "ok" | "degraded" | "down"
    mode: str
    model_loaded: bool
    uptime_s: float


class MetricsResponse(BaseModel):
    total_scored: int
    decisions: dict
    p50_ms: float
    p99_ms: float
    shadow_mode: bool
    drift_alert: bool
    drift_report: Optional[dict] = None


# ---------------------------------------------------------------------------
# Application state (module-level singleton)
# ---------------------------------------------------------------------------

class AppState:
    def __init__(self):
        self.mode = CURRENT_MODE
        self.model_version = MODEL_VERSION
        self.start_time = time.time()

        # Components — loaded lazily / at startup
        self.rule_engine: Optional[object] = None
        self.window_aggregator: Optional[object] = None
        self.scorer: Optional[object] = None
        self.shap_explainer: Optional[object] = None
        self.registry: Optional[object] = None
        self.audit_log: Optional[object] = None
        self.drift_detector: Optional[object] = None

        # In-memory metrics
        self.latencies: list[float] = []
        self.decision_counts: dict[str, int] = {
            Decision.BLOCK: 0,
            Decision.REVIEW: 0,
            Decision.PASS: 0,
        }
        self.total_scored = 0
        self.last_drift_report: Optional[dict] = None
        self.drift_alert = False

    def is_model_loaded(self) -> bool:
        return self.scorer is not None

    def record_latency(self, ms: float) -> None:
        self.latencies.append(ms)
        # Keep rolling window of last 10 000 samples
        if len(self.latencies) > 10_000:
            self.latencies = self.latencies[-10_000:]

    def percentile(self, p: float) -> float:
        if not self.latencies:
            return 0.0
        import math
        s = sorted(self.latencies)
        idx = math.ceil(p / 100.0 * len(s)) - 1
        return s[max(0, idx)]


_state = AppState()

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Verity — Transaction Intelligence API",
    description="Real-time fraud detection with explainability and audit logging.",
    version="1.0.0",
)


# ---------------------------------------------------------------------------
# Startup / shutdown
# ---------------------------------------------------------------------------

@app.on_event("startup")
async def startup():
    """Load models and initialise components on startup."""
    if not _IMPORTS_OK:
        return  # running in test mode

    try:
        _state.registry = ModelRegistry()
        _state.audit_log = AuditLog()
        _state.rule_engine = RuleEngine()
        _state.window_aggregator = SlidingWindowAggregator()

        active = _state.registry.get_active("ensemble")
        if active:
            _state.scorer = EnsembleScorer.load(active.artifact_path)
            _state.model_version = active.version

            # Init SHAP if background data available
            bg_path = os.getenv("VERITY_BG_DATA", "models/background.npy")
            if os.path.exists(bg_path):
                import numpy as np
                bg = np.load(bg_path)
                _state.shap_explainer = KernelSHAP(
                    predict_fn=lambda X: _state.scorer.predict_proba(X),
                    background=bg,
                    n_samples=256,
                )

            # Load drift detector if reference saved
            drift_path = os.getenv("VERITY_DRIFT_REF", "models/drift_ref.json")
            if os.path.exists(drift_path):
                _state.drift_detector = PSIDriftDetector.load_reference(drift_path)
    except Exception as e:
        print(f"[startup] Warning: {e}")
        _state.mode = OperatingMode.DEGRADED


# ---------------------------------------------------------------------------
# Core scoring logic (extracted for reuse in batch + shadow)
# ---------------------------------------------------------------------------

def _score_transaction(req: TransactionRequest) -> ScoreResponse:
    t0 = time.perf_counter()

    try:
        txn = req.dict()
        feature_vector = req.feature_vector

        # --- Rule engine ---
        rule_result = None
        rule_score = 0.0
        if _state.rule_engine is not None:
            try:
                rule_result = _state.rule_engine.evaluate(txn)
                rule_score = getattr(rule_result, "rule_score", 0.0)
                # Immediate block from rules
                if getattr(rule_result, "decision", None) == "BLOCK":
                    decision = Decision.BLOCK
                    ml_score = rule_score
                    ensemble_score = 1.0
                    return _build_response(
                        req, ensemble_score, rule_score, ml_score,
                        decision, t0, feature_vector
                    )
            except Exception:
                rule_score = 0.0

        # --- Feature vector (from window aggregator if not provided) ---
        if feature_vector is None and _state.window_aggregator is not None:
            try:
                fv = _state.window_aggregator.get_features(
                    req.user_id, req.device_id, req.merchant_id,
                    req.amount, req.city
                )
                feature_vector = fv
            except Exception:
                feature_vector = [0.0] * 23   # zero vector fallback

        # --- ML scoring ---
        ml_score = rule_score  # fallback
        if _state.mode == OperatingMode.DEGRADED or _state.scorer is None:
            ml_score = rule_score
        else:
            try:
                import numpy as np
                fv_arr = np.array(feature_vector, dtype=float).reshape(1, -1)
                ml_score = float(_state.scorer.predict_proba(fv_arr)[0])
            except Exception:
                if _state.mode == OperatingMode.FAIL_CLOSED:
                    return _error_response(req, Decision.BLOCK, t0)
                ml_score = rule_score   # degrade gracefully

        # Ensemble: simple weighted combination (matches ml/ensemble.py weights)
        ensemble_score = float(0.4 * ml_score + 0.6 * rule_score) \
            if _state.mode != OperatingMode.DEGRADED \
            else rule_score

        # --- Decision ---
        decision = _threshold_decision(ensemble_score)

        return _build_response(
            req, ensemble_score, rule_score, ml_score,
            decision, t0, feature_vector
        )

    except Exception:
        tb = traceback.format_exc()
        print(f"[score] Unhandled error:\n{tb}")
        if _state.mode == OperatingMode.FAIL_CLOSED:
            return _error_response(req, Decision.BLOCK, t0)
        return _error_response(req, Decision.PASS, t0)


def _build_response(
    req: TransactionRequest,
    ensemble_score: float,
    rule_score: float,
    ml_score: float,
    decision: Decision,
    t0: float,
    feature_vector,
) -> ScoreResponse:
    latency_ms = (time.perf_counter() - t0) * 1000

    shadow = _state.mode == OperatingMode.SHADOW
    enforced_decision = Decision.PASS if shadow else decision

    # Audit
    if _state.audit_log is not None and feature_vector is not None:
        _state.audit_log.log(
            transaction_id=req.transaction_id,
            model_version=_state.model_version,
            feature_vector=feature_vector,
            rule_score=rule_score,
            ml_score=ml_score,
            ensemble_score=ensemble_score,
            decision=enforced_decision,
            top_shap_features=[],   # filled on /explain call
            latency_ms=latency_ms,
            shadow_mode=shadow,
            metadata=req.metadata or {},
        )

    # Drift monitoring
    if _state.drift_detector is not None and feature_vector is not None:
        import numpy as np
        _state.drift_detector.push(np.array(feature_vector), ensemble_score)
        if _state.drift_detector.buffer_size() >= _state.drift_detector.window_size:
            report = _state.drift_detector.check()
            _state.last_drift_report = report.to_dict()
            _state.drift_alert = report.alert

    # Metrics
    _state.record_latency(latency_ms)
    _state.total_scored += 1
    _state.decision_counts[enforced_decision] = \
        _state.decision_counts.get(enforced_decision, 0) + 1

    return ScoreResponse(
        transaction_id=req.transaction_id,
        risk_score=round(ensemble_score, 4),
        rule_score=round(rule_score, 4),
        ml_score=round(ml_score, 4),
        decision=enforced_decision,
        shadow_mode=shadow,
        model_version=_state.model_version,
        latency_ms=round(latency_ms, 2),
    )


def _error_response(req: TransactionRequest, decision: Decision, t0: float) -> ScoreResponse:
    latency_ms = (time.perf_counter() - t0) * 1000
    return ScoreResponse(
        transaction_id=req.transaction_id,
        risk_score=1.0 if decision == Decision.BLOCK else 0.0,
        rule_score=0.0,
        ml_score=0.0,
        decision=decision,
        shadow_mode=_state.mode == OperatingMode.SHADOW,
        model_version=_state.model_version,
        latency_ms=round(latency_ms, 2),
    )


def _threshold_decision(score: float) -> Decision:
    if score >= BLOCK_THRESHOLD:
        return Decision.BLOCK
    elif score >= REVIEW_THRESHOLD:
        return Decision.REVIEW
    return Decision.PASS


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.post("/score", response_model=ScoreResponse)
async def score(req: TransactionRequest):
    """Score a single transaction. Target latency < 20ms p99."""
    return _score_transaction(req)


@app.post("/score/batch", response_model=list[ScoreResponse])
async def score_batch(requests: list[TransactionRequest]):
    """Score a batch of transactions."""
    return [_score_transaction(r) for r in requests]


@app.post("/explain", response_model=ExplainResponse)
async def explain(req: TransactionRequest):
    """
    Return full KernelSHAP explanation for a transaction.
    Slower than /score — not on the critical path.
    """
    score_resp = _score_transaction(req)

    if _state.shap_explainer is None or req.feature_vector is None:
        return ExplainResponse(
            transaction_id=req.transaction_id,
            risk_score=score_resp.risk_score,
            decision=score_resp.decision,
            baseline=0.0,
            top_features=[],
            explanation="SHAP explainer not initialised or feature vector missing.",
            model_version=_state.model_version,
        )

    import numpy as np
    shap_result = _state.shap_explainer.explain(
        np.array(req.feature_vector, dtype=float)
    )
    tf = [
        {"feature": c["feature"], "shap": round(c["shap"], 4), "value": c["value"]}
        for c in shap_result["contributions"][:8]
    ]

    return ExplainResponse(
        transaction_id=req.transaction_id,
        risk_score=score_resp.risk_score,
        decision=score_resp.decision,
        baseline=round(shap_result["baseline"], 4),
        top_features=tf,
        explanation=shap_result["explanation"],
        model_version=_state.model_version,
    )


@app.get("/audit/{transaction_id}", response_model=AuditResponse)
async def get_audit(transaction_id: str):
    """Retrieve all audit entries for a transaction."""
    if _state.audit_log is None:
        raise HTTPException(status_code=503, detail="Audit log not initialised.")
    entries = _state.audit_log.get(transaction_id)
    return AuditResponse(
        found=bool(entries),
        entries=[e.to_dict() for e in entries],
    )


@app.get("/health", response_model=HealthResponse)
async def health():
    status = "ok"
    if _state.mode == OperatingMode.DEGRADED:
        status = "degraded"
    elif not _state.is_model_loaded():
        status = "degraded"
    return HealthResponse(
        status=status,
        mode=_state.mode,
        model_loaded=_state.is_model_loaded(),
        uptime_s=round(time.time() - _state.start_time, 1),
    )


@app.get("/metrics", response_model=MetricsResponse)
async def metrics():
    return MetricsResponse(
        total_scored=_state.total_scored,
        decisions=_state.decision_counts,
        p50_ms=round(_state.percentile(50), 2),
        p99_ms=round(_state.percentile(99), 2),
        shadow_mode=_state.mode == OperatingMode.SHADOW,
        drift_alert=_state.drift_alert,
        drift_report=_state.last_drift_report,
    )


@app.post("/mode/{new_mode}")
async def set_mode(new_mode: OperatingMode):
    """Hot-switch the operating mode without restart."""
    _state.mode = new_mode
    return {"mode": new_mode, "message": f"Mode switched to {new_mode}."}