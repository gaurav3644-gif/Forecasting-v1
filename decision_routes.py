from fastapi import APIRouter, Request
from fastapi import HTTPException
from pydantic import BaseModel
from typing import Dict, List, Optional
from dataclasses import asdict, is_dataclass

from decision_models import DecisionContext
from option_engine import generate_decision_options
from option_engine import generate_decision_options
from lead_time_option_selector import generate_context_valid_options




from option_scoring import compute_decision_option_scores, score_decision_options
from decision_ai import DecisionAIError, recommend_decision_option
from decision_context_builder import DecisionContextBuildError, build_decision_context

router = APIRouter()


def _narrator_action_label(action: str) -> str:
    a = (action or "").strip()
    if a == "follow_policy":
        return "Follow policy"
    if a == "expedite_now":
        return "Aggressive buffer (expedite now)"
    if a == "capacity_capped":
        return "Capacity capped"
    if a == "aggressive_buffer":
        return "Aggressive buffer"
    return a or "Option"


def _deterministic_ai_fallback(scored_options: list[dict], service_level_target: float, *, reason: str) -> dict:
    """
    If the LLM output is invalid (common: it invents numbers), return a deterministic
    narration so the UI doesn't error.

    This keeps the same response shape as /ai-decision.
    """
    if not scored_options:
        return {
            "recommended_option_id": "A",
            "decision_statement": "Choose Option A.",
            "expected_impact": "No options were available to evaluate.",
            "tradeoffs": ["Unable to generate options; run supply plan first."],
            "confidence": 0.2,
        }

    # scored_options is already ranked best->worst in /ai-decision
    best = scored_options[0]
    best_id = str(best.get("option_id") or "").strip() or "A"
    best_action = _narrator_action_label(str(best.get("action") or ""))

    base = next((o for o in scored_options if str(o.get("option_id")) == "A"), None) or scored_options[-1]

    def _num(o: dict, k: str, default: float = 0.0) -> float:
        try:
            return float(o.get(k, default))
        except Exception:
            return float(default)

    # Prefer precomputed deltas; if missing, fall back to absolute values only.
    stockout_reduction = _num(best, "delta_projected_stockout_units_abs", 0.0)
    order_qty = _num(best, "order_qty", 0.0)
    service_level = _num(best, "service_level", 0.0)
    ending_inventory = _num(best, "ending_inventory", 0.0)
    base_stockout = _num(base, "projected_stockout_units", 0.0)
    sl_target = _num(best, "service_level_target", service_level_target)

    decision_statement = f"Choose Option {best_id} – {best_action}."
    expected_impact = (
        f"Reduces projected stock-out by {stockout_reduction:.0f} units "
        f"(baseline stock-out {base_stockout:.0f} units), with order quantity {order_qty:.0f} units "
        f"and service level {service_level*100.0:.0f}% (target {sl_target*100.0:.0f}%)."
    )
    tradeoffs = [
        f"Ending inventory projected at {ending_inventory:.0f} units for the selected period.",
        f"Narration fallback used because LLM output failed validation: {reason[:120]}",
    ]

    return {
        "recommended_option_id": best_id,
        "decision_statement": decision_statement,
        "expected_impact": expected_impact,
        "tradeoffs": tradeoffs,
        "confidence": 0.6,
    }

class GenerateOptionsRequest(BaseModel):
    sku: str
    period: str
    # Optional "connected" parameters: if omitted, we build DecisionContext from forecast/supply artifacts.
    location: Optional[str] = None
    session_id: str = "default"
    service_level_target: Optional[float] = None
    narration_guidance: Optional[str] = None


def _get_session(request: Request, session_id: str) -> dict:
    store = getattr(request.app.state, "data_store", None)
    if not isinstance(store, dict):
        raise HTTPException(status_code=500, detail="Server is missing app.state.data_store.")
    sess = store.get(session_id)
    if not isinstance(sess, dict):
        raise HTTPException(status_code=400, detail=f"Unknown session_id={session_id!r}. Upload data and run forecast first.")
    return sess


def _build_ctx_from_req_or_session(request: Request, req: GenerateOptionsRequest) -> DecisionContext:
    # Connected mode: build from forecast/supply artifacts.
    session = _get_session(request, req.session_id)
    try:
        return build_decision_context(
            session,
            sku=req.sku,
            period=req.period,
            location=req.location,
            service_level_target=req.service_level_target,
        )
    except DecisionContextBuildError as e:
        raise HTTPException(status_code=400, detail=str(e))

class DecisionOptionResponse(BaseModel):
    option_id: str
    action: str
    parameters: Dict
    order_qty: float
    expedite_now_qty: float
    total_cost: float
    stockout_risk: float
    projected_stockout_units: float
    service_level: float
    ending_inventory: float
    inventory_band: Optional[str] = None

class ScoredDecisionOptionResponse(DecisionOptionResponse):
    decision_score: float


class GenerateOptionsResponse(BaseModel):
    sku: str
    period: str
    issue_detected: str
    options: List[DecisionOptionResponse]

class ScoreOptionsResponse(BaseModel):
    sku: str
    period: str
    location: Optional[str] = None
    issue_detected: str
    service_level_target: float
    forecast_demand: float
    beginning_on_hand: float
    lead_time_months: int
    safety_stock: float
    target_level: float
    options: List[ScoredDecisionOptionResponse]

class AIDecisionResponse(BaseModel):
    recommended_option_id: str
    decision_statement: str
    expected_impact: str
    tradeoffs: List[str]
    confidence: float


class DecisionDeltaResponse(BaseModel):
    delta_order_qty: float
    delta_order_qty_pct: Optional[float] = None
    delta_total_cost: float
    delta_stockout_risk: float
    delta_service_level: float


class DecisionRecommendationResponse(BaseModel):
    sku: str
    period: str
    location: Optional[str] = None
    service_level_target: float

    baseline: ScoredDecisionOptionResponse
    recommended: ScoredDecisionOptionResponse
    deltas: DecisionDeltaResponse

@router.post(
    "/generate-options",
    response_model=GenerateOptionsResponse,
    tags=["Decision Intelligence"]
)
def generate_options(request: Request, req: GenerateOptionsRequest):
    ctx = _build_ctx_from_req_or_session(request, req)

    #options = generate_decision_options(ctx)
    options = generate_context_valid_options(ctx)

    issue = "stockout_risk" if any(o.stockout_risk > 0.1 for o in options) else "none"

    return GenerateOptionsResponse(
        sku=req.sku,
        period=req.period,
        issue_detected=issue,
        options=[asdict(o) if is_dataclass(o) else dict(o) for o in options]
    )


@router.post(
    "/score-options",
    response_model=ScoreOptionsResponse,
    tags=["Decision Intelligence"]
)
def score_options(request: Request, req: GenerateOptionsRequest):
    """
    Generate decision options and score them deterministically (no LLM).
    """
    ctx = _build_ctx_from_req_or_session(request, req)

    options = generate_decision_options(ctx)
    slt = float(req.service_level_target) if req.service_level_target is not None else float(ctx.service_level_target)
    scored_order = score_decision_options(options, slt)
    scores = compute_decision_option_scores(options, slt)
    score_by_option_id = {str(getattr(o, "option_id", "")): float(s) for o, s in zip(options, scores)}

    issue = "stockout_risk" if any(getattr(o, "stockout_risk", 0) > 0.1 for o in options) else "none"

    scored_options: list[dict] = []
    for opt in scored_order:
        if is_dataclass(opt):
            payload = asdict(opt)
        elif isinstance(opt, dict):
            payload = dict(opt)
        else:
            payload = {
                "option_id": getattr(opt, "option_id"),
                "action": getattr(opt, "action"),
                "parameters": getattr(opt, "parameters"),
                "order_qty": getattr(opt, "order_qty", 0.0),
                "expedite_now_qty": getattr(opt, "expedite_now_qty", 0.0),
                "total_cost": getattr(opt, "total_cost"),
                "stockout_risk": getattr(opt, "stockout_risk"),
                "projected_stockout_units": getattr(opt, "projected_stockout_units", 0.0),
                "service_level": getattr(opt, "service_level"),
                "ending_inventory": getattr(opt, "ending_inventory"),
            }

        # Add a simple inventory band for display (relative to safety_stock/target_level).
        try:
            ending_inv = float(payload.get("ending_inventory", 0.0) or 0.0)
            ss = float(getattr(ctx, "safety_stock", 0.0) or 0.0)
            tl = float(getattr(ctx, "target_level", 0.0) or 0.0)
            if ending_inv < ss:
                payload["inventory_band"] = "Low"
            elif tl > 0 and ending_inv < tl:
                payload["inventory_band"] = "Medium"
            else:
                payload["inventory_band"] = "High"
        except Exception:
            payload["inventory_band"] = None

        payload["decision_score"] = score_by_option_id.get(str(payload.get("option_id", "")), 0.0)
        scored_options.append(payload)

    return ScoreOptionsResponse(
        sku=req.sku,
        period=req.period,
        location=req.location,
        issue_detected=issue,
        service_level_target=float(slt),
        forecast_demand=float(getattr(ctx, "forecast_demand", 0.0) or 0.0),
        beginning_on_hand=float(getattr(ctx, "beginning_on_hand", 0.0) or 0.0),
        lead_time_months=int(getattr(ctx, "lead_time_months", 1) or 1),
        safety_stock=float(getattr(ctx, "safety_stock", 0.0) or 0.0),
        target_level=float(getattr(ctx, "target_level", 0.0) or 0.0),
        options=scored_options,
    )


@router.post(
    "/decision/recommendation",
    response_model=DecisionRecommendationResponse,
    tags=["Decision Intelligence"],
)
def recommend_option(request: Request, req: GenerateOptionsRequest):
    """
    Deterministic (non-LLM) recommendation: generate + score options and return the best option plus deltas vs baseline.
    """
    ctx = _build_ctx_from_req_or_session(request, req)

    options = generate_decision_options(ctx)
    slt = float(req.service_level_target) if req.service_level_target is not None else float(ctx.service_level_target)

    ranked = score_decision_options(options, slt)
    scores = compute_decision_option_scores(options, slt)
    score_by_option_id = {str(getattr(o, "option_id", "")): float(s) for o, s in zip(options, scores)}

    baseline = next((o for o in options if str(getattr(o, "option_id", "")) == "A"), options[0])
    recommended = ranked[0]

    def _to_payload(opt) -> dict:
        if is_dataclass(opt):
            d = asdict(opt)
        elif isinstance(opt, dict):
            d = dict(opt)
        else:
            d = {
                "option_id": getattr(opt, "option_id"),
                "action": getattr(opt, "action"),
                "parameters": getattr(opt, "parameters"),
                "order_qty": getattr(opt, "order_qty", 0.0),
                "expedite_now_qty": getattr(opt, "expedite_now_qty", 0.0),
                "total_cost": getattr(opt, "total_cost"),
                "stockout_risk": getattr(opt, "stockout_risk"),
                "projected_stockout_units": getattr(opt, "projected_stockout_units", 0.0),
                "service_level": getattr(opt, "service_level"),
                "ending_inventory": getattr(opt, "ending_inventory"),
            }

        try:
            ending_inv = float(d.get("ending_inventory", 0.0) or 0.0)
            ss = float(getattr(ctx, "safety_stock", 0.0) or 0.0)
            tl = float(getattr(ctx, "target_level", 0.0) or 0.0)
            if ending_inv < ss:
                d["inventory_band"] = "Low"
            elif tl > 0 and ending_inv < tl:
                d["inventory_band"] = "Medium"
            else:
                d["inventory_band"] = "High"
        except Exception:
            d["inventory_band"] = None

        d["decision_score"] = score_by_option_id.get(str(d.get("option_id", "")), 0.0)
        return d

    base_p = _to_payload(baseline)
    rec_p = _to_payload(recommended)

    def _order_qty(p: dict) -> float:
        params = p.get("parameters") or {}
        if isinstance(params, dict) and "order_qty" in params:
            try:
                return float(params["order_qty"])
            except Exception:
                return 0.0
        return 0.0

    base_order = _order_qty(base_p)
    rec_order = _order_qty(rec_p)
    delta_order = float(rec_order - base_order)
    delta_order_pct = None
    if abs(base_order) > 1e-9:
        delta_order_pct = float((delta_order / base_order) * 100.0)

    deltas = {
        "delta_order_qty": round(delta_order, 4),
        "delta_order_qty_pct": round(delta_order_pct, 4) if delta_order_pct is not None else None,
        "delta_total_cost": round(float(rec_p.get("total_cost", 0.0)) - float(base_p.get("total_cost", 0.0)), 4),
        "delta_stockout_risk": round(float(rec_p.get("stockout_risk", 0.0)) - float(base_p.get("stockout_risk", 0.0)), 4),
        "delta_service_level": round(float(rec_p.get("service_level", 0.0)) - float(base_p.get("service_level", 0.0)), 4),
    }

    return DecisionRecommendationResponse(
        sku=req.sku,
        period=req.period,
        location=req.location,
        service_level_target=float(slt),
        baseline=base_p,  # type: ignore[arg-type]
        recommended=rec_p,  # type: ignore[arg-type]
        deltas=deltas,  # type: ignore[arg-type]
    )


@router.post(
    "/ai-decision",
    response_model=AIDecisionResponse,
    tags=["Decision Intelligence"],
)
async def ai_decision(request: Request, req: GenerateOptionsRequest):
    """
    Generate and score decision options, then ask an LLM to choose exactly ONE option.

    Returns ONLY the structured JSON decision from the LLM (validated server-side).
    """
    ctx = _build_ctx_from_req_or_session(request, req)

    options = generate_decision_options(ctx)
    slt = float(req.service_level_target) if req.service_level_target is not None else float(ctx.service_level_target)
    ranked = score_decision_options(options, slt)
    scores = compute_decision_option_scores(options, slt)
    score_by_option_id = {str(getattr(o, "option_id", "")): float(s) for o, s in zip(options, scores)}

    # Baseline (policy) option for delta fields (so the LLM can cite changes without calculating).
    baseline_opt = next((o for o in options if str(getattr(o, "option_id", "")) == "A"), options[0])
    base_order_qty = float(getattr(baseline_opt, "order_qty", 0.0) or 0.0)
    base_cost = float(getattr(baseline_opt, "total_cost", 0.0) or 0.0)
    base_stockout_risk = float(getattr(baseline_opt, "stockout_risk", 0.0) or 0.0)
    base_stockout_units = float(getattr(baseline_opt, "projected_stockout_units", 0.0) or 0.0)
    base_service_level = float(getattr(baseline_opt, "service_level", 0.0) or 0.0)

    scored_payload: list[dict] = []
    for opt in ranked:
        if is_dataclass(opt):
            d = asdict(opt)
        elif isinstance(opt, dict):
            d = dict(opt)
        else:
            d = {
                "option_id": getattr(opt, "option_id"),
                "action": getattr(opt, "action"),
                "parameters": getattr(opt, "parameters"),
                "order_qty": getattr(opt, "order_qty", 0.0),
                "expedite_now_qty": getattr(opt, "expedite_now_qty", 0.0),
                "total_cost": getattr(opt, "total_cost"),
                "stockout_risk": getattr(opt, "stockout_risk"),
                "projected_stockout_units": getattr(opt, "projected_stockout_units", 0.0),
                "service_level": getattr(opt, "service_level"),
                "ending_inventory": getattr(opt, "ending_inventory"),
            }

        try:
            ending_inv = float(d.get("ending_inventory", 0.0) or 0.0)
            ss = float(getattr(ctx, "safety_stock", 0.0) or 0.0)
            tl = float(getattr(ctx, "target_level", 0.0) or 0.0)
            if ending_inv < ss:
                d["inventory_band"] = "Low"
            elif tl > 0 and ending_inv < tl:
                d["inventory_band"] = "Medium"
            else:
                d["inventory_band"] = "High"
        except Exception:
            d["inventory_band"] = None

        # Add shared context fields so the narrator can cite them without calculating.
        d["forecast_demand"] = float(getattr(ctx, "forecast_demand", 0.0) or 0.0)
        d["beginning_on_hand"] = float(getattr(ctx, "beginning_on_hand", 0.0) or 0.0)
        d["lead_time_months"] = int(getattr(ctx, "lead_time_months", 1) or 1)
        d["safety_stock"] = float(getattr(ctx, "safety_stock", 0.0) or 0.0)
        d["target_level"] = float(getattr(ctx, "target_level", 0.0) or 0.0)
        d["service_level_target"] = float(slt)

        # Delta fields vs baseline policy option (A). Provide both signed and absolute magnitudes.
        try:
            oq = float(d.get("order_qty", 0.0) or 0.0)
            cost = float(d.get("total_cost", 0.0) or 0.0)
            sr = float(d.get("stockout_risk", 0.0) or 0.0)
            su = float(d.get("projected_stockout_units", 0.0) or 0.0)
            svc = float(d.get("service_level", 0.0) or 0.0)

            d["delta_order_qty"] = oq - base_order_qty
            d["delta_order_qty_abs"] = abs(oq - base_order_qty)

            d["delta_total_cost"] = cost - base_cost
            d["delta_total_cost_abs"] = abs(cost - base_cost)

            d["delta_stockout_risk"] = sr - base_stockout_risk
            d["delta_stockout_risk_abs"] = abs(sr - base_stockout_risk)

            d["delta_projected_stockout_units"] = su - base_stockout_units
            d["delta_projected_stockout_units_abs"] = abs(su - base_stockout_units)

            d["delta_service_level"] = svc - base_service_level
            d["delta_service_level_abs"] = abs(svc - base_service_level)
        except Exception:
            # If any numeric conversion fails, omit deltas; the narrator will fall back to absolute fields.
            pass

        d["decision_score"] = score_by_option_id.get(str(d.get("option_id", "")), 0.0)
        scored_payload.append(d)

    try:
        # Use the planning question (if any) as short narration guidance.
        user_q = None
        try:
            user_q = (getattr(req, "narration_guidance", None) or "").strip() or None
        except Exception:
            user_q = None
        decision = await recommend_decision_option(scored_payload, slt, user_question=user_q)
        return decision
    except DecisionAIError as e:
        # Includes invalid JSON / schema violations / invented numbers from the LLM.
        # Do not fail the UI; fall back deterministically.
        return _deterministic_ai_fallback(scored_payload, slt, reason=str(e))
    except Exception as e:
        return _deterministic_ai_fallback(scored_payload, slt, reason=f"AI decision failed: {e}")
