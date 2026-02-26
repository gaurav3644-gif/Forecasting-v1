from fastapi import APIRouter, Body, Request
from fastapi import HTTPException
from pydantic import BaseModel
from typing import Dict, List, Optional
from dataclasses import asdict, is_dataclass
import json
import logging
import os

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


# ── Decision Engine: Intent Router + Query ─────────────────────────────────────

_INTENT_SYSTEM_PROMPT = """You are a supply chain analytics router.
Classify the user question into exactly ONE intent:
  - volatility_analysis   : instability, variance, erratic patterns, which region/SKU fluctuates
  - forecast_performance  : forecast accuracy, MAPE, bias, over/under forecasting, forecast quality
  - inventory_risk        : stockout, overstock, safety stock breach, inventory risk, out-of-stock

Also extract filters if mentioned.

Return ONLY valid JSON — no markdown, no explanation:
{
  "intent": "volatility_analysis" | "forecast_performance" | "inventory_risk",
  "filters": {
    "dimension": "region" | "sku",
    "value": null,
    "start_date": null,
    "end_date": null,
    "sku": null,
    "region": null,
    "future_periods": 4
  }
}"""

_INTENT_KEYWORDS = {
    "volatility_analysis":  ["volatil", "varianc", "unstable", "erratic", "fluctuat", "instab", "spike", "spiky"],
    "forecast_performance": ["mape", "accuracy", "accurate", "bias", "over-forecast", "underforecast",
                             "over forecast", "forecast quality", "forecast error", "forecast accuracy"],
}


async def _route_intent(message: str) -> dict:
    """
    LLM-based intent classification (classification only — no creativity).
    Falls back to keyword matching if OpenAI is unavailable.
    """
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if api_key:
        try:
            import openai  # type: ignore
            client = openai.AsyncOpenAI(api_key=api_key,
                                        base_url=os.environ.get("OPENAI_BASE_URL") or None)
            resp = await client.chat.completions.create(
                model=os.environ.get("DECISION_AI_MODEL", "gpt-4o-mini"),
                messages=[
                    {"role": "system", "content": _INTENT_SYSTEM_PROMPT},
                    {"role": "user",   "content": message},
                ],
                max_tokens=200,
                temperature=0,
            )
            raw = resp.choices[0].message.content.strip()
            result = json.loads(raw)
            if result.get("intent") in ("volatility_analysis", "forecast_performance", "inventory_risk"):
                return result
        except Exception as e:
            logging.debug(f"[decision/query] intent LLM failed, using keywords: {e}")

    # Keyword fallback
    msg_lower = message.lower()
    for intent, kws in _INTENT_KEYWORDS.items():
        if any(kw in msg_lower for kw in kws):
            return {"intent": intent, "filters": {"dimension": "sku", "future_periods": 4}}
    return {"intent": "inventory_risk", "filters": {"future_periods": 4}}


_EXPLAIN_CONTEXT = {
    "volatility_analysis": (
        "Explain: which dimension drives instability, the planning implication, "
        "and where to stabilize. Be specific about the top contributor."
    ),
    "forecast_performance": (
        "Explain: whether bias is systematic, what direction (over/under), "
        "and what planning correction is needed."
    ),
    "inventory_risk": (
        "Explain: why risk exists, what the planner should do immediately, "
        "and the urgency level."
    ),
}


async def _explain_output(question: str, intent: str, engine_output: dict) -> str:
    """
    LLM explanation layer — uses only numbers from the structured engine output.
    Falls back to plain-text summary if OpenAI is unavailable.
    """
    output_json = json.dumps(engine_output, indent=2)
    context_instruction = _EXPLAIN_CONTEXT.get(intent, "Provide a concise business explanation.")

    system_prompt = (
        f"You are a supply chain decision assistant.\n"
        f"The user asked: \"{question}\"\n\n"
        f"The analytics engine returned:\n{output_json}\n\n"
        f"{context_instruction}\n\n"
        f"Rules:\n"
        f"- Use ONLY numbers from the engine output above. Do NOT invent figures.\n"
        f"- 3–5 sentences maximum.\n"
        f"- End with a concrete, actionable recommendation.\n"
        f"- Be direct and business-focused."
    )

    api_key = os.environ.get("OPENAI_API_KEY", "")
    if api_key:
        try:
            import openai  # type: ignore
            client = openai.AsyncOpenAI(api_key=api_key,
                                        base_url=os.environ.get("OPENAI_BASE_URL") or None)
            resp = await client.chat.completions.create(
                model=os.environ.get("DECISION_AI_MODEL", "gpt-4o-mini"),
                messages=[{"role": "user", "content": system_prompt}],
                max_tokens=300,
                temperature=0.3,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            logging.debug(f"[decision/query] explanation LLM failed: {e}")

    # Plain-text fallback
    rl    = engine_output.get("risk_level", "Unknown")
    at    = engine_output.get("analysis_type", intent)
    score = engine_output.get("volatility_score") or engine_output.get("risk_score") or engine_output.get("forecast_error_score")
    score_str = f" Score: {score}/100." if score is not None else ""
    return f"Analysis complete ({at}).{score_str} Risk level: {rl}. Review the structured output above for details."


def _resolve_run_session(request: Request, run_session_id: str) -> Optional[dict]:
    """
    Resolve the per-run session dict from app.state.data_store.
    Tries email-keyed path first, then falls back to direct key lookup.
    """
    store = getattr(request.app.state, "data_store", {})
    if not isinstance(store, dict):
        return None

    # Primary path: store[email]["runs"][run_session_id]
    email = getattr(getattr(request, "state", None), "user_email", None)
    if email:
        email_norm = str(email).strip().lower()
        for key in (f"user:{email_norm}", email_norm, email):
            user_store = store.get(key)
            if isinstance(user_store, dict):
                run = user_store.get("runs", {}).get(run_session_id)
                if isinstance(run, dict):
                    return run

    # Secondary: flat store[run_session_id]
    flat = store.get(run_session_id)
    if isinstance(flat, dict):
        return flat

    # Tertiary: any user's runs (last resort — single-user dev mode)
    for v in store.values():
        if isinstance(v, dict):
            run = v.get("runs", {}).get(run_session_id)
            if isinstance(run, dict):
                return run

    return None


@router.post("/decision/query", tags=["Decision Intelligence"])
async def decision_query(request: Request, payload: Dict = Body(...)):
    """
    Decision Engine: NL question → intent classification → deterministic engine → LLM explanation.

    Flow:
      1. LLM classifies intent → volatility_analysis | forecast_performance | inventory_risk
      2. Deterministic Python engine runs against session DataFrames
      3. LLM explains the structured output (no number invention)
      4. Returns {intent, filters, engine_output, explanation}
    """
    from decision_engine import volatility_engine, forecast_engine, risk_engine

    message        = str(payload.get("message") or "").strip()
    run_session_id = str(payload.get("run_session_id") or "").strip()
    combo_key      = str(payload.get("combo_key") or "").strip() or None

    if not message:
        return {"error": "No question provided."}

    # Resolve the run session
    session = _resolve_run_session(request, run_session_id) if run_session_id else None
    if not session:
        # Try the first available run (single-user dev mode)
        store = getattr(request.app.state, "data_store", {})
        for v in store.values():
            if isinstance(v, dict):
                for run in v.get("runs", {}).values():
                    if isinstance(run, dict):
                        session = run
                        break
            if session:
                break

    if not session:
        return {"error": "No forecast session found. Run a forecast first."}

    # Step 1: Intent routing (LLM classification only)
    intent_result = await _route_intent(message)
    intent  = str(intent_result.get("intent") or "inventory_risk")
    filters = dict(intent_result.get("filters") or {})

    # Merge combo_key into filters (sku + region)
    if combo_key:
        parts = combo_key.split("|||")
        if len(parts) >= 1 and parts[0] and not filters.get("sku"):
            filters["sku"] = parts[0]
        if len(parts) >= 2 and parts[1] and not filters.get("region"):
            filters["region"] = parts[1]

    # Step 2: Deterministic engine
    if intent == "volatility_analysis":
        engine_output = volatility_engine(session, filters)
    elif intent == "forecast_performance":
        engine_output = forecast_engine(session, filters)
    else:
        engine_output = risk_engine(session, filters)

    if "error" in engine_output:
        return {
            "intent":        intent,
            "filters":       filters,
            "engine_output": engine_output,
            "explanation":   engine_output["error"],
        }

    # Step 3: LLM explanation layer
    explanation = await _explain_output(message, intent, engine_output)

    return {
        "intent":        intent,
        "filters":       filters,
        "engine_output": engine_output,
        "explanation":   explanation,
    }
