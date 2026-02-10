from __future__ import annotations

import json
import os
import re
from math import isfinite
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import httpx


class DecisionAIError(RuntimeError):
    pass


@dataclass(frozen=True)
class DecisionAIConfig:
    """
    OpenAI-compatible chat completion config.

    Env defaults:
      - OPENAI_API_KEY (required unless api_key passed explicitly)
      - OPENAI_BASE_URL (optional; default: https://api.openai.com/v1)
      - DECISION_AI_MODEL (optional; falls back to OPENAI_MODEL then gpt-4o-mini)
      - DECISION_AI_MAX_TOKENS (default: 500)
      - DECISION_AI_TEMPERATURE (default: 0.0)
      - DECISION_AI_TIMEOUT_S (default: 45)
    """

    api_key: str
    base_url: str = "https://api.openai.com/v1"
    model: str = "gpt-4o-mini"
    max_tokens: int = 500
    temperature: float = 0.0
    timeout_s: float = 45.0

    @staticmethod
    def from_env(*, api_key: str | None = None) -> "DecisionAIConfig":
        key = (api_key or os.getenv("OPENAI_API_KEY") or "").strip()
        if not key:
            raise DecisionAIError("OPENAI_API_KEY is not set (and api_key was not provided).")
        base_url = (os.getenv("OPENAI_BASE_URL") or "https://api.openai.com/v1").strip().rstrip("/")
        model = (os.getenv("DECISION_AI_MODEL") or os.getenv("OPENAI_MODEL") or "gpt-4o-mini").strip()
        max_tokens = int(os.getenv("DECISION_AI_MAX_TOKENS", "500"))
        temperature = float(os.getenv("DECISION_AI_TEMPERATURE", "0.0"))
        timeout_s = float(os.getenv("DECISION_AI_TIMEOUT_S", "45"))
        return DecisionAIConfig(
            api_key=key,
            base_url=base_url,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            timeout_s=timeout_s,
        )


def _get_field(opt: Any, key: str) -> Any:
    if isinstance(opt, Mapping):
        if key in opt:
            return opt[key]
        raise DecisionAIError(f"Option missing required field {key!r}: {opt!r}")
    if hasattr(opt, key):
        return getattr(opt, key)
    raise DecisionAIError(f"Option missing required field {key!r}: {opt!r}")


def _get_optional(opt: Any, key: str, default: Any) -> Any:
    if isinstance(opt, Mapping):
        return opt.get(key, default)
    if hasattr(opt, key):
        return getattr(opt, key)
    return default


def _get_optional_number(opt: Any, key: str, default: float = 0.0) -> float:
    val = _get_optional(opt, key, default)
    try:
        return float(val)
    except Exception:
        return float(default)


def _option_to_dict(opt: Any) -> dict[str, Any]:
    """
    Normalize an option (dict or object) into a JSON-safe dict used for prompting.
    """
    return {
        "option_id": _get_field(opt, "option_id"),
        "action": _get_field(opt, "action"),
        "parameters": _get_field(opt, "parameters"),
        "order_qty": _get_optional_number(opt, "order_qty", 0.0),
        "expedite_now_qty": _get_optional_number(opt, "expedite_now_qty", 0.0),
        "total_cost": float(_get_field(opt, "total_cost")),
        "stockout_risk": float(_get_field(opt, "stockout_risk")),
        "projected_stockout_units": _get_optional_number(opt, "projected_stockout_units", 0.0),
        "service_level": float(_get_field(opt, "service_level")),
        "ending_inventory": float(_get_field(opt, "ending_inventory")),
        "inventory_band": _get_optional(opt, "inventory_band", None),
        # Shared context (duplicated across options when present)
        "forecast_demand": _get_optional_number(opt, "forecast_demand", 0.0),
        "beginning_on_hand": _get_optional_number(opt, "beginning_on_hand", 0.0),
        "lead_time_months": _get_optional_number(opt, "lead_time_months", 0.0),
        "safety_stock": _get_optional_number(opt, "safety_stock", 0.0),
        "target_level": _get_optional_number(opt, "target_level", 0.0),
        "service_level_target": _get_optional_number(opt, "service_level_target", 0.0),
        # Precomputed deltas vs baseline option (so the LLM can cite changes without calculating).
        "delta_order_qty": _get_optional_number(opt, "delta_order_qty", 0.0),
        "delta_order_qty_abs": _get_optional_number(opt, "delta_order_qty_abs", 0.0),
        "delta_total_cost": _get_optional_number(opt, "delta_total_cost", 0.0),
        "delta_total_cost_abs": _get_optional_number(opt, "delta_total_cost_abs", 0.0),
        "delta_stockout_risk": _get_optional_number(opt, "delta_stockout_risk", 0.0),
        "delta_stockout_risk_abs": _get_optional_number(opt, "delta_stockout_risk_abs", 0.0),
        "delta_projected_stockout_units": _get_optional_number(opt, "delta_projected_stockout_units", 0.0),
        "delta_projected_stockout_units_abs": _get_optional_number(opt, "delta_projected_stockout_units_abs", 0.0),
        "delta_service_level": _get_optional_number(opt, "delta_service_level", 0.0),
        "delta_service_level_abs": _get_optional_number(opt, "delta_service_level_abs", 0.0),
        "decision_score": float(_get_field(opt, "decision_score")),
    }


_NUM_RE = re.compile(r"(?<![A-Za-z])[-+]?\d{1,3}(?:,\d{3})*(?:\.\d+)?%?")


def _expand_allowed_numbers(values: Sequence[float]) -> list[float]:
    """
    Expand allowed numbers to tolerate common formatting/rounding:
      - raw values
      - rounded to 0..3 decimals
      - for 0..1 values: also allow percent form (x*100), rounded to 0..2 decimals
    """
    out: list[float] = []
    for v in values:
        try:
            fv = float(v)
        except Exception:
            continue
        out.append(fv)

        # Add multiple rounding variants:
        # - Python round (banker's)
        # - Decimal ROUND_HALF_UP (more "human" / spreadsheet-like)
        for d in (0, 1, 2, 3):
            out.append(round(fv, d))
            try:
                q = Decimal("1") if d == 0 else Decimal("1").scaleb(-d)
                out.append(float(Decimal(str(fv)).quantize(q, rounding=ROUND_HALF_UP)))
            except (InvalidOperation, ValueError):
                pass

        if 0.0 <= fv <= 1.0:
            pct = fv * 100.0
            out.append(pct)
            for d in (0, 1, 2):
                out.append(round(pct, d))
                try:
                    q = Decimal("1") if d == 0 else Decimal("1").scaleb(-d)
                    out.append(float(Decimal(str(pct)).quantize(q, rounding=ROUND_HALF_UP)))
                except (InvalidOperation, ValueError):
                    pass

    # Allow small ordinals (common list numbering / month counts) so the narrator
    # doesn't fail validation for harmless "1)", "2)" formatting.
    out.extend(float(i) for i in range(0, 13))
    # De-dup while preserving order
    seen = set()
    uniq: list[float] = []
    for x in out:
        # normalize -0.0
        if x == 0.0:
            x = 0.0
        if x in seen:
            continue
        seen.add(x)
        uniq.append(x)
    return uniq


def _extract_numbers(text: str) -> list[tuple[str, float, bool]]:
    """
    Return [(token, value, is_percent_token)] from a string.
    """
    if not isinstance(text, str) or not text:
        return []
    hits: list[tuple[str, float, bool]] = []
    for m in _NUM_RE.finditer(text):
        tok = m.group(0)
        if not tok or tok in ("-", "+"):
            continue
        is_pct = tok.endswith("%")
        raw = tok[:-1] if is_pct else tok
        try:
            val = float(raw.replace(",", ""))
        except Exception:
            continue
        hits.append((tok, val, is_pct))
    return hits


def _is_allowed_number(val: float, allowed: Sequence[float]) -> bool:
    for a in allowed:
        try:
            af = float(a)
        except Exception:
            continue
        if abs(val - af) <= 1e-9:
            return True
        # Relative tolerance for large numbers
        if abs(af) > 1e-9 and abs(val - af) / abs(af) <= 1e-6:
            return True
    return False


def _validate_llm_json(
    obj: Any,
    *,
    valid_option_ids: set[str],
    allowed_numbers: Sequence[float],
) -> dict[str, Any]:
    if not isinstance(obj, dict):
        raise DecisionAIError("LLM did not return a JSON object.")

    required = {"recommended_option_id", "decision_statement", "expected_impact", "tradeoffs", "confidence"}
    extra = set(obj.keys()) - required
    missing = required - set(obj.keys())
    if missing:
        raise DecisionAIError(f"LLM JSON missing keys: {sorted(missing)}")
    if extra:
        raise DecisionAIError(f"LLM JSON has extra keys (not allowed): {sorted(extra)}")

    reco = obj.get("recommended_option_id")
    if not isinstance(reco, str) or not reco.strip():
        raise DecisionAIError("recommended_option_id must be a non-empty string.")
    if reco not in valid_option_ids:
        raise DecisionAIError(f"recommended_option_id={reco!r} is not one of the provided option_id values.")

    for k in ("decision_statement", "expected_impact"):
        if not isinstance(obj.get(k), str) or not obj[k].strip():
            raise DecisionAIError(f"{k} must be a non-empty string.")

    tradeoffs = obj.get("tradeoffs")
    if not isinstance(tradeoffs, list) or not tradeoffs:
        raise DecisionAIError("tradeoffs must be a non-empty list of strings.")
    if not all(isinstance(t, str) and t.strip() for t in tradeoffs):
        raise DecisionAIError("tradeoffs must be a non-empty list of strings.")

    conf = obj.get("confidence")
    try:
        conf_f = float(conf)
    except Exception as e:
        raise DecisionAIError("confidence must be a number between 0 and 1.") from e
    if not (0.0 <= conf_f <= 1.0):
        raise DecisionAIError("confidence must be between 0 and 1.")
    obj["confidence"] = conf_f

    # Ensure the model didn't invent any numbers in free-text fields.
    # If a number is mentioned, it must match a number from the provided option inputs
    # (allowing common rounding/percent formatting).
    expanded_allowed = _expand_allowed_numbers(list(allowed_numbers) + [conf_f])
    text_fields: list[str] = [obj["decision_statement"], obj["expected_impact"], *obj["tradeoffs"]]
    invented: list[str] = []
    for t in text_fields:
        for tok, val, is_pct in _extract_numbers(t):
            # If token is a percent, allow either "20%" matching 20 or matching 0.2 depending on what was provided.
            candidates = [val / 100.0, val] if is_pct else [val]
            ok = any(_is_allowed_number(c, expanded_allowed) for c in candidates)
            if not ok:
                invented.append(tok)
    if invented:
        uniq = sorted(set(invented), key=invented.index)
        raise DecisionAIError(
            "LLM invented numeric values in text fields (not allowed). "
            f"Unexpected numbers: {uniq[:12]}{'...' if len(uniq) > 12 else ''}"
        )

    return obj


async def recommend_decision_option(
    scored_options: Sequence[Any],
    service_level_target: float,
    *,
    config: DecisionAIConfig | None = None,
    user_question: str | None = None,
) -> dict[str, Any]:
    """
    Choose exactly ONE recommended option using an OpenAI-compatible LLM.

    Requirements enforced:
      - Deterministic inputs (no tool use / no external data)
      - Output must be strict JSON with required keys only
      - LLM must not calculate numbers; it must cite provided values

    Input:
      scored_options: sequence of options, each with required fields INCLUDING decision_score
      service_level_target: target service level for the decision
      user_question: optional follow-up context from a user (kept short)
    """
    if not isinstance(scored_options, Sequence):
        raise TypeError("scored_options must be a sequence.")
    if len(scored_options) == 0:
        raise ValueError("scored_options must be non-empty.")

    cfg = config or DecisionAIConfig.from_env()
    try:
        slt = float(service_level_target)
    except Exception as e:
        raise ValueError("service_level_target must be numeric.") from e
    if not (0.0 <= slt <= 1.0):
        raise ValueError("service_level_target must be between 0 and 1.")

    options_payload = [_option_to_dict(o) for o in scored_options]
    valid_ids = {str(o["option_id"]) for o in options_payload}
    allowed_numbers: list[float] = [float(slt)]
    for o in options_payload:
        for k in (
            "order_qty",
            "expedite_now_qty",
            "total_cost",
            "stockout_risk",
            "projected_stockout_units",
            "service_level",
            "ending_inventory",
            "decision_score",
            "forecast_demand",
            "beginning_on_hand",
            "lead_time_months",
            "safety_stock",
            "target_level",
            "service_level_target",
            "delta_order_qty",
            "delta_order_qty_abs",
            "delta_total_cost",
            "delta_total_cost_abs",
            "delta_stockout_risk",
            "delta_stockout_risk_abs",
            "delta_projected_stockout_units",
            "delta_projected_stockout_units_abs",
            "delta_service_level",
            "delta_service_level_abs",
        ):
            try:
                allowed_numbers.append(float(o[k]))
            except Exception:
                continue
        # Also allow numeric scalars inside parameters (e.g., effective_lead_time_months)
        params = o.get("parameters") if isinstance(o, dict) else None
        if isinstance(params, Mapping):
            for v in params.values():
                if isinstance(v, (int, float)) and isfinite(float(v)):
                    allowed_numbers.append(float(v))

    # Keep prompt compact and explicit.
    prompt_obj = {
        "service_level_target": slt,
        "options": options_payload,
    }

    user_q = (user_question or "").strip()
    if len(user_q) > 800:
        user_q = user_q[:800] + "..."

    system_instructions = (
        "You are a decision assistant. Your job is to choose exactly ONE option_id from the provided options.\n"
        "\n"
        "Hard rules:\n"
        "- Output MUST be strict JSON (single object) with EXACTLY these keys:\n"
        "  recommended_option_id, decision_statement, expected_impact, tradeoffs, confidence\n"
        "- DO NOT output markdown. DO NOT output any text outside the JSON object.\n"
        "- You MUST NOT calculate numbers. You may ONLY quote/cite numbers that already appear in the provided JSON.\n"
        "- You MUST NOT invent costs, savings, percentages, or service levels.\n"
        "- Use the service_level_target as a hard constraint: options below it are strongly discouraged.\n"
        "- Use decision_score only as a ranking hint; still explain trade-offs using the given numeric fields.\n"
        "- When justifying the choice, prefer citing provided fields like projected_stockout_units, order_qty, total_cost, and service_level.\n"
        "- Do NOT mention dates/years or any numbers that are not present in the provided JSON. If needed, say 'the selected period'.\n"
        "- Do NOT mention unchosen option_ids; focus only on the chosen option.\n"
        "- If you need to talk about 'change vs baseline', use the provided delta_* fields (e.g., delta_projected_stockout_units_abs). Do NOT compute differences.\n"
        "\n"
        "Output schema:\n"
        "{\n"
        '  "recommended_option_id": "A",\n'
        '  "decision_statement": "...",\n'
        '  "expected_impact": "...",\n'
        '  "tradeoffs": ["...", "..."],\n'
        '  "confidence": 0.0\n'
        "}\n"
    )

    user_message = (
        "Choose the best single option and explain trade-offs using provided numbers.\n"
        f"USER_QUESTION (optional): {user_q if user_q else '(none)'}\n\n"
        f"DECISION_INPUT_JSON:\n{json.dumps(prompt_obj, ensure_ascii=False)}"
    )

    url = f"{cfg.base_url}/chat/completions"
    headers = {"Authorization": f"Bearer {cfg.api_key}", "Content-Type": "application/json"}

    async def _call_llm(*, sys_msg: str, user_msg: str) -> str:
        payload = {
            "model": cfg.model,
            "messages": [
                {"role": "system", "content": sys_msg},
                {"role": "user", "content": user_msg},
            ],
            "max_tokens": int(cfg.max_tokens),
            "temperature": float(cfg.temperature),
            # Many OpenAI-compatible servers support this; if unsupported they may ignore it.
            "response_format": {"type": "json_object"},
        }

        async with httpx.AsyncClient(timeout=cfg.timeout_s) as client:
            res = await client.post(url, headers=headers, json=payload)
            try:
                res.raise_for_status()
            except httpx.HTTPStatusError as e:
                detail = (e.response.text or "").strip()
                raise DecisionAIError(f"LLM request failed: HTTP {e.response.status_code}. {detail[:800]}") from e

        data = res.json()
        content = ((data.get("choices") or [{}])[0].get("message") or {}).get("content")
        if not isinstance(content, str) or not content.strip():
            raise DecisionAIError("LLM returned an empty response.")
        return content

    async def _parse_and_validate(content: str) -> dict[str, Any]:
        try:
            obj = json.loads(content)
        except Exception as e:
            raise DecisionAIError(f"LLM did not return valid JSON. Raw content: {content[:400]!r}") from e
        return _validate_llm_json(obj, valid_option_ids=valid_ids, allowed_numbers=allowed_numbers)

    # Attempt 1: allow numeric citations (strictly validated).
    try:
        content1 = await _call_llm(sys_msg=system_instructions, user_msg=user_message)
        return await _parse_and_validate(content1)
    except DecisionAIError as e:
        # Common failure mode: LLM invents numbers. Retry once with "no digits" rule.
        msg = str(e)
        if "invented numeric values" not in msg:
            raise

    system_instructions_no_digits = (
        system_instructions
        + "\n"
        + "Retry rules:\n"
        + "- Do NOT include any digits 0-9 anywhere in decision_statement, expected_impact, or tradeoffs.\n"
        + "- If you need to refer to numbers, refer to them qualitatively (e.g., 'reduces stockouts') without digits.\n"
    )
    content2 = await _call_llm(sys_msg=system_instructions_no_digits, user_msg=user_message)
    return await _parse_and_validate(content2)
