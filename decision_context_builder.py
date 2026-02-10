from __future__ import annotations

import os
from typing import Optional

import pandas as pd

from decision_models import DecisionContext


class DecisionContextBuildError(RuntimeError):
    pass


def _env_float(name: str, default: float) -> float:
    v = os.getenv(name)
    if v is None or not str(v).strip():
        return float(default)
    try:
        return float(v)
    except Exception as e:
        raise DecisionContextBuildError(f"Env var {name} must be numeric, got {v!r}") from e


def _parse_period_to_month_start(period: str) -> pd.Timestamp:
    if not isinstance(period, str) or not period.strip():
        raise DecisionContextBuildError("period must be a non-empty string (e.g., '2026-01' or '2026-01-01').")
    ts = pd.to_datetime(period, errors="coerce")
    if pd.isna(ts):
        raise DecisionContextBuildError(f"Could not parse period={period!r} as a date.")
    return ts.to_period("M").start_time


def _pick_forecast_col(df: pd.DataFrame) -> str:
    for col in ("forecast", "forecast_p60", "forecast_p50", "forecast_p90", "forecast_p30", "forecast_p10"):
        if col in df.columns:
            return col
    raise DecisionContextBuildError("forecast_df has no forecast column (expected forecast or forecast_pXX).")


def _maybe_filter_grain(df: pd.DataFrame, *, sku: str, location: Optional[str]) -> pd.DataFrame:
    out = df
    if "item" in out.columns:
        out = out[out["item"].astype(str) == str(sku)]
    elif "sku_id" in out.columns:
        out = out[out["sku_id"].astype(str) == str(sku)]

    if location is not None:
        if "store" in out.columns:
            out = out[out["store"].astype(str) == str(location)]
        elif "location" in out.columns:
            out = out[out["location"].astype(str) == str(location)]
    return out


def build_decision_context(
    session: dict,
    *,
    sku: str,
    period: str,
    location: Optional[str] = None,
    service_level_target: Optional[float] = None,
) -> DecisionContext:
    """
    Build DecisionContext from existing app session artifacts:
      - supply_plan_full_df (monthly) -> forecast + inventory + constraints + policy (already time-phased)

    This is the "bridge" that connects Forecast/Inventory/Supply -> Decision.
    """
    if not isinstance(session, dict):
        raise DecisionContextBuildError("session must be a dict.")

    month_start = _parse_period_to_month_start(period)
    supply_plan_full_df = session.get("supply_plan_full_df")
    if not isinstance(supply_plan_full_df, pd.DataFrame) or supply_plan_full_df.empty:
        raise DecisionContextBuildError("No supply_plan_full_df found in session. Generate a supply plan first.")

    sp = supply_plan_full_df.copy()
    if "period_start" not in sp.columns:
        raise DecisionContextBuildError("supply_plan_full_df is missing 'period_start'.")
    sp["period_start"] = pd.to_datetime(sp["period_start"], errors="coerce")
    sp = sp.dropna(subset=["period_start"])

    sp = _maybe_filter_grain(sp, sku=sku, location=location)
    sp = sp[sp["period_start"].dt.to_period("M").apply(lambda p: p.start_time) == month_start]
    if sp.empty:
        loc_msg = f", location={location!r}" if location is not None else ""
        raise DecisionContextBuildError(f"No supply plan rows for sku={sku!r}{loc_msg} in month {month_start.strftime('%Y-%m')}.")

    plan_row = sp.iloc[0].to_dict()

    def _num(key: str, default: float = 0.0) -> float:
        v = plan_row.get(key, None)
        try:
            fv = float(v)
        except Exception:
            return float(default)
        return fv if pd.notna(fv) else float(default)

    def _int(key: str, default: int = 0) -> int:
        v = plan_row.get(key, None)
        try:
            return int(float(v))
        except Exception:
            return int(default)

    forecast_demand = _num("forecast_demand", 0.0)
    beginning_on_hand = _num("beginning_on_hand", _num("starting_net_on_hand", 0.0))
    inventory_position = _num("inventory_position", beginning_on_hand)

    lead_time_days = _int("lead_time_days", int(_env_float("DECISION_DEFAULT_LEAD_TIME_DAYS", 14)))
    lead_time_months = _int("lead_time_months", max(1, int((lead_time_days + 29) / 30)))

    max_capacity_per_week = _num("max_capacity_per_week", 0.0)
    moq = _int("moq", 0)
    order_multiple = _int("order_multiple", 1) or 1

    safety_stock = _num("safety_stock", 0.0)
    target_level = _num("target_level", 0.0)
    stockout_qty = _num("stockout_qty", 0.0)
    risk_flag = str(plan_row.get("risk_flag", "OK") or "OK")

    unit_cost = _num("unit_cost", _env_float("DECISION_UNIT_COST", 1.0))
    holding_cost_per_unit = _num("holding_cost_per_unit", _env_float("DECISION_HOLDING_COST_PER_UNIT", 0.1))
    expedite_cost_per_unit = _num(
        "stockout_cost_per_unit",
        _num("expedite_cost_per_unit", _env_float("DECISION_EXPEDITE_COST_PER_UNIT", 2.0)),
    )

    sl_target = service_level_target
    if sl_target is None:
        svc_from_plan = plan_row.get("service_level", None)
        try:
            sl_target = float(svc_from_plan)
        except Exception:
            sl_target = None
    if sl_target is None:
        sl_target = _env_float("DECISION_SERVICE_LEVEL_TARGET", 0.97)
    sl_target = float(sl_target)
    if not (0.0 < sl_target <= 1.0):
        raise DecisionContextBuildError(f"service_level_target must be in (0, 1], got {sl_target}")

    return DecisionContext(
        sku=str(sku),
        location=str(location) if location is not None else (str(plan_row.get("location")) if plan_row.get("location") is not None else None),
        period=month_start.strftime("%Y-%m"),
        forecast_demand=float(forecast_demand),
        beginning_on_hand=float(beginning_on_hand),
        inventory_position=float(inventory_position),
        lead_time_days=int(lead_time_days),
        lead_time_months=int(lead_time_months),
        max_capacity_per_week=float(max_capacity_per_week),
        moq=int(moq),
        order_multiple=int(order_multiple),
        safety_stock=float(safety_stock),
        target_level=float(target_level),
        service_level_target=float(sl_target),
        unit_cost=float(unit_cost),
        holding_cost_per_unit=float(holding_cost_per_unit),
        expedite_cost_per_unit=float(expedite_cost_per_unit),
        risk_flag=risk_flag,
        stockout_qty=float(stockout_qty),
    )
