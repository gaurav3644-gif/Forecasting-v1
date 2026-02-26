"""
decision_engine.py
──────────────────
Three deterministic analytical engines for the Decision Engine panel.

Architecture:
  User (NL) → Intent Router (LLM) → Engine (Python, deterministic)
                                   → Structured JSON
                                   → LLM Explanation

Each engine pulls from the in-session DataFrames, runs pure-Python /
NumPy analytics, and returns a scored + ranked JSON dict.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
import pandas as pd
from requests import session


# ── Shared helpers ─────────────────────────────────────────────────────────────

def _pick_col(df: pd.DataFrame, candidates: list[str]) -> Optional[str]:
    """Return the first column name in *candidates* that exists in *df*."""
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _score_to_risk(score: float) -> str:
    if score >= 70:
        return "High"
    if score >= 40:
        return "Medium"
    return "Low"


def _to_num(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(0)


# ── Engine 1: Volatility ───────────────────────────────────────────────────────

def volatility_engine(session: dict, filters: dict, *,
                      user_email: Optional[str] = None,
                      dataset_id: Optional[int] = None) -> dict:
    """
    Identify demand instability and its contribution per region or SKU.

    Filters accepted:
        dimension : "region" | "sku"  (default "sku")
        value     : specific entity to filter to (optional)
        start_date: ISO date string (optional)
        end_date  : ISO date string (optional)

    Returns a scored, ranked dict or {"error": str} on failure.
    """
    # raw_df = session.get("df") or session.get("original_sales_df") or session.get("raw_df")
    raw_df = session.get("df")

    if raw_df is None:
        raw_df = session.get("original_sales_df")

    if raw_df is None:
        raw_df = session.get("raw_df")

    if (raw_df is None or (hasattr(raw_df, "empty") and raw_df.empty)) and user_email and dataset_id:
        try:
            from history_store import load_dataset_raw
            raw_df = load_dataset_raw(str(user_email), int(dataset_id))
        except Exception:
            pass

    if raw_df is None or (hasattr(raw_df, "empty") and raw_df.empty):
        return {"error": "No raw sales data available in this session."}

    df = raw_df.copy()
    dimension = str(filters.get("dimension") or "sku").lower()
    value      = filters.get("value")
    start_date = filters.get("start_date")
    end_date   = filters.get("end_date")

    # Detect columns
    date_col  = _pick_col(df, ["date", "ds", "timestamp"])
    sales_col = _pick_col(df, ["sales", "actual", "qty", "quantity", "units", "demand", "revenue"])
    sku_col   = _pick_col(df, ["sku_id", "item", "sku", "product"])
    loc_col   = _pick_col(df, ["location", "store", "site", "region"])

    if not date_col or not sales_col:
        return {"error": "Required columns (date, sales) not found in raw data."}

    # Resolve dimension column
    dim_col = loc_col if dimension == "region" else sku_col
    if not dim_col:
        # Try the other dimension
        dim_col = sku_col or loc_col
    if not dim_col:
        return {"error": f"No column found for dimension '{dimension}'. Upload data with a SKU or location column."}

    # Date parsing + filtering
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col])
    if start_date:
        df = df[df[date_col] >= pd.to_datetime(start_date)]
    if end_date:
        df = df[df[date_col] <= pd.to_datetime(end_date)]

    # Optional entity filter
    if value:
        df = df[df[dim_col].astype(str).str.strip() == str(value).strip()]

    df[sales_col] = _to_num(df[sales_col])
    if df.empty:
        return {"error": "No data after applying filters."}

    df["_month"] = df[date_col].dt.to_period("M").astype(str)

    # Aggregate by month + dimension entity
    grouped = (
        df.groupby(["_month", dim_col])[sales_col]
        .sum()
        .reset_index()
    )
    grouped.columns = ["month", "entity", "revenue"]

    # Total monthly revenue + variance
    total_monthly = grouped.groupby("month")["revenue"].sum()
    total_var = total_monthly.var()

    if total_var == 0 or math.isnan(total_var):
        return {"error": "Insufficient variance in data — all months have the same revenue."}

    # Variance per entity
    entity_var = (
        grouped.groupby("entity")
               .apply(lambda x: x.set_index("month")["revenue"].reindex(total_monthly.index, fill_value=0).var())
    ).fillna(0)

    # Contribution % (how much of total variance each entity explains)
    contribution = ((entity_var / total_var) * 100).clip(lower=0).round(2)
    contribution = contribution.sort_values(ascending=False)

    top_entity = str(contribution.index[0]) if len(contribution) > 0 else "N/A"
    top_pct    = float(contribution.iloc[0]) if len(contribution) > 0 else 0.0

    # Volatility score: coefficient of variation of total monthly revenue, capped at 100
    mean_rev = total_monthly.mean()
    cv = float(total_monthly.std() / mean_rev * 100) if mean_rev > 0 else 0.0
    volatility_score = round(min(100.0, cv), 1)

    return {
        "analysis_type":           "volatility",
        "dimension":               dimension,
        "top_contributor":         top_entity,
        "top_contribution_percent": top_pct,
        "volatility_score":        volatility_score,
        "risk_level":              _score_to_risk(volatility_score),
        "ranking":                 {str(k): round(float(v), 2) for k, v in contribution.items()},
        "monthly_totals":          {str(k): round(float(v), 2) for k, v in total_monthly.items()},
        "num_entities":            int(len(contribution)),
    }


# ── Engine 2: Forecast Performance ────────────────────────────────────────────

def forecast_engine(session: dict, filters: dict, *,
                    user_email: Optional[str] = None,
                    forecast_run_id: Optional[int] = None) -> dict:
    """
    Measure forecast quality: MAPE, bias, over/under-forecast ratio.

    Filters accepted:
        dimension : "sku" | "region"  (default "sku")
        value     : specific entity (optional)
        start_date: ISO date string (optional)
        end_date  : ISO date string (optional)

    Returns scored accuracy metrics or {"error": str} on failure.
    """
    forecast_df = session.get("forecast_df")

    if (forecast_df is None or (hasattr(forecast_df, "empty") and forecast_df.empty)) and user_email and forecast_run_id:
        try:
            from history_store import load_forecast_run
            loaded = load_forecast_run(str(user_email), int(forecast_run_id))
            forecast_df = loaded.get("forecast_df")
        except Exception:
            pass

    if forecast_df is None or (hasattr(forecast_df, "empty") and forecast_df.empty):
        return {"error": "No forecast data available. Run a forecast first."}

    df = forecast_df.copy()
    dimension  = str(filters.get("dimension") or "sku").lower()
    value      = filters.get("value")
    start_date = filters.get("start_date")
    end_date   = filters.get("end_date")

    # Detect columns
    date_col = _pick_col(df, ["date", "ds"])
    fc_col   = _pick_col(df, ["forecast", "yhat", "prediction", "forecast_p50", "forecast_p60"])
    act_col  = _pick_col(df, ["actual", "y", "sales"])
    sku_col  = _pick_col(df, ["sku_id", "item", "sku", "product"])
    loc_col  = _pick_col(df, ["location", "store", "site", "region"])

    if not fc_col or not act_col:
        return {"error": "Required columns (forecast, actual) not found in forecast data."}

    dim_col = loc_col if dimension == "region" else sku_col

    # Date filtering
    if date_col:
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df = df.dropna(subset=[date_col])
        if start_date:
            df = df[df[date_col] >= pd.to_datetime(start_date)]
        if end_date:
            df = df[df[date_col] <= pd.to_datetime(end_date)]

    if value and dim_col and dim_col in df.columns:
        df = df[df[dim_col].astype(str).str.strip() == str(value).strip()]

    df[fc_col]  = _to_num(df[fc_col])
    df[act_col] = _to_num(df[act_col])

    # Keep only rows with valid historical actuals (future periods have actual=0 or NaN)
    df = df.dropna(subset=[act_col])
    df = df[df[act_col] > 0]

    if df.empty:
        return {"error": "No historical data rows with actual sales found in forecast output."}

    actual   = df[act_col].values
    forecast = df[fc_col].values

    bias                = float(np.mean(forecast - actual))
    mape                = float(np.mean(np.abs((actual - forecast) / actual)) * 100)
    over_forecast_ratio = float(np.mean(forecast > actual))

    # Accuracy rating
    if mape < 10:
        accuracy_rating = "Excellent"
    elif mape < 20:
        accuracy_rating = "Good"
    elif mape < 35:
        accuracy_rating = "Moderate"
    else:
        accuracy_rating = "Poor"

    # Risk flag
    mean_actual = float(np.mean(actual))
    relative_bias = abs(bias) / mean_actual if mean_actual > 0 else 0.0
    if relative_bias > 0.10 and over_forecast_ratio > 0.60:
        risk_flag = "Systematic Overestimation"
    elif relative_bias > 0.10 and over_forecast_ratio < 0.40:
        risk_flag = "Systematic Underestimation"
    elif mape > 35:
        risk_flag = "High Error Rate"
    else:
        risk_flag = "Acceptable"

    forecast_error_score = round(min(100.0, mape * 2), 1)

    # Per-dimension MAPE ranking
    ranking: dict = {}
    if dim_col and dim_col in df.columns:
        per_dim = (
            df.groupby(dim_col)
              .apply(lambda g: float(
                  np.mean(np.abs(
                      (_to_num(g[act_col]).values - _to_num(g[fc_col]).values)
                      / np.clip(_to_num(g[act_col]).values, 1, None)
                  )) * 100
              ))
              .round(2)
              .sort_values(ascending=False)
        )
        ranking = {str(k): float(v) for k, v in per_dim.items()}

    return {
        "analysis_type":       "forecast_performance",
        "dimension":           dimension,
        "value":               value,
        "bias":                round(bias, 2),
        "mape":                round(mape, 2),
        "over_forecast_ratio": round(over_forecast_ratio, 3),
        "accuracy_rating":     accuracy_rating,
        "risk_flag":           risk_flag,
        "forecast_error_score": forecast_error_score,
        "risk_level":          _score_to_risk(forecast_error_score),
        "ranking":             ranking,
        "rows_analyzed":       int(len(df)),
    }


# ── Engine 3: Inventory Risk ───────────────────────────────────────────────────

def risk_engine(session: dict, filters: dict, *,
                user_email: Optional[str] = None,
                forecast_run_id: Optional[int] = None,
                combo_key: Optional[str] = None) -> dict:
    """
    Identify projected stockout / overstock risk from the supply plan.

    Filters accepted:
        sku            : specific SKU (optional)
        region         : specific location (optional)
        future_periods : number of upcoming months to analyse (default 4)

    Returns {"error": str} if no supply plan is in session.
    """
    sp_df = session.get("supply_plan_full_df")
    sp_df = sp_df if sp_df is not None else session.get("supply_plan_df")

    # In-memory plan may only cover the currently selected combo.
    # Always supplement with the full DB dataset so every SKU+Store is visible.
    if user_email and forecast_run_id:
        try:
            from history_store import load_all_supply_plans_combined
            db_combined = load_all_supply_plans_combined(str(user_email), int(forecast_run_id))
            if isinstance(db_combined, pd.DataFrame) and not db_combined.empty:
                sp_df = db_combined          # use the full cross-combo dataset
        except Exception:
            pass

    if sp_df is None or (hasattr(sp_df, "empty") and sp_df.empty):
        return {"error": "No supply plan data available. Generate a supply plan first."}

    if sp_df is None or (hasattr(sp_df, "empty") and sp_df.empty):
        return {"error": "No supply plan data available. Generate a supply plan first."}

    df = sp_df.copy()
    sku_filter    = filters.get("sku")
    region_filter = filters.get("region")
    try:
        future_periods = int(filters.get("future_periods") or 4)
    except (TypeError, ValueError):
        future_periods = 4

    sku_col  = _pick_col(df, ["sku_id", "item", "sku", "product"])
    loc_col  = _pick_col(df, ["location", "store", "site", "region"])
    per_col  = _pick_col(df, ["period_start", "date", "month", "ds"])

    # Apply entity filters
    if sku_filter and sku_col:
        df = df[df[sku_col].astype(str).str.strip() == str(sku_filter).strip()]
    if region_filter and loc_col:
        df = df[df[loc_col].astype(str).str.strip() == str(region_filter).strip()]

    # Apply date range filter (honour explicit start_date/end_date from the intent router)
    if per_col:
        df[per_col] = pd.to_datetime(df[per_col], errors="coerce")
        df = df.dropna(subset=[per_col])

        start_date = filters.get("start_date")
        end_date   = filters.get("end_date")

        if start_date:
            df = df[df[per_col] >= pd.to_datetime(start_date)]
        if end_date:
            df = df[df[per_col] <= pd.to_datetime(end_date)]

        # When no explicit date range given, restrict to future periods only
        # (no row cap — always process the full forward horizon for all entities)
        if not start_date and not end_date:
            today = pd.Timestamp.today().normalize()
            future_df = df[df[per_col] >= today].copy()
            if not future_df.empty:
                df = future_df

    if df.empty:
        return {"error": "No supply plan rows found after applying filters."}

    # Key columns
    eoh_col      = _pick_col(df, ["ending_on_hand", "ending_inventory"])
    ss_col       = _pick_col(df, ["safety_stock"])
    demand_col   = _pick_col(df, ["forecast_demand", "demand", "forecast"])
    stockout_col = _pick_col(df, ["stockout_qty"])

    def _col(c):
        return _to_num(df[c]) if c else pd.Series([0] * len(df), index=df.index)

    eoh          = _col(eoh_col)
    ss           = _col(ss_col)
    demand       = _col(demand_col)
    stockout_qty = _col(stockout_col)

    # Use stockout_qty > 0 as the ground truth for actual stockouts
    # (matches the CSV output — eoh < ss alone means "below safety buffer",
    #  not an actual unmet demand event).
    stockout_mask  = stockout_qty > 0
    overstock_mask = (ss > 0) & (eoh > ss * 3)

    # Per-entity risk
    high_risk_skus: list[str] = []
    overstock_items: list[str] = []
    group_cols = [c for c in [sku_col, loc_col] if c]

    if group_cols:
        for key, grp in df.groupby(group_cols):
            label = " | ".join(str(k) for k in (key if isinstance(key, tuple) else [key]))
            g_eoh      = _to_num(grp[eoh_col])      if eoh_col      else pd.Series([0])
            g_ss       = _to_num(grp[ss_col])       if ss_col       else pd.Series([0])
            g_stockout = _to_num(grp[stockout_col]) if stockout_col else pd.Series([0])

            if g_stockout.sum() > 0:
                high_risk_skus.append(label)
            if g_ss.mean() > 0 and (g_eoh > g_ss * 3).all():
                overstock_items.append(label)

    total_rows            = max(len(df), 1)
    stockout_probability  = float(stockout_mask.sum() / total_rows)
    overstock_ratio       = float(overstock_mask.sum() / total_rows)
    total_stockout_units  = float(stockout_qty.sum())
    total_demand          = float(demand.sum())

    stockout_unit_ratio = min(1.0, total_stockout_units / max(total_demand, 1))
    risk_score = round(
        min(100.0,
            stockout_probability * 60
            + overstock_ratio * 20
            + stockout_unit_ratio * 20),
        1,
    )

    return {
        "analysis_type":       "inventory_risk",
        "high_risk_skus":      high_risk_skus[:10],
        "overstock_items":     overstock_items[:10],
        "stockout_probability": round(stockout_probability, 3),
        "total_stockout_units": round(total_stockout_units, 0),
        "overstock_ratio":     round(overstock_ratio, 3),
        "risk_score":          risk_score,
        "risk_level":          _score_to_risk(risk_score),
        "periods_analyzed":    future_periods,
        "rows_analyzed":       int(len(df)),
    }
