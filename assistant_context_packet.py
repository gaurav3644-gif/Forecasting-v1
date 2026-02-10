from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import pandas as pd
import on_demand_aggregator as oda


def _parse_combo_key(combo_key: Optional[str]) -> tuple[Optional[str], Optional[str]]:
    if not isinstance(combo_key, str) or "|||" not in combo_key:
        return None, None
    parts = combo_key.split("|||", 1)
    if len(parts) != 2:
        return None, None
    sku = parts[0].strip() or None
    loc = parts[1].strip() or None
    return sku, loc


def _pick_col(df: pd.DataFrame, candidates: list[str]) -> Optional[str]:
    lower = {str(c).lower(): str(c) for c in df.columns}
    for c in candidates:
        if c.lower() in lower:
            return lower[c.lower()]
    return None


def _filter_grain(df: pd.DataFrame, *, sku: Optional[str], loc: Optional[str]) -> pd.DataFrame:
    out = df
    if sku is not None:
        if "item" in out.columns:
            out = out[out["item"].astype(str) == str(sku)]
        elif "sku_id" in out.columns:
            out = out[out["sku_id"].astype(str) == str(sku)]
    if loc is not None:
        if "store" in out.columns:
            out = out[out["store"].astype(str) == str(loc)]
        elif "location" in out.columns:
            out = out[out["location"].astype(str) == str(loc)]
    return out


def _as_iso_date(v: Any) -> str:
    try:
        ts = pd.to_datetime(v, errors="coerce")
        if pd.isna(ts):
            return ""
        return ts.date().isoformat()
    except Exception:
        return ""


def _safe_records(df: pd.DataFrame, cols: list[str], max_rows: int) -> list[dict[str, Any]]:
    view = df[cols].head(max_rows).copy()
    # Convert timestamps
    for c in cols:
        if pd.api.types.is_datetime64_any_dtype(view[c]):
            view[c] = view[c].dt.strftime("%Y-%m-%d")
    # Convert to JSON-safe
    out: list[dict[str, Any]] = []
    for r in view.to_dict(orient="records"):
        clean: dict[str, Any] = {}
        for k, v in r.items():
            if v is None:
                clean[k] = None
                continue
            if isinstance(v, float):
                clean[k] = round(v, 4)
                continue
            clean[k] = v
        out.append(clean)
    return out


def build_raw_sales_block(
    raw_df: pd.DataFrame,
    *,
    sku: Optional[str],
    loc: Optional[str],
    max_rows: int = 30,
) -> dict[str, Any]:
    date_col = _pick_col(raw_df, ["date", "ds", "timestamp"])
    sales_col = _pick_col(raw_df, ["sales", "actual", "qty", "quantity", "units", "demand"])
    inv_col = _pick_col(raw_df, ["on_hand", "onhand", "inventory_level", "inventory", "qoh", "stock"])
    price_col = _pick_col(raw_df, ["price", "unit_price"])
    promo_col = _pick_col(raw_df, ["promo", "promotion", "is_promo"])

    df = raw_df.copy()
    if date_col:
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df = df.dropna(subset=[date_col]).sort_values(date_col)

    df = _filter_grain(df, sku=sku, loc=loc)

    rows_total = int(df.shape[0])
    date_range = None
    if date_col and not df.empty:
        date_range = {"min": _as_iso_date(df[date_col].min()), "max": _as_iso_date(df[date_col].max())}

    metrics: dict[str, Any] = {"rows": rows_total}
    if sales_col and sales_col in df.columns:
        s = pd.to_numeric(df[sales_col], errors="coerce").fillna(0.0)
        metrics.update(
            {
                "sales_col": sales_col,
                "total_sales": float(s.sum()),
                "avg_sales": float(s.mean()) if rows_total else 0.0,
                "max_sales": float(s.max()) if rows_total else 0.0,
            }
        )

    # Monthly rollup for raw sales (useful for questions about specific months)
    monthly: list[dict[str, Any]] = []
    if date_col and sales_col and date_col in df.columns and sales_col in df.columns and not df.empty:
        m = df[[date_col, sales_col]].copy()
        m[date_col] = pd.to_datetime(m[date_col], errors="coerce")
        m = m.dropna(subset=[date_col])
        m["month"] = m[date_col].dt.to_period("M").astype(str)
        m[sales_col] = pd.to_numeric(m[sales_col], errors="coerce").fillna(0.0)
        agg = m.groupby("month", as_index=False)[sales_col].sum().rename(columns={sales_col: "total_sales"})
        monthly = agg.tail(24).to_dict(orient="records")

    cols: list[str] = []
    for c in [date_col, sales_col, inv_col, price_col, promo_col]:
        if c and c in df.columns and c not in cols:
            cols.append(c)
    # Always show at least some columns
    if not cols:
        cols = [str(c) for c in df.columns[:6]]

    tail = df.tail(max_rows)
    records = _safe_records(tail, cols, max_rows=max_rows) if not tail.empty else []

    return {
        "grain": {"sku": sku, "location": loc},
        "date_col": date_col,
        "date_range": date_range,
        "available_columns": [str(c) for c in raw_df.columns],
        "metrics": metrics,
        "inv_col": inv_col,
        "sample_rows": records,
        "monthly_rollup": monthly,
    }


def build_forecast_block(
    forecast_df: pd.DataFrame,
    *,
    sku: Optional[str],
    loc: Optional[str],
    max_rows: int = 60,
) -> dict[str, Any]:
    date_col = _pick_col(forecast_df, ["date", "ds"])
    fc_col = _pick_col(forecast_df, ["forecast", "yhat", "prediction", "forecast_p50", "forecast_p60"])
    act_col = _pick_col(forecast_df, ["actual", "y", "sales"])

    df = forecast_df.copy()
    if date_col:
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df = df.dropna(subset=[date_col]).sort_values(date_col)

    df = _filter_grain(df, sku=sku, loc=loc)
    rows_total = int(df.shape[0])

    date_range = None
    if date_col and not df.empty:
        date_range = {"min": _as_iso_date(df[date_col].min()), "max": _as_iso_date(df[date_col].max())}

    metrics: dict[str, Any] = {"rows": rows_total, "forecast_col": fc_col, "actual_col": act_col}
    if fc_col and fc_col in df.columns:
        f = pd.to_numeric(df[fc_col], errors="coerce").fillna(0.0)
        metrics["total_forecast"] = float(f.sum())
    if act_col and act_col in df.columns:
        a = pd.to_numeric(df[act_col], errors="coerce").fillna(0.0)
        metrics["total_actual"] = float(a.sum())

    cols: list[str] = []
    for c in [date_col, act_col, fc_col]:
        if c and c in df.columns and c not in cols:
            cols.append(c)
    if not cols:
        cols = [str(c) for c in df.columns[:8]]

    tail = df.tail(max_rows)
    records = _safe_records(tail, cols, max_rows=max_rows) if not tail.empty else []

    # Monthly rollup (useful for supply questions)
    monthly: list[dict[str, Any]] = []
    if date_col and fc_col and date_col in df.columns and fc_col in df.columns and not df.empty:
        m = df[[date_col, fc_col] + ([act_col] if act_col and act_col in df.columns else [])].copy()
        m["month"] = m[date_col].dt.to_period("M").astype(str)
        m[fc_col] = pd.to_numeric(m[fc_col], errors="coerce").fillna(0.0)
        agg = m.groupby("month", as_index=False)[fc_col].sum().rename(columns={fc_col: "forecast_units"})
        if act_col and act_col in m.columns:
            m[act_col] = pd.to_numeric(m[act_col], errors="coerce").fillna(0.0)
            agg2 = m.groupby("month", as_index=False)[act_col].sum().rename(columns={act_col: "actual_units"})
            agg = agg.merge(agg2, on="month", how="left")
        monthly = agg.tail(12).to_dict(orient="records")

    return {
        "grain": {"sku": sku, "location": loc},
        "date_col": date_col,
        "date_range": date_range,
        "available_columns": [str(c) for c in forecast_df.columns],
        "metrics": metrics,
        "sample_rows": records,
        "monthly_rollup": monthly,
    }


def build_supply_plan_block(
    supply_df: pd.DataFrame,
    *,
    sku: Optional[str],
    loc: Optional[str],
    max_rows: int = 24,
) -> dict[str, Any]:
    df = supply_df.copy()
    if "period_start" in df.columns:
        df["period_start"] = pd.to_datetime(df["period_start"], errors="coerce")
        df = df.dropna(subset=["period_start"]).sort_values("period_start")

    df = _filter_grain(df, sku=sku, loc=loc)
    rows_total = int(df.shape[0])

    cols = [
        c
        for c in [
            "period_start",
            "forecast_demand",
            "beginning_on_hand",
            "receipts",
            "order_qty",
            "ending_on_hand",
            "inventory_position",
            "reorder_point",
            "safety_stock",
            "target_level",
            "risk_flag",
            "stockout_qty",
            "lead_time_days",
            "lead_time_months",
            "moq",
            "order_multiple",
            "max_capacity_per_week",
            "service_level",
        ]
        if c in df.columns
    ]
    tail = df.tail(max_rows)
    records = _safe_records(tail, cols, max_rows=max_rows) if not tail.empty and cols else []

    metrics: dict[str, Any] = {"rows": rows_total}
    if "stockout_qty" in df.columns:
        s = pd.to_numeric(df["stockout_qty"], errors="coerce").fillna(0.0)
        metrics["total_stockout_units"] = float(s.sum())
        metrics["months_with_stockout"] = int((s > 0).sum())
    if "order_qty" in df.columns:
        o = pd.to_numeric(df["order_qty"], errors="coerce").fillna(0.0)
        metrics["total_order_qty"] = float(o.sum())

    return {
        "grain": {"sku": sku, "location": loc},
        "available_columns": [str(c) for c in supply_df.columns],
        "metrics": metrics,
        "sample_rows": records,
    }


def build_context_packet(
    session: dict,
    *,
    combo_key: Optional[str] = None,
    max_rows_raw: int = 30,
    max_rows_forecast: int = 60,
    max_rows_supply: int = 24,
) -> dict[str, Any]:
    """
    Canonical context packet for LLM calls.

    This packet is the ONLY allowed knowledge source for the assistant.
    """
    sku, loc = _parse_combo_key(combo_key)
    raw_df = session.get("df")
    forecast_df = session.get("forecast_df")
    # Avoid using a DataFrame in a boolean context (ambiguous). Prefer full then fallback.
    supply_df = session.get("supply_plan_full_df")
    if not (isinstance(supply_df, pd.DataFrame) and not supply_df.empty):
        supply_df = session.get("supply_plan_df")

    packet: dict[str, Any] = {
        "grain": {"sku": sku, "location": loc, "combo_key": combo_key},
        "raw_sales": None,
        "forecast_output": None,
        "supply_plan": None,
    }

    # If session did not provide a raw dataframe, try loading a local fallback CSV
    if not (isinstance(raw_df, pd.DataFrame) and not raw_df.empty):
        try:
            import os

            base = os.path.dirname(__file__)
            candidate = os.path.join(base, "sales_forecasting_data.csv")
            if os.path.exists(candidate):
                df_try = pd.read_csv(candidate)
                # only accept if non-empty
                if isinstance(df_try, pd.DataFrame) and not df_try.empty:
                    raw_df = df_try
        except Exception:
            # Best-effort fallback; do not raise here
            raw_df = raw_df

    if isinstance(raw_df, pd.DataFrame) and not raw_df.empty:
        rs = build_raw_sales_block(raw_df, sku=sku, loc=loc, max_rows=max_rows_raw)
        # compute a Dec-2024 aggregate if possible and attach to metrics
        try:
            date_col = rs.get("date_col")
            sales_col = rs.get("metrics", {}).get("sales_col")
            if date_col and sales_col and date_col in raw_df.columns and sales_col in raw_df.columns:
                raw_df[date_col] = pd.to_datetime(raw_df[date_col], errors="coerce")
                sel = raw_df[raw_df[date_col].dt.to_period("M") == pd.Period("2024-12")]
                if not sel.empty:
                    s = pd.to_numeric(sel[sales_col], errors="coerce").fillna(0.0)
                    rs.setdefault("metrics", {})["dec_2024_total_sales"] = float(s.sum())
                else:
                    rs.setdefault("metrics", {})["dec_2024_total_sales"] = None
        except Exception:
            rs.setdefault("metrics", {})["dec_2024_total_sales"] = None

        packet["raw_sales"] = rs

        # If the session did not include an `inventory_df`, attempt to derive
        # inventory inputs from the raw data using the detected inventory column.
        try:
            if not session.get("inventory_df"):
                inv_col = rs.get("inv_col")
                # find sku and location column names in the raw df
                sku_col = _pick_col(raw_df, ["sku_id", "item", "sku", "product"])
                loc_col = _pick_col(raw_df, ["location", "store", "site"])
                if inv_col and inv_col in raw_df.columns and sku_col and sku_col in raw_df.columns:
                    tmp = raw_df.copy()
                    # ensure datetime if available to pick latest inventory snapshot
                    date_col = rs.get("date_col")
                    if date_col and date_col in tmp.columns:
                        tmp[date_col] = pd.to_datetime(tmp[date_col], errors="coerce")
                        # pick most recent row per sku/loc
                        group_cols = [sku_col] + ([loc_col] if loc_col and loc_col in tmp.columns else [])
                        idx = tmp.groupby(group_cols)[date_col].idxmax().dropna()
                        inv_rows = tmp.loc[idx, group_cols + [inv_col]]
                    else:
                        # fallback: take last observed value per group
                        group_cols = [sku_col] + ([loc_col] if loc_col and loc_col in tmp.columns else [])
                        inv_rows = (
                            tmp.groupby(group_cols, as_index=False)[inv_col].last()
                        )

                    # Normalize to inventory fields expected by planner
                    inv_rows = inv_rows.rename(columns={sku_col: "sku_id"})
                    if loc_col and loc_col in inv_rows.columns:
                        inv_rows = inv_rows.rename(columns={loc_col: "location"})
                    else:
                        inv_rows["location"] = ""
                    inv_rows = inv_rows.rename(columns={inv_col: "on_hand"})
                    inv_rows["allocated"] = 0.0
                    inv_rows["backorders"] = 0.0
                    # Serialize to JSON-safe records
                    packet["derived_inventory"] = {
                        "columns": [c for c in ["sku_id", "location", "on_hand", "allocated", "backorders"]],
                        "rows": _safe_records(inv_rows, ["sku_id", "location", "on_hand", "allocated", "backorders"], max_rows=500),
                    }
        except Exception:
            # best-effort; do not break packet construction
            packet.setdefault("derived_inventory", {})

    if isinstance(forecast_df, pd.DataFrame) and not forecast_df.empty:
        packet["forecast_output"] = build_forecast_block(forecast_df, sku=sku, loc=loc, max_rows=max_rows_forecast)

    if isinstance(supply_df, pd.DataFrame) and not supply_df.empty:
        packet["supply_plan"] = build_supply_plan_block(supply_df, sku=sku, loc=loc, max_rows=max_rows_supply)

    # Attach authoritative numeric answers and on-demand monthly rollups
    try:
        numeric: dict[str, Any] = {}
        # Raw sales aggregates
        if isinstance(raw_df, pd.DataFrame) and not raw_df.empty and packet.get("raw_sales"):
            rc = packet["raw_sales"]
            date_col = rc.get("date_col")
            sales_col = rc.get("metrics", {}).get("sales_col")
            if date_col and sales_col and date_col in raw_df.columns and sales_col in raw_df.columns:
                # Dec-2024 total
                dec = oda.compute_period_total(raw_df, "2024-12-01", "2025-01-01", date_col=date_col, value_col=sales_col)
                numeric["dec_2024_total_sales"] = dec[0]["total"] if dec else None
                # Last 24 months monthly rollup
                numeric["raw_monthly_rollup"] = oda.compute_monthly_rollup(
                    raw_df, date_col=date_col, value_col=sales_col, months=24, end_date=raw_df[date_col].max()
                )

        # Forecast aggregates
        if isinstance(forecast_df, pd.DataFrame) and not forecast_df.empty and packet.get("forecast_output"):
            fc = packet["forecast_output"]
            fc_date = fc.get("date_col")
            fc_col = fc.get("metrics", {}).get("forecast_col")
            if fc_date and fc_col and fc_date in forecast_df.columns and fc_col in forecast_df.columns:
                numeric["forecast_monthly_rollup"] = oda.compute_monthly_rollup(
                    forecast_df, date_col=fc_date, value_col=fc_col, months=12, end_date=forecast_df[fc_date].max()
                )

        # Item and store aggregates (only when no grain filter)
        if sku is None and loc is None:
            # Raw sales aggregates by item and store
            if isinstance(raw_df, pd.DataFrame) and not raw_df.empty and packet.get("raw_sales"):
                rc = packet["raw_sales"]
                sales_col = rc.get("metrics", {}).get("sales_col")

                # Item aggregates
                item_col = _pick_col(raw_df, ["sku_id", "item", "sku", "product"])
                if item_col and sales_col and item_col in raw_df.columns and sales_col in raw_df.columns:
                    agg = raw_df[[item_col, sales_col]].copy()
                    agg[sales_col] = pd.to_numeric(agg[sales_col], errors="coerce").fillna(0.0)
                    agg = agg.dropna(subset=[item_col])
                    item_totals = agg.groupby(item_col, as_index=False)[sales_col].sum()
                    item_totals = item_totals.rename(columns={item_col: "item", sales_col: "total_sales"})
                    item_totals = item_totals.sort_values("total_sales", ascending=False)
                    numeric["raw_sales_by_item"] = item_totals.to_dict(orient="records")

                # Store aggregates
                store_col = _pick_col(raw_df, ["location", "store", "site"])
                if store_col and sales_col and store_col in raw_df.columns and sales_col in raw_df.columns:
                    agg = raw_df[[store_col, sales_col]].copy()
                    agg[sales_col] = pd.to_numeric(agg[sales_col], errors="coerce").fillna(0.0)
                    agg = agg.dropna(subset=[store_col])
                    store_totals = agg.groupby(store_col, as_index=False)[sales_col].sum()
                    store_totals = store_totals.rename(columns={store_col: "location", sales_col: "total_sales"})
                    store_totals = store_totals.sort_values("total_sales", ascending=False)
                    numeric["raw_sales_by_store"] = store_totals.to_dict(orient="records")

            # Forecast aggregates by item and store
            if isinstance(forecast_df, pd.DataFrame) and not forecast_df.empty and packet.get("forecast_output"):
                fc = packet["forecast_output"]
                fc_col = fc.get("metrics", {}).get("forecast_col")

                # Item aggregates
                item_col = _pick_col(forecast_df, ["sku_id", "item", "sku", "product"])
                if item_col and fc_col and item_col in forecast_df.columns and fc_col in forecast_df.columns:
                    agg = forecast_df[[item_col, fc_col]].copy()
                    agg[fc_col] = pd.to_numeric(agg[fc_col], errors="coerce").fillna(0.0)
                    agg = agg.dropna(subset=[item_col])
                    item_totals = agg.groupby(item_col, as_index=False)[fc_col].sum()
                    item_totals = item_totals.rename(columns={item_col: "item", fc_col: "total_forecast"})
                    item_totals = item_totals.sort_values("total_forecast", ascending=False)
                    numeric["forecast_by_item"] = item_totals.to_dict(orient="records")

                # Store aggregates
                store_col = _pick_col(forecast_df, ["location", "store", "site"])
                if store_col and fc_col and store_col in forecast_df.columns and fc_col in forecast_df.columns:
                    agg = forecast_df[[store_col, fc_col]].copy()
                    agg[fc_col] = pd.to_numeric(agg[fc_col], errors="coerce").fillna(0.0)
                    agg = agg.dropna(subset=[store_col])
                    store_totals = agg.groupby(store_col, as_index=False)[fc_col].sum()
                    store_totals = store_totals.rename(columns={store_col: "location", fc_col: "total_forecast"})
                    store_totals = store_totals.sort_values("total_forecast", ascending=False)
                    numeric["forecast_by_store"] = store_totals.to_dict(orient="records")

        packet["numeric_answers"] = numeric
    except Exception:
        packet["numeric_answers"] = {}

    return packet

