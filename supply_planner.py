import pandas as pd
import numpy as np
from statistics import NormalDist

def generate_supply_plan(
    forecast_df: pd.DataFrame,
    inventory_df: pd.DataFrame,
    constraints_df: pd.DataFrame,
    policy_df: pd.DataFrame,
    current_date: str,
    strict: bool = True,
) -> pd.DataFrame:

    forecast_df = forecast_df.copy()
    inventory_df = inventory_df.copy()
    constraints_df = constraints_df.copy()
    policy_df = policy_df.copy()

    _require_columns(forecast_df, ["sku_id", "location", "week_start", "forecast_demand"], "forecast_df")
    _require_columns(inventory_df, ["sku_id", "location", "on_hand", "allocated", "backorders"], "inventory_df")
    _require_columns(
        constraints_df,
        ["sku_id", "lead_time_days", "moq", "order_multiple"],
        "constraints_df",
    )
    _require_columns(policy_df, ["sku_id", "service_level"], "policy_df")

    plans = []
    forecast_df["sku_id"] = forecast_df["sku_id"].astype(str)
    forecast_df["location"] = forecast_df["location"].astype(str)
    forecast_df["week_start"] = pd.to_datetime(forecast_df["week_start"], errors="coerce")
    forecast_df["forecast_demand"] = pd.to_numeric(forecast_df["forecast_demand"], errors="coerce")
    forecast_df = forecast_df.dropna(subset=["week_start", "forecast_demand", "sku_id", "location"])

    inventory_df["sku_id"] = inventory_df["sku_id"].astype(str)
    inventory_df["location"] = inventory_df["location"].astype(str)
    for col in ["on_hand", "allocated", "backorders"]:
        inventory_df[col] = pd.to_numeric(inventory_df[col], errors="coerce").fillna(0.0)

    constraints_df["sku_id"] = constraints_df["sku_id"].astype(str)
    for col in ["lead_time_days", "moq", "order_multiple", "max_capacity_per_week"]:
        if col in constraints_df.columns:
            constraints_df[col] = pd.to_numeric(constraints_df[col], errors="coerce")

    policy_df["sku_id"] = policy_df["sku_id"].astype(str)
    policy_df["service_level"] = pd.to_numeric(policy_df["service_level"], errors="coerce")

    current_date_ts = pd.to_datetime(current_date, errors="coerce")
    if pd.isna(current_date_ts):
        raise ValueError("current_date must be a valid date string (e.g., '2026-02-03').")

    inv_lookup = (
        inventory_df.sort_values(["sku_id", "location"])
        .drop_duplicates(subset=["sku_id", "location"], keep="first")
        .set_index(["sku_id", "location"])
    )
    cons_lookup = (
        constraints_df.sort_values(["sku_id"])
        .drop_duplicates(subset=["sku_id"], keep="first")
        .set_index("sku_id")
    )
    policy_lookup = (
        policy_df.sort_values(["sku_id"])
        .drop_duplicates(subset=["sku_id"], keep="first")
        .set_index("sku_id")
    )

    for (sku, location), f_df in forecast_df.groupby(["sku_id", "location"]):
        if sku not in cons_lookup.index:
            if strict:
                raise ValueError(f"Missing constraints for sku_id={sku}.")
            plans.append({
                "sku_id": sku,
                "location": location,
                "recommended_order_qty": 0,
                "lead_time_demand": np.nan,
                "safety_stock": np.nan,
                "inventory_position": np.nan,
                "risk_flag": "MISSING_CONSTRAINTS",
                "explanation": "No constraints row found for this SKU; cannot compute order quantity.",
            })
            continue
        if sku not in policy_lookup.index:
            if strict:
                raise ValueError(f"Missing policy for sku_id={sku}.")
            plans.append({
                "sku_id": sku,
                "location": location,
                "recommended_order_qty": 0,
                "lead_time_demand": np.nan,
                "safety_stock": np.nan,
                "inventory_position": np.nan,
                "risk_flag": "MISSING_POLICY",
                "explanation": "No policy row found for this SKU; cannot compute safety stock.",
            })
            continue

        cons = cons_lookup.loc[sku]
        policy = policy_lookup.loc[sku]

        lead_time_days = float(cons["lead_time_days"])
        if not np.isfinite(lead_time_days) or lead_time_days < 0:
            if strict:
                raise ValueError(f"Invalid lead_time_days for sku_id={sku}: {cons['lead_time_days']}")
            plans.append({
                "sku_id": sku,
                "location": location,
                "recommended_order_qty": 0,
                "lead_time_demand": np.nan,
                "safety_stock": np.nan,
                "inventory_position": np.nan,
                "risk_flag": "INVALID_CONSTRAINTS",
                "explanation": "Invalid lead_time_days; cannot compute lead-time demand.",
            })
            continue
        lead_time_weeks = int(np.ceil(lead_time_days / 7.0))
        lead_time_weeks = max(1, lead_time_weeks)

        window_end = current_date_ts + pd.Timedelta(weeks=lead_time_weeks)
        f_window = f_df[(f_df["week_start"] >= current_date_ts) & (f_df["week_start"] < window_end)]
        demand_lt = float(f_window["forecast_demand"].sum())

        demand_std = float(f_df["forecast_demand"].std())
        if not np.isfinite(demand_std) or demand_std <= 0:
            demand_std = 0.1

        service_level = float(policy["service_level"])
        if not np.isfinite(service_level) or not (0 < service_level < 1):
            if strict:
                raise ValueError(f"Invalid service_level for sku_id={sku}: {policy['service_level']} (must be 0<sl<1)")
            plans.append({
                "sku_id": sku,
                "location": location,
                "recommended_order_qty": 0,
                "lead_time_demand": round(demand_lt, 2),
                "safety_stock": np.nan,
                "inventory_position": np.nan,
                "risk_flag": "INVALID_POLICY",
                "explanation": "Invalid service_level; cannot compute safety stock.",
            })
            continue
        z = NormalDist().inv_cdf(service_level)

        safety_stock = float(z * demand_std * np.sqrt(lead_time_weeks))

        if (sku, location) in inv_lookup.index:
            inv = inv_lookup.loc[(sku, location)]
            on_hand = float(inv["on_hand"])
            allocated = float(inv["allocated"])
            backorders = float(inv["backorders"])
        else:
            on_hand = 0.0
            allocated = 0.0
            backorders = 0.0

        inventory_position = float(on_hand - allocated - backorders)

        order_qty = max(
            0,
            demand_lt + safety_stock - inventory_position
        )

        if order_qty > 0:
            moq = float(cons["moq"])
            if np.isfinite(moq) and moq > 0:
                order_qty = max(order_qty, moq)

            order_multiple = float(cons["order_multiple"])
            if not np.isfinite(order_multiple) or order_multiple <= 0:
                order_multiple = 1.0
            order_qty = float(np.ceil(order_qty / order_multiple) * order_multiple)

            if "max_capacity_per_week" in cons.index and np.isfinite(cons["max_capacity_per_week"]):
                cap = float(cons["max_capacity_per_week"])
                if cap > 0:
                    order_qty = min(order_qty, cap)

        plans.append({
            "sku_id": sku,
            "location": location,
            "recommended_order_qty": int(order_qty),
            "lead_time_demand": round(demand_lt, 2),
            "safety_stock": round(safety_stock, 2),
            "inventory_position": inventory_position,
            "risk_flag": "STOCKOUT" if order_qty > 0 else "OK",
            "explanation": _explain(order_qty, demand_lt, inventory_position)
        })

    return pd.DataFrame(plans)


def _explain(order_qty, demand, inventory):
    if order_qty <= 0:
        return "Current inventory sufficient for forecasted demand."
    return (
        f"Forecasted demand during lead time is {round(demand)}, "
        f"available inventory is {inventory}. "
        f"Ordering to prevent stockout."
    )


def _require_columns(df: pd.DataFrame, required: list[str], name: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{name} missing required columns: {missing}")


def generate_time_phased_supply_plan(
    forecast_df: pd.DataFrame,
    inventory_df: pd.DataFrame,
    constraints_df: pd.DataFrame,
    policy_df: pd.DataFrame,
    start_date: str,
    months: int = 10,
    strict: bool = False,
) -> pd.DataFrame:
    """
    Time-phased monthly supply plan (sawtooth inventory projection).

    Inputs (monthly buckets recommended):
      forecast_df: sku_id, location, period_start, forecast_demand
      inventory_df: sku_id, location, on_hand, allocated, backorders
      constraints_df: sku_id, lead_time_days, moq, order_multiple (optional: max_capacity_per_week)
      policy_df: sku_id, service_level

    Returns rows per sku_id/location/month with:
      forecast_demand, beginning_on_hand, receipts, order_qty, ending_on_hand, reorder_point, safety_stock
    """
    forecast_df = forecast_df.copy()
    inventory_df = inventory_df.copy()
    constraints_df = constraints_df.copy()
    policy_df = policy_df.copy()

    _require_columns(forecast_df, ["sku_id", "location", "period_start", "forecast_demand"], "forecast_df")
    _require_columns(inventory_df, ["sku_id", "location", "on_hand", "allocated", "backorders"], "inventory_df")
    _require_columns(constraints_df, ["sku_id", "lead_time_days", "moq", "order_multiple"], "constraints_df")
    _require_columns(policy_df, ["sku_id", "service_level"], "policy_df")

    start_ts = pd.to_datetime(start_date, errors="coerce")
    if pd.isna(start_ts):
        raise ValueError("start_date must be a valid date string (e.g., '2026-02-03').")

    if months <= 0:
        raise ValueError("months must be > 0")

    forecast_df["sku_id"] = forecast_df["sku_id"].astype(str)
    forecast_df["location"] = forecast_df["location"].astype(str)
    forecast_df["period_start"] = pd.to_datetime(forecast_df["period_start"], errors="coerce")
    forecast_df["forecast_demand"] = pd.to_numeric(forecast_df["forecast_demand"], errors="coerce").fillna(0.0)
    forecast_df = forecast_df.dropna(subset=["period_start"])

    inventory_df["sku_id"] = inventory_df["sku_id"].astype(str)
    inventory_df["location"] = inventory_df["location"].astype(str)
    for col in ["on_hand", "allocated", "backorders"]:
        inventory_df[col] = pd.to_numeric(inventory_df[col], errors="coerce").fillna(0.0)

    constraints_df["sku_id"] = constraints_df["sku_id"].astype(str)
    for col in ["lead_time_days", "moq", "order_multiple", "max_capacity_per_week"]:
        if col in constraints_df.columns:
            constraints_df[col] = pd.to_numeric(constraints_df[col], errors="coerce")

    policy_df["sku_id"] = policy_df["sku_id"].astype(str)
    policy_df["service_level"] = pd.to_numeric(policy_df["service_level"], errors="coerce")

    # Normalize start to month-start
    start_month = start_ts.to_period("M").to_timestamp()
    months_index = pd.date_range(start=start_month, periods=months, freq="MS")

    inv_lookup = (
        inventory_df.sort_values(["sku_id", "location"])
        .drop_duplicates(subset=["sku_id", "location"], keep="first")
        .set_index(["sku_id", "location"])
    )
    cons_lookup = (
        constraints_df.sort_values(["sku_id"])
        .drop_duplicates(subset=["sku_id"], keep="first")
        .set_index("sku_id")
    )
    policy_lookup = (
        policy_df.sort_values(["sku_id"])
        .drop_duplicates(subset=["sku_id"], keep="first")
        .set_index("sku_id")
    )

    # Build monthly demand series for every sku/location and fill missing months with 0
    demand = (
        forecast_df[forecast_df["period_start"].isin(months_index)]
        .groupby(["sku_id", "location", "period_start"], as_index=False)["forecast_demand"]
        .sum()
    )
    combos = forecast_df[["sku_id", "location"]].drop_duplicates()
    all_grid = combos.assign(_k=1).merge(pd.DataFrame({"period_start": months_index, "_k": 1}), on="_k").drop(columns=["_k"])
    demand = all_grid.merge(demand, on=["sku_id", "location", "period_start"], how="left")
    demand["forecast_demand"] = demand["forecast_demand"].fillna(0.0)

    plans: list[dict] = []

    def _get_inventory_inputs(sku_id: str, location_id: str) -> tuple[float, float, float, float]:
        if (sku_id, location_id) in inv_lookup.index:
            inv = inv_lookup.loc[(sku_id, location_id)]
            on_hand_raw = float(inv["on_hand"])
            allocated_raw = float(inv["allocated"])
            backorders_raw = float(inv["backorders"])
        else:
            on_hand_raw = 0.0
            allocated_raw = 0.0
            backorders_raw = 0.0

        net_on_hand = float(on_hand_raw - allocated_raw - backorders_raw)
        if not np.isfinite(net_on_hand):
            net_on_hand = 0.0
        net_on_hand = max(0.0, net_on_hand)
        return on_hand_raw, allocated_raw, backorders_raw, net_on_hand

    for (sku, location), d_df in demand.groupby(["sku_id", "location"]):
        inv_on_hand_raw, inv_allocated_raw, inv_backorders_raw, inv_net_on_hand = _get_inventory_inputs(sku, location)

        if sku not in cons_lookup.index:
            if strict:
                raise ValueError(f"Missing constraints for sku_id={sku}.")
            # Still emit a time series with flags
            for m in months_index:
                plans.append({
                    "sku_id": sku,
                    "location": location,
                    "period_start": m,
                    "forecast_demand": float(d_df.loc[d_df["period_start"] == m, "forecast_demand"].sum()),
                    "lead_time_days": np.nan,
                    "moq": np.nan,
                    "order_multiple": np.nan,
                    "max_capacity_per_week": np.nan,
                    "service_level": np.nan,
                    "input_on_hand": inv_on_hand_raw,
                    "input_allocated": inv_allocated_raw,
                    "input_backorders": inv_backorders_raw,
                    "starting_net_on_hand": inv_net_on_hand,
                    "beginning_on_hand": np.nan,
                    "receipts": 0.0,
                    "order_qty": 0.0,
                    "ending_on_hand": np.nan,
                    "reorder_point": np.nan,
                    "safety_stock": np.nan,
                    "target_level": np.nan,
                    "lead_time_months": np.nan,
                    "risk_flag": "MISSING_CONSTRAINTS",
                    "explanation": "No constraints row found for this SKU; cannot compute plan.",
                })
            continue

        if sku not in policy_lookup.index:
            if strict:
                raise ValueError(f"Missing policy for sku_id={sku}.")
            cons = cons_lookup.loc[sku]
            lead_time_days = float(cons["lead_time_days"])
            if not np.isfinite(lead_time_days) or lead_time_days < 0:
                lead_time_days = 14.0
            lead_time_months = int(np.ceil(lead_time_days / 30.0))
            lead_time_months = max(1, lead_time_months)
            moq = float(cons["moq"])
            order_multiple = float(cons["order_multiple"])
            cap_week = float(cons["max_capacity_per_week"]) if "max_capacity_per_week" in cons.index else np.nan
            if not np.isfinite(order_multiple) or order_multiple <= 0:
                order_multiple = 1.0
            for m in months_index:
                plans.append({
                    "sku_id": sku,
                    "location": location,
                    "period_start": m,
                    "forecast_demand": float(d_df.loc[d_df["period_start"] == m, "forecast_demand"].sum()),
                    "lead_time_days": round(float(lead_time_days), 2),
                    "moq": moq if np.isfinite(moq) else np.nan,
                    "order_multiple": order_multiple,
                    "max_capacity_per_week": cap_week if np.isfinite(cap_week) else np.nan,
                    "service_level": np.nan,
                    "input_on_hand": inv_on_hand_raw,
                    "input_allocated": inv_allocated_raw,
                    "input_backorders": inv_backorders_raw,
                    "starting_net_on_hand": inv_net_on_hand,
                    "beginning_on_hand": np.nan,
                    "receipts": 0.0,
                    "order_qty": 0.0,
                    "ending_on_hand": np.nan,
                    "reorder_point": np.nan,
                    "safety_stock": np.nan,
                    "target_level": np.nan,
                    "lead_time_months": lead_time_months,
                    "risk_flag": "MISSING_POLICY",
                    "explanation": "No policy row found for this SKU; cannot compute safety stock.",
                })
            continue

        cons = cons_lookup.loc[sku]
        policy = policy_lookup.loc[sku]

        lead_time_days = float(cons["lead_time_days"])
        if not np.isfinite(lead_time_days) or lead_time_days < 0:
            if strict:
                raise ValueError(f"Invalid lead_time_days for sku_id={sku}: {cons['lead_time_days']}")
            lead_time_days = 14.0

        lead_time_months = int(np.ceil(lead_time_days / 30.0))
        lead_time_months = max(1, lead_time_months)

        service_level = float(policy["service_level"])
        if not np.isfinite(service_level) or not (0 < service_level < 1):
            if strict:
                raise ValueError(f"Invalid service_level for sku_id={sku}: {policy['service_level']} (must be 0<sl<1)")
            service_level = 0.95

        z = NormalDist().inv_cdf(service_level)
        monthly_std = float(d_df["forecast_demand"].std())
        if not np.isfinite(monthly_std) or monthly_std <= 0:
            monthly_std = 0.1
        safety_stock = float(z * monthly_std * np.sqrt(lead_time_months))

        moq = float(cons["moq"])
        if not np.isfinite(moq):
            moq = np.nan

        order_multiple = float(cons["order_multiple"])
        if not np.isfinite(order_multiple) or order_multiple <= 0:
            order_multiple = 1.0

        cap_week = float(cons["max_capacity_per_week"]) if "max_capacity_per_week" in cons.index else np.nan
        if not np.isfinite(cap_week):
            cap_week = np.nan

        # Starting net available inventory (usable)
        on_hand = float(inv_net_on_hand)

        pipeline = [0.0] * lead_time_months  # quantities scheduled to arrive in i months (index 0 arrives this month)

        # Helper to sum demand over a forward window
        demand_series = d_df.set_index("period_start")["forecast_demand"]

        def _sum_future(month_start: pd.Timestamp, count: int) -> float:
            if count <= 0:
                return 0.0
            idx = pd.date_range(start=month_start, periods=count, freq="MS")
            return float(demand_series.reindex(idx).fillna(0.0).sum())

        for m in months_index:
            receipts = float(pipeline.pop(0)) if pipeline else 0.0
            on_hand += receipts

            # Inventory position includes on-hand plus on-order (pipeline)
            inventory_position = float(on_hand + sum(pipeline))

            lead_time_demand = _sum_future(m, lead_time_months)
            reorder_point = float(safety_stock + lead_time_demand)

            # Order-up-to target covers lead time + one review period (this month included)
            cover_demand = _sum_future(m, lead_time_months + 1)
            target_level = float(safety_stock + cover_demand)

            raw_order = max(0.0, target_level - inventory_position)

            if raw_order > 0:
                if np.isfinite(moq) and moq > 0:
                    raw_order = max(raw_order, moq)
                raw_order = float(np.ceil(raw_order / order_multiple) * order_multiple)

                # Optional capacity (weekly -> approximate monthly)
                if np.isfinite(cap_week) and cap_week > 0:
                    cap_month = cap_week * 4.0
                    raw_order = min(raw_order, cap_month)
                    raw_order = float(np.ceil(raw_order / order_multiple) * order_multiple)

            order_qty = float(raw_order)
            if lead_time_months > 0:
                if len(pipeline) < lead_time_months:
                    pipeline = pipeline + [0.0] * (lead_time_months - len(pipeline))
                pipeline[-1] += order_qty

            demand_m = float(demand_series.get(m, 0.0))
            beginning_on_hand = float(on_hand)

            # Consume demand through the month
            if on_hand >= demand_m:
                on_hand -= demand_m
                stockout = 0.0
            else:
                stockout = demand_m - on_hand
                on_hand = 0.0

            risk_flag = "OK" if stockout <= 0 else "STOCKOUT"
            explanation = (
                f"Demand={demand_m:.0f}, BeginInv={beginning_on_hand:.0f}, "
                f"Receipts={receipts:.0f}, LT={lead_time_days:.0f}d({lead_time_months}m), "
                f"ROP={reorder_point:.0f}, SS={safety_stock:.0f}, Order={order_qty:.0f}."
            )

            plans.append({
                "sku_id": sku,
                "location": location,
                "period_start": m,
                "forecast_demand": demand_m,
                "lead_time_days": round(float(lead_time_days), 2),
                "moq": moq,
                "order_multiple": order_multiple,
                "max_capacity_per_week": cap_week,
                "service_level": service_level,
                "input_on_hand": inv_on_hand_raw,
                "input_allocated": inv_allocated_raw,
                "input_backorders": inv_backorders_raw,
                "starting_net_on_hand": inv_net_on_hand,
                "beginning_on_hand": round(beginning_on_hand, 2),
                "receipts": round(receipts, 2),
                "order_qty": round(order_qty, 2),
                "ending_on_hand": round(float(on_hand), 2),
                "inventory_position": round(inventory_position, 2),
                "reorder_point": round(reorder_point, 2),
                "safety_stock": round(safety_stock, 2),
                "target_level": round(target_level, 2),
                "lead_time_months": lead_time_months,
                "risk_flag": risk_flag,
                "stockout_qty": round(float(stockout), 2),
                "explanation": explanation,
            })

    return pd.DataFrame(plans)
