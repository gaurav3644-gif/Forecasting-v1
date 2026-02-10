from __future__ import annotations

from decision_models import DecisionContext, DecisionOption
from simulators import apply_moq_and_multiple, calculate_service_level, estimate_stockout_risk


def _evaluate_option(
    ctx: DecisionContext,
    *,
    order_qty_total: float,
    expedite_now_qty: float = 0.0,
) -> tuple[float, float, float, float, float]:
    """
    Deterministic, single-period evaluation:
      available_now = beginning_on_hand + expedite_now_qty
      ending_inventory = max(0, available - forecast_demand)
      service_level = min(1, available / demand)
      stockout_risk = heuristic based on shortfall vs demand

    Note: This is intentionally simple and does not re-simulate a full multi-month plan.
    """
    demand = float(ctx.forecast_demand or 0.0)
    available_now = float(ctx.beginning_on_hand or 0.0) + float(expedite_now_qty or 0.0)

    ending_inventory = max(0.0, available_now - demand)
    service_level = calculate_service_level(available_now, demand)
    stockout_risk = estimate_stockout_risk(available_now, demand) if demand > 0 else 0.0

    # Cost proxy: purchase + holding + expected stockout penalty.
    purchase_cost = float(order_qty_total or 0.0) * float(ctx.unit_cost or 0.0)
    holding_cost = float(ending_inventory) * float(ctx.holding_cost_per_unit or 0.0)

    # Interpret expedite_cost_per_unit as a generic "expedite premium" AND "stockout penalty" when needed.
    expedite_premium = float(expedite_now_qty or 0.0) * float(ctx.expedite_cost_per_unit or 0.0)
    expected_stockout_cost = float(stockout_risk) * float(demand) * float(ctx.expedite_cost_per_unit or 0.0)
    total_cost = purchase_cost + holding_cost + expedite_premium + expected_stockout_cost

    projected_stockout_units = max(0.0, demand - available_now)

    return total_cost, stockout_risk, projected_stockout_units, service_level, ending_inventory


def option_follow_policy(ctx: DecisionContext) -> DecisionOption:
    required_qty = max(0.0, float(ctx.target_level) - float(ctx.inventory_position))

    order_qty = float(
        apply_moq_and_multiple(
            required_qty,
            int(ctx.moq),
            int(ctx.order_multiple),
        )
    )

    total_cost, stockout_risk, projected_stockout_units, service_level, ending_inventory = _evaluate_option(
        ctx,
        order_qty_total=order_qty,
        expedite_now_qty=0.0,
    )

    return DecisionOption(
        option_id="A",
        action="follow_policy",
        parameters={"order_qty": round(order_qty, 2)},
        order_qty=round(order_qty, 2),
        expedite_now_qty=0.0,
        total_cost=round(float(total_cost), 4),
        stockout_risk=round(float(stockout_risk), 4),
        projected_stockout_units=round(float(projected_stockout_units), 4),
        service_level=round(float(service_level), 4),
        ending_inventory=round(float(ending_inventory), 4),
    )


def option_expedite_now(ctx: DecisionContext) -> DecisionOption:
    """
    Expedite the "policy" order to arrive in the current bucket (monthly model approximation).

    This can reduce immediate stockout risk when lead_time_months >= 1.
    """
    required_qty = max(0.0, float(ctx.target_level) - float(ctx.inventory_position))
    order_qty = float(apply_moq_and_multiple(required_qty, int(ctx.moq), int(ctx.order_multiple)))

    total_cost, stockout_risk, projected_stockout_units, service_level, ending_inventory = _evaluate_option(
        ctx,
        order_qty_total=order_qty,
        expedite_now_qty=order_qty,
    )

    return DecisionOption(
        option_id="B",
        action="expedite_now",
        parameters={
            "order_qty": round(order_qty, 2),
            "expedite_now_qty": round(order_qty, 2),
            "effective_lead_time_months": 0,
        },
        order_qty=round(order_qty, 2),
        expedite_now_qty=round(order_qty, 2),
        total_cost=round(float(total_cost), 4),
        stockout_risk=round(float(stockout_risk), 4),
        projected_stockout_units=round(float(projected_stockout_units), 4),
        service_level=round(float(service_level), 4),
        ending_inventory=round(float(ending_inventory), 4),
    )


def option_aggressive_buffer(ctx: DecisionContext) -> DecisionOption:
    required_qty = max(0.0, float(ctx.target_level) + float(ctx.stockout_qty or 0.0) - float(ctx.inventory_position))

    order_qty = float(apply_moq_and_multiple(required_qty, int(ctx.moq), int(ctx.order_multiple)))

    total_cost, stockout_risk, projected_stockout_units, service_level, ending_inventory = _evaluate_option(
        ctx,
        order_qty_total=order_qty,
        expedite_now_qty=0.0,
    )

    return DecisionOption(
        option_id="D",
        action="aggressive_buffer",
        parameters={"order_qty": round(order_qty, 2)},
        order_qty=round(order_qty, 2),
        expedite_now_qty=0.0,
        total_cost=round(float(total_cost), 4),
        stockout_risk=round(float(stockout_risk), 4),
        projected_stockout_units=round(float(projected_stockout_units), 4),
        service_level=round(float(service_level), 4),
        ending_inventory=round(float(ending_inventory), 4),
    )


def option_capacity_capped(ctx: DecisionContext) -> DecisionOption:
    max_monthly_capacity = max(0.0, float(ctx.max_capacity_per_week or 0.0) * 4.0)

    required_qty = max(0.0, float(ctx.target_level) - float(ctx.inventory_position))
    capped_qty = min(required_qty, max_monthly_capacity) if max_monthly_capacity > 0 else required_qty

    order_qty = float(apply_moq_and_multiple(capped_qty, int(ctx.moq), int(ctx.order_multiple)))

    total_cost, stockout_risk, projected_stockout_units, service_level, ending_inventory = _evaluate_option(
        ctx,
        order_qty_total=order_qty,
        expedite_now_qty=0.0,
    )

    return DecisionOption(
        option_id="C",
        action="capacity_capped",
        parameters={"order_qty": round(order_qty, 2)},
        order_qty=round(order_qty, 2),
        expedite_now_qty=0.0,
        total_cost=round(float(total_cost), 4),
        stockout_risk=round(float(stockout_risk), 4),
        projected_stockout_units=round(float(projected_stockout_units), 4),
        service_level=round(float(service_level), 4),
        ending_inventory=round(float(ending_inventory), 4),
    )

def generate_decision_options(ctx: DecisionContext):
    return [
        option_follow_policy(ctx),
        option_expedite_now(ctx),
        option_capacity_capped(ctx),
	option_aggressive_buffer(ctx)
    ]
