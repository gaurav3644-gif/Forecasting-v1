# app/planning/lead_time_option_selector.py

from decision_models import DecisionContext, DecisionOption

from option_engine import (
    option_follow_policy,
    option_expedite_now,
    option_capacity_capped,
    option_aggressive_buffer,
)


def _classify_context(ctx: DecisionContext) -> str:
    """
    Classify the planning situation based on lead time and risk.

    Returns:
        - NO_STOCKOUT
        - TEMPORARY_UNAVOIDABLE
        - MULTI_PERIOD_UNAVOIDABLE
    """

    if ctx.risk_flag != "STOCKOUT" or (ctx.stockout_qty or 0) <= 0:
        return "NO_STOCKOUT"

    if ctx.lead_time_months <= 1:
        return "TEMPORARY_UNAVOIDABLE"

    return "MULTI_PERIOD_UNAVOIDABLE"


def generate_context_valid_options(ctx: DecisionContext) -> list[DecisionOption]:
    """
    Generate ONLY the options that make sense given lead-time physics.
    """

    situation = _classify_context(ctx)

    # --------------------------------------------------
    # Case 1: No stockout → optimization / tuning phase
    # --------------------------------------------------
    if situation == "NO_STOCKOUT":
        return [
            option_follow_policy(ctx),
            option_capacity_capped(ctx),
        ]

    # --------------------------------------------------
    # Case 2: Stockout but lead time ≤ 1 month
    # Cannot fix quantity in time → response decisions
    # --------------------------------------------------
    if situation == "TEMPORARY_UNAVOIDABLE":
        return [
            option_follow_policy(ctx),     # accept temporary stockout
            option_expedite_now(ctx),       # pay to fix now
        ]

    # --------------------------------------------------
    # Case 3: Stockout spans multiple months
    # Structural problem → stronger actions allowed
    # --------------------------------------------------
    if situation == "MULTI_PERIOD_UNAVOIDABLE":
        return [
            option_follow_policy(ctx),     # accept multi-period stockout
            option_expedite_now(ctx),       # reduce duration via expedite
            option_aggressive_buffer(ctx), # over-order to prevent recurrence
        ]

    # Safety fallback (should never happen)
    return [option_follow_policy(ctx)]


def generate_context_valid_options(ctx: DecisionContext) -> list[DecisionOption]:
    situation = _classify_context(ctx)

    print("DEBUG: lead_time_months =", ctx.lead_time_months)
    print("DEBUG: stockout_qty =", ctx.stockout_qty)
    print("DEBUG: situation =", situation)

    ...

