from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class DecisionContext:
    """
    Minimal, deterministic context required to generate and score decision options.

    This is intentionally separate from `model.py` (forecasting models) to avoid
    importing heavy ML dependencies in decision endpoints.
    """

    sku: str
    period: str  # "YYYY-MM"
    location: Optional[str] = None

    forecast_demand: float = 0.0
    beginning_on_hand: float = 0.0
    inventory_position: float = 0.0

    lead_time_days: int = 14
    lead_time_months: int = 1

    max_capacity_per_week: float = 0.0
    moq: int = 0
    order_multiple: int = 1

    safety_stock: float = 0.0
    target_level: float = 0.0
    service_level_target: float = 0.97

    # Simple cost knobs (can be overridden via env / policy table)
    unit_cost: float = 1.0
    holding_cost_per_unit: float = 0.1
    expedite_cost_per_unit: float = 2.0

    # Context from the plan (optional but helpful for messaging)
    risk_flag: str = "OK"
    stockout_qty: float = 0.0


@dataclass(frozen=True)
class DecisionOption:
    option_id: str
    action: str
    parameters: Dict[str, Any] = field(default_factory=dict)

    # Convenience fields (duplicated from parameters where applicable)
    order_qty: float = 0.0
    expedite_now_qty: float = 0.0

    total_cost: float = 0.0
    stockout_risk: float = 0.0
    projected_stockout_units: float = 0.0
    service_level: float = 1.0
    ending_inventory: float = 0.0
