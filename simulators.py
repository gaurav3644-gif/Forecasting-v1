def estimate_stockout_risk(available_qty: float, demand: float) -> float:
    if available_qty >= demand:
        return 0.02
    shortfall = demand - available_qty
    return min(0.95, shortfall / demand)


def calculate_service_level(available_qty: float, demand: float) -> float:
    if demand == 0:
        return 1.0
    return min(1.0, available_qty / demand)

import math

def round_to_multiple(qty: float, multiple: int) -> int:
    return int(math.ceil(qty / multiple) * multiple)


def apply_moq_and_multiple(qty: float, moq: int, multiple: int) -> int:
    qty = max(qty, moq)
    return round_to_multiple(qty, multiple)
