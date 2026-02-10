from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Any, Iterable, Sequence


@dataclass(frozen=True)
class _ScoredOption:
    option: Any
    score: float
    index: int


def _get_number(option: Any, field: str) -> float:
    """
    Read a numeric field from either a dict-like option or an object with attributes.
    """
    if isinstance(option, dict):
        val = option.get(field, None)
    else:
        val = getattr(option, field, None)

    if val is None:
        raise ValueError(f"Option is missing required field {field!r}: {option!r}")
    try:
        num = float(val)
    except Exception as e:
        raise ValueError(f"Option field {field!r} must be numeric, got {val!r}: {option!r}") from e
    if not isfinite(num):
        raise ValueError(f"Option field {field!r} must be finite, got {val!r}: {option!r}")
    return num


def _minmax_norm(value: float, vmin: float, vmax: float) -> float:
    if vmax <= vmin:
        return 0.5
    return (value - vmin) / (vmax - vmin)


def compute_decision_option_scores(options: Sequence[Any], service_level_target: float) -> list[float]:
    """
    Compute a deterministic score for each option (aligned to the input order).

    See `score_decision_options()` for the scoring rules.
    """
    if not isinstance(options, Sequence):
        raise TypeError("options must be a sequence (e.g., list of dicts/objects).")
    if len(options) == 0:
        return []

    try:
        target = float(service_level_target)
    except Exception as e:
        raise ValueError("service_level_target must be numeric.") from e
    if not isfinite(target) or not (0.0 <= target <= 1.0):
        raise ValueError("service_level_target must be between 0 and 1.")

    costs: list[float] = []
    risks: list[float] = []
    service_levels: list[float] = []
    for opt in options:
        costs.append(_get_number(opt, "total_cost"))
        risks.append(_get_number(opt, "stockout_risk"))
        service_levels.append(_get_number(opt, "service_level"))

    cmin, cmax = min(costs), max(costs)
    rmin, rmax = min(risks), max(risks)

    # All terms are kept roughly in [0, 1] except the penalty (intentionally large).
    cost_weight = 1.0
    risk_weight = 1.0
    service_bonus_weight = 0.25

    # Heavy penalty scale for missing the service target.
    penalty_weight = 50.0

    scores: list[float] = []
    for i in range(len(options)):
        cost = costs[i]
        risk = risks[i]
        svc = service_levels[i]

        # Normalize cost/risk relative to the provided option set.
        cost_norm = _minmax_norm(cost, cmin, cmax)
        risk_norm = _minmax_norm(risk, rmin, rmax)

        cost_score = 1.0 - cost_norm
        risk_score = 1.0 - risk_norm

        deficit = max(0.0, target - svc)
        miss_penalty = penalty_weight * (deficit / max(target, 1e-9))

        score = (cost_weight * cost_score) + (risk_weight * risk_score) + (service_bonus_weight * svc) - miss_penalty
        scores.append(float(score))

    return scores


def score_decision_options(options: Sequence[Any], service_level_target: float) -> list[Any]:
    """
    Score and rank supply/planning decision options.

    Each option must have:
      - total_cost (lower is better)
      - stockout_risk (lower is better)
      - service_level (higher is better)

    Scoring rules:
      - Options below service_level_target receive a heavy penalty
      - Lower cost is preferred
      - Lower stockout risk is preferred

    Returns:
      A list of the original option objects sorted by descending score (best first).
    """
    scores = compute_decision_option_scores(options, service_level_target)
    scored: list[_ScoredOption] = []
    for i, opt in enumerate(options):
        scored.append(_ScoredOption(option=opt, score=float(scores[i]), index=i))

    scored.sort(key=lambda s: (-s.score, s.index))
    return [s.option for s in scored]
