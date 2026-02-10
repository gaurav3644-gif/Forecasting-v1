# app/planning/decision_classifier.py

def classify_stockout_situation(ctx):
    """
    Classifies the decision situation based on lead time and risk.

    Returns:
        - NO_STOCKOUT
        - TEMPORARY_UNAVOIDABLE
        - MULTI_PERIOD_UNAVOIDABLE
    """

    if ctx.risk_flag != "STOCKOUT":
        return "NO_STOCKOUT"

    # Lead time <= 1 month → cannot fix current month only
    if ctx.lead_time_months <= 1:
        return "TEMPORARY_UNAVOIDABLE"

    # Lead time > 1 month → stockout spans multiple periods
    return "MULTI_PERIOD_UNAVOIDABLE"
