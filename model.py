# model.py
from xgboost import XGBRegressor
from config import XGB_PARAMS
from lightgbm import LGBMRegressor

def train_model(train_df, features):
    model = XGBRegressor(**XGB_PARAMS)
    model.fit(train_df[features], train_df["sales"])
    return model

def train_quantile_model(train_df, features, quantile):
    # Use XGBoost for quantile regression
    # XGBoost supports quantile regression with objective='reg:quantile' and alpha=quantile
    from xgboost import XGBRegressor
    from config import XGB_PARAMS
    xgb_params = XGB_PARAMS.copy()
    # Ensure quantile is a float strictly between 0 and 1
    q = float(quantile)
    if not (0 < q < 1):
        raise ValueError(f"Quantile alpha must be between 0 and 1 (exclusive), got {quantile}")
    xgb_params.update({
        'objective': 'reg:quantileerror',
        'quantile_alpha': q
    })
    model = XGBRegressor(**xgb_params)
    model.fit(train_df[features], train_df["sales"])
    return model


# ================================
# Decision Intelligence Models
# ================================
from dataclasses import dataclass

@dataclass
class DecisionContext:
    item: int
    store: str
    period_start: str

    forecast_demand: float
    beginning_on_hand: float
    inventory_position: float

    lead_time_months: int
    max_capacity_per_week: int

    moq: int
    order_multiple: int

    safety_stock: float
    target_level: float
    service_level: float

    risk_flag: str
    stockout_qty: float
