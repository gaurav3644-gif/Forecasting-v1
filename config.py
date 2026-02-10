# config.py

DATA_PATH = r"C:\Forecasting\v2\streamlit v2\store_sales.csv"

GROUP_COLS = ["item", "store"]


FEATURES = [
    "lag_1","lag_7","lag_14","lag_28","lag_56","lag_180","lag_365",
    "rolling_mean_7","rolling_mean_14","rolling_std_7",
    "dow","month","is_q4","months_to_dec",
    "is_pre_peak","is_peak_build",
    "time_idx","time_x_is_peak",
    "sin_y","cos_y"
]

# Toggleable seasonal feature switches (default: enabled)
SEASONAL_FLAGS = {
    "is_q4": True,
    "months_to_dec": True,
    "is_pre_peak": True,
    "is_peak_build": True,
    "time_x_is_peak": True,
    # alias supported: 'timex_is_peak'
}

TRAIN_RATIO = 0.8
FORECAST_HORIZON_DAYS = 365 * 2

XGB_PARAMS = dict(
    n_estimators=400,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="reg:squarederror",
    random_state=42
)
