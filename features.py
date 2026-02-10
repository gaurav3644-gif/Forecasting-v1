# features.py
import numpy as np
import pandas as pd

GROUP_COLS = ["item", "store"]
TARGET_COL = "sales"
DATE_COL = "date"

def get_feature_columns(df: pd.DataFrame):
    """
    Return ONLY engineered features (lag, rolling, temporal, seasonal).
    Original data columns are NOT included here - they should be added via extra_features parameter.
    """

    # Engineered feature patterns and explicit names
    engineered_features = {
        # Temporal features
        'dow', 'month',
        # Fourier features
        'sin_y', 'cos_y'
    }

    numeric_features = []
    categorical_features = []

    for col in df.columns:
        # Include if it matches engineered feature patterns
        is_engineered = (
            col in engineered_features or
            col.startswith('lag_') or
            col.startswith('rolling_')
        )

        if is_engineered:
            if pd.api.types.is_numeric_dtype(df[col]):
                numeric_features.append(col)
            else:
                categorical_features.append(col)

    return numeric_features, categorical_features

def add_features(df: pd.DataFrame, seasonal_flags: dict | None = None) -> pd.DataFrame:
    """
    Add engineered features to `df`.

    Note: seasonal_flags parameter is kept for backwards compatibility but is no longer used.
    Seasonal patterns are now captured only through Fourier features (sin_y, cos_y).
    """
    df = df.copy()

    # Ensure date column is datetime
    if not pd.api.types.is_datetime64_any_dtype(df[DATE_COL]):
        df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    
    # --------------------
    # Lag features
    # --------------------
    valid_group_cols = [col for col in GROUP_COLS if col in df.columns]
    for lag in [1,7,14,28,56,180,365]:
        if valid_group_cols:
            df[f"lag_{lag}"] = (
                df.groupby(valid_group_cols)[TARGET_COL].shift(lag)
            )
        else:
            df[f"lag_{lag}"] = df[TARGET_COL].shift(lag)

    # --------------------
    # Rolling features
    # --------------------
    if valid_group_cols:
        df["rolling_mean_14"] = (
            df.groupby(valid_group_cols)[TARGET_COL]
              .apply(lambda x: x.shift(1).rolling(14).mean())
              .reset_index(level=valid_group_cols, drop=True)
        )
    else:
        df["rolling_mean_14"] = df[TARGET_COL].shift(1).rolling(14).mean()
    # else:
    #     df["rolling_mean_14"] = df[TARGET_COL].shift(1).rolling(14).mean()


    if valid_group_cols:
        df["rolling_std_7"] = (
            df.groupby(valid_group_cols)[TARGET_COL]
              .apply(lambda x: x.shift(1).rolling(7).std())
              .reset_index(level=valid_group_cols, drop=True)
        )
    else:
        df["rolling_std_7"] = df[TARGET_COL].shift(1).rolling(7).std()

    # --------------------
    # Calendar features
    # --------------------
    df["dow"] = df[DATE_COL].dt.dayofweek
    df["month"] = df[DATE_COL].dt.month

    # --------------------
    # Fourier yearly seasonality
    # --------------------
    doy = df[DATE_COL].dt.dayofyear
    df["sin_y"] = np.sin(2 * np.pi * doy / 365)
    df["cos_y"] = np.cos(2 * np.pi * doy / 365)

    return df
