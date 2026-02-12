# run_forecast.py
from data_loader import load_sales_data
from features import add_features
from model import train_model, train_quantile_model
from forecasting import recursive_forecast
from evaluation import plot_monthly
from config import FEATURES, TRAIN_RATIO
import config as _config
from features import add_features, get_feature_columns
import pandas as pd
import numpy as np
import logging
import builtins

# Ensure plain print() writes to stdout and is also captured in logging at INFO level.
# This helps when the app runs under uvicorn/reloader where prints can be missed.
_original_print = builtins.print
def print(*args, **kwargs):
    _original_print(*args, **kwargs)
    try:
        logging.info(" ".join(str(a) for a in args))
    except Exception:
        pass



def forecast_all_combined_prob(df, start_date=None, months=12, grain=None, extra_features=None, progress_callback=None):
    """
    Runs forecast for ALL items and stores together using quantile regression for probabilistic forecasts.
    Returns DataFrame with quantile columns: forecast_p10, forecast_p30, forecast_p60, forecast_p90.

    Args:
        extra_features: List of additional column names from the data to use as features
    """
    from config import GROUP_COLS, TRAIN_RATIO
    from features import add_features, get_feature_columns
    from model import train_quantile_model
    import numpy as np
    import pandas as pd
    print("gg inside combined_prob: ", grain)
    # Use user-selected grain columns if provided
    if grain is not None and len(grain) > 0:
        group_cols = [col for col in grain if col in df.columns]
    else:
        group_cols = [col for col in (GROUP_COLS if 'GROUP_COLS' in locals() or 'GROUP_COLS' in globals() else ["item", "store"]) if col in df.columns]

    # Handle extra features from raw data
    extra_features = extra_features or []
    extra_features = [col for col in extra_features if col in df.columns and col not in ['date', 'sales']]

    if progress_callback:
        progress_callback(0.01, "Validating input data...")
    logging.debug(f"[PROB] Using group columns for forecasting: {group_cols}")
    logging.debug(f"[PROB] Using extra features for training: {extra_features}")

    # Validate input DataFrame
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Input must be a pandas DataFrame")
    required_cols = {"date", "sales"}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Add item/store if missing (for ungrouped forecasts)
    added_default_grain = False
    if "item" not in df.columns:
        df = df.copy()
        df["item"] = "ALL"
        added_default_grain = True
        logging.debug("[PROB] Added missing 'item' column with default value 'ALL'")
    if "store" not in df.columns:
        df = df.copy()
        df["store"] = "ALL"
        added_default_grain = True
        logging.debug("[PROB] Added missing 'store' column with default value 'ALL'")

    # Update grain to include item/store if we added them
    if added_default_grain:
        logging.debug(f"[PROB] DEBUG: grain value before update: {grain}, type: {type(grain)}")
        # If grain is empty, set it to default item/store
        is_empty_grain = (grain is None or
                         len(grain) == 0 or
                         (isinstance(grain, list) and all(not str(g).strip() for g in grain)))
        if is_empty_grain:
            grain = ["item", "store"]
            logging.debug("[PROB] Updated grain to include default item/store columns")

        # Recalculate group_cols now that item/store columns exist in df
        # This handles the case where grain was already set but columns didn't exist
        group_cols = [col for col in (grain if grain else []) if col in df.columns]
        logging.debug(f"[PROB] Recalculated group_cols after adding columns: {group_cols}")

    print("gg grain line 93", grain)

    # Ensure date column is datetime
    if not pd.api.types.is_datetime64_any_dtype(df["date"]):
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"])

    df = df.sort_values("date").reset_index(drop=True)
    if progress_callback:
        progress_callback(0.10, "Cleaning sales data...")

    # Clean sales column: remove NaN and inf values
    if "sales" in df.columns:
        df["sales"] = df["sales"].replace([np.inf, -np.inf], np.nan)
        group_cols_for_clean = grain if grain else ["item", "store"]
        def clean_sales(group):
            rolling_mean = group["sales"].shift(1).rolling(7, min_periods=1).mean()
            group["sales"] = group["sales"].fillna(rolling_mean)
            group["sales"] = group["sales"].fillna(0)
            # Outlier capping removed - preserve extreme values
            # cap = group["sales"].quantile(0.99)
            # group["sales"] = np.where(group["sales"] > cap, cap, group["sales"])
            return group
        if group_cols_for_clean:
            try:
                df = df.groupby(group_cols_for_clean, group_keys=False).apply(clean_sales)
            except KeyError as e:
                logging.debug(f"[PROB] Warning: Skipping groupby cleaning due to missing columns: {e}")
                df = clean_sales(df)
        else:
            df = clean_sales(df)

    # Add features (respect seasonal flags from config)
    df_feat = add_features(df, seasonal_flags=_config.SEASONAL_FLAGS)
    if progress_callback:
        progress_callback(0.20, "Feature engineering...")
    numeric_features, categorical_features = get_feature_columns(df_feat)

    # Add user-selected extra features from raw data
    extra_numeric = []
    extra_categorical = []
    for col in extra_features:
        if col in df_feat.columns and col not in numeric_features and col not in categorical_features:
            if pd.api.types.is_numeric_dtype(df_feat[col]):
                extra_numeric.append(col)
                numeric_features.append(col)
            else:
                extra_categorical.append(col)
                categorical_features.append(col)

    if extra_features:
        logging.debug(f"[PROB] Added {len(extra_numeric)} numeric and {len(extra_categorical)} categorical extra features")

    # Remove duplicates from feature lists
    numeric_features = list(dict.fromkeys(numeric_features))
    categorical_features = list(dict.fromkeys(categorical_features))

    df_feat = df_feat.replace([np.inf, -np.inf], np.nan).fillna(0)
    df_feat = pd.get_dummies(
        df_feat,
        columns=categorical_features,
        drop_first=True,
        dtype=int
    )
    FEATURES_COMBINED = numeric_features + [
        col for col in df_feat.columns
        if any(col.startswith(cat + "_") for cat in categorical_features)
    ]

    # Remove duplicates and ensure all features exist
    FEATURES_COMBINED = [col for col in dict.fromkeys(FEATURES_COMBINED) if col in df_feat.columns]
    logging.debug(f"[PROB] Total features for training: {len(FEATURES_COMBINED)}")

    split_idx = int(len(df_feat) * TRAIN_RATIO)
    train_df = df_feat.iloc[:split_idx]

    if progress_callback:
        progress_callback(0.30, "Training quantile models...")
    # Train quantile models for each quantile
    quantiles = [0.1, 0.3, 0.6, 0.9]
    quantile_models = {}
    for q in quantiles:
        quantile_models[q] = train_quantile_model(train_df, FEATURES_COMBINED, quantile=q)

    # Driver artifacts (for UI explanations)
    driver_artifacts = {}
    try:
        import xgboost as xgb
        driver_model = quantile_models.get(0.6)
        if driver_model is not None:
            sample_n = int(min(600, max(1, len(train_df))))
            sample_X = train_df[FEATURES_COMBINED].fillna(0).sample(n=sample_n, random_state=42) if len(train_df) > sample_n else train_df[FEATURES_COMBINED].fillna(0)
            booster = driver_model.get_booster()
            dm = xgb.DMatrix(sample_X, feature_names=FEATURES_COMBINED)
            contrib = booster.predict(dm, pred_contribs=True)  # (n, n_features+1) last col = bias
            shap = contrib[:, :-1]
            mean_abs = np.mean(np.abs(shap), axis=0)
            mean_val = np.mean(shap, axis=0)
            max_abs = float(np.max(mean_abs)) if mean_abs.size else 0.0

            rows = []
            for i, feat in enumerate(FEATURES_COMBINED):
                strength = float(mean_abs[i]) if i < len(mean_abs) else 0.0
                mean_s = float(mean_val[i]) if i < len(mean_val) else 0.0
                norm = (strength / max_abs) if max_abs > 0 else 0.0
                if norm >= 0.66:
                    eff = "High"
                elif norm >= 0.33:
                    eff = "Medium"
                else:
                    eff = "Low"
                if strength <= 1e-12:
                    direction = "Mixed"
                else:
                    direction = "Mixed" if abs(mean_s) < (0.05 * strength) else ("↑ increases" if mean_s > 0 else "↓ decreases")
                rows.append({
                    "feature": feat,
                    "strength": strength,
                    "strength_norm": norm,
                    "effect": eff,
                    "direction": direction,
                    "mean_shap": mean_s,
                })
            rows.sort(key=lambda r: r["strength"], reverse=True)
            driver_artifacts = {
                "features": FEATURES_COMBINED,
                "model": driver_model,
                "directional": rows[:40],
            }
    except Exception as e:
        logging.debug(f"[PROB] Warning: driver artifacts unavailable: {e}")
        driver_artifacts = {}

    last_sales_date = df["date"].max()
    if start_date is None:
        start_date = last_sales_date + pd.Timedelta(days=1)

    df_feat_history = df_feat[df_feat["date"] < start_date].copy()
    if df_feat_history.empty:
        logging.debug("[PROB] Warning: No historical data available for forecasting. Using all available data.")
        df_feat_history = df_feat.copy()

    history = df_feat_history.copy()
    future_preds = []
    horizon = months * 30
    valid_group_cols = [col for col in group_cols if col in history.columns]
    observed_combinations = []
    grain_cols = [col for col in (grain or []) if col in df.columns]
    if grain_cols:
        observed_combinations = df[grain_cols].drop_duplicates().dropna().to_dict(orient="records")

    print("gg grain line 242", grain)

    # Calculate historical statistics for dampening
    historical_sales = df_feat_history["sales"]
    hist_mean = historical_sales.mean()
    hist_std = historical_sales.std()
    hist_max = historical_sales.max()

    # Adaptive dampening parameters
    MAX_DAILY_GROWTH_RATE = 1.20  # Allow up to 20% growth per day
    HISTORICAL_MAX_MULTIPLIER = 1.5  # Allow up to 1.5x historical max
    EXTREME_CAP_MULTIPLIER = 3.0  # Extreme safety cap at 3x historical max

    logging.debug(f"[PROB] Dampening enabled - Hist Mean: {hist_mean:.2f}, Std: {hist_std:.2f}, Max: {hist_max:.2f}")
    logging.debug(f"[PROB] Max daily growth: {(MAX_DAILY_GROWTH_RATE-1)*100:.1f}%, Historical max cap: {hist_max * HISTORICAL_MAX_MULTIPLIER:.2f}")

    for step in range(horizon):
        if progress_callback and horizon > 0:
            progress_callback(0.35 + 0.6 * (step / horizon), f"Forecasting step {step+1}/{horizon}")
        next_date = history["date"].max() + pd.Timedelta(days=1)
        step_rows = []
        if observed_combinations:
            for combo_dict in observed_combinations:
                for col in grain_cols:
                    if col not in history.columns:
                        history[col] = np.nan
                mask = (history["date"] == history["date"].max())
                for col, val in combo_dict.items():
                    mask &= (history[col] == val)
                last_row = history[mask].iloc[-1:] if mask.any() else None
                if last_row is None or last_row.empty:
                    last_row = pd.DataFrame([{**combo_dict, "date": next_date, **{f: 0 for f in FEATURES_COMBINED}}])
                else:
                    last_row = last_row.copy()
                    last_row["date"] = next_date
                    for col in grain_cols:
                        if col not in last_row.columns:
                            last_row[col] = combo_dict[col]
                missing_features = [f for f in FEATURES_COMBINED if f not in last_row.columns]
                for f in missing_features:
                    last_row[f] = 0
                X = last_row[FEATURES_COMBINED].fillna(0)
                row_dict = {**combo_dict, "date": next_date}

                # Get historical sales for this combo for dampening
                hist_mask = (history["date"] < next_date)
                for col, val in combo_dict.items():
                    if col in history.columns:
                        hist_mask &= (history[col] == val)
                combo_historical_sales = history.loc[hist_mask, "sales"] if hist_mask.any() else pd.Series([hist_mean])
                combo_mean = combo_historical_sales.mean() if len(combo_historical_sales) > 0 else hist_mean
                combo_max = combo_historical_sales.max() if len(combo_historical_sales) > 0 else hist_max

                # Use rolling average for baseline
                if len(combo_historical_sales) >= 7:
                    recent_avg = combo_historical_sales.iloc[-7:].mean()
                elif len(combo_historical_sales) > 0:
                    recent_avg = combo_historical_sales.iloc[-min(3, len(combo_historical_sales)):].mean()
                else:
                    recent_avg = combo_mean

                for i, q in enumerate(quantiles):
                    y_pred_raw = max(0, quantile_models[q].predict(X)[0])

                    # Apply smart dampening
                    if y_pred_raw <= combo_max:
                        # Within historical range - allow model prediction
                        y_pred = y_pred_raw
                    else:
                        # Above historical max - apply progressive dampening
                        if y_pred_raw <= combo_max * HISTORICAL_MAX_MULTIPLIER:
                            y_pred = min(y_pred_raw, recent_avg * MAX_DAILY_GROWTH_RATE, combo_max * HISTORICAL_MAX_MULTIPLIER)
                        else:
                            y_pred = min(y_pred_raw, combo_max * HISTORICAL_MAX_MULTIPLIER)

                    # Extreme safety cap
                    y_pred = min(y_pred, combo_max * EXTREME_CAP_MULTIPLIER)
                    y_pred = max(0, y_pred)

                    row_dict[f"forecast_p{int(q*100)}"] = y_pred
                step_rows.append(row_dict)
            step_df = pd.DataFrame(step_rows)
        else:
            last_row = history.iloc[[-1]].copy()
            last_row["date"] = next_date
            X = last_row[FEATURES_COMBINED].fillna(0)
            row_dict = {"date": next_date}

            # Use rolling average for baseline
            recent_sales = history["sales"].iloc[-min(7, len(history)):]
            recent_avg = recent_sales.mean()

            for i, q in enumerate(quantiles):
                y_pred_raw = max(0, quantile_models[q].predict(X)[0])

                # Apply smart dampening
                if y_pred_raw <= hist_max:
                    # Within historical range - allow model prediction
                    y_pred = y_pred_raw
                else:
                    # Above historical max - apply progressive dampening
                    if y_pred_raw <= hist_max * HISTORICAL_MAX_MULTIPLIER:
                        y_pred = min(y_pred_raw, recent_avg * MAX_DAILY_GROWTH_RATE, hist_max * HISTORICAL_MAX_MULTIPLIER)
                    else:
                        y_pred = min(y_pred_raw, hist_max * HISTORICAL_MAX_MULTIPLIER)

                # Extreme safety cap
                y_pred = min(y_pred, hist_max * EXTREME_CAP_MULTIPLIER)
                y_pred = max(0, y_pred)

                row_dict[f"forecast_p{int(q*100)}"] = y_pred
            step_df = pd.DataFrame([row_dict])
        # Add missing grain columns efficiently
        missing_grain_cols = [col for col in grain_cols if col not in step_df.columns]
        if missing_grain_cols:
            step_df = step_df.assign(**{col: np.nan for col in missing_grain_cols})

        step_df = step_df.loc[:, ~step_df.columns.duplicated()]
        history = history.loc[:, ~history.columns.duplicated()]

        # Add missing columns to history efficiently
        missing_cols = [col for col in step_df.columns if col not in history.columns]
        if missing_cols:
            history = history.assign(**{col: np.nan for col in missing_cols})

        # Add missing columns to step_df efficiently
        missing_cols = [col for col in history.columns if col not in step_df.columns]
        if missing_cols:
            step_df = step_df.assign(**{col: np.nan for col in missing_cols})
        step_df = step_df[history.columns]
        history = pd.concat([history, step_df], ignore_index=True)
        history = add_features(history, seasonal_flags=_config.SEASONAL_FLAGS)
        existing_categoricals = [col for col in categorical_features if col in history.columns]
        if existing_categoricals:
            history = history.replace([np.inf, -np.inf], np.nan).fillna(0)
            history = pd.get_dummies(
                history,
                columns=existing_categoricals,
                drop_first=True,
                dtype=int
            )
        # Add missing features efficiently
        missing_features = [col for col in FEATURES_COMBINED if col not in history.columns]
        if missing_features:
            history = history.assign(**{col: 0 for col in missing_features})
        future_preds.append(step_df)
        if (step + 1) % 10 == 0 or step == 0:
            logging.debug(f"[PROB] Completed step {step + 1}/{horizon}")
    if progress_callback:
        progress_callback(0.95, "Finalizing results...")

    forecast_df = pd.concat(future_preds) if future_preds else pd.DataFrame()
    if forecast_df.empty:
        logging.debug("[PROB] Warning: No forecast data generated. This may be due to insufficient historical data.")
        actual_df = df[["date", "item", "store", "sales"]].copy()
        actual_df.rename(columns={"sales": "actual"}, inplace=True)
        for q in quantiles:
            actual_df[f"forecast_p{int(q*100)}"] = 0
        feature_importance = {}
        return actual_df, feature_importance, driver_artifacts

    print("gg grain line 401", grain)
    # --- Classic forecast for 'forecast' column ---
    from run_forecast2 import forecast_all_combined
    classic_result, _ = forecast_all_combined(df, start_date=start_date, months=months, grain=grain)
    # classic_result: columns include 'date', 'item', 'store', 'actual', 'forecast', ...

    grain_cols_in_forecast = [col for col in (grain if grain else ["item", "store"]) if col in forecast_df.columns]
    base_cols = ["date", "sales"]
    default_grain = [c for c in ["item", "store"] if c in df.columns]
    requested_grain = [c for c in (grain or []) if c and c in df.columns]
    actual_cols = []
    for col in base_cols + default_grain + requested_grain:
        if col in df.columns and col not in actual_cols:
            actual_cols.append(col)
    actual_df = df[actual_cols].copy()
    if "sales" in actual_df.columns:
        actual_df.rename(columns={"sales": "actual"}, inplace=True)

    forecast_cols = [col for col in (["date"] + (grain or [])) if col in forecast_df.columns]
    forecast_df_renamed = forecast_df[forecast_cols + [f"forecast_p{int(q*100)}" for q in quantiles]].copy()

    # Normalize item/store columns to str for both DataFrames
    def _normalize_grain_cols(df_obj):
        for col in ["item", "store"]:
            if col in df_obj.columns:
                df_obj[col] = df_obj[col].astype(str).fillna("ALL")
    _normalize_grain_cols(actual_df)
    _normalize_grain_cols(forecast_df_renamed)

    # Merge quantile and classic forecast
    merge_keys = ["date"] + [col for col in ["item", "store"] if col in actual_df.columns and col in forecast_df_renamed.columns]
    if grain_cols_in_forecast:
        for col in grain_cols_in_forecast:
            if col not in merge_keys and col in actual_df.columns and col in forecast_df_renamed.columns:
                merge_keys.append(col)
    merge_cols = merge_keys

    result = pd.merge(
        actual_df,
        forecast_df_renamed,
        on=merge_cols,
        how="left",
        suffixes=(None, "_forecast")
    )
    # Fallback: if merge didn't find per item/store forecast, map date-level forecast where available
    all_forecast_map = {}
    if "item" in forecast_df_renamed.columns and "store" in forecast_df_renamed.columns:
        special = forecast_df_renamed[
            (forecast_df_renamed["item"] == "ALL") &
            (forecast_df_renamed["store"] == "ALL")
        ]
        if not special.empty:
            for _, row in special.iterrows():
                all_forecast_map[row["date"]] = row.to_dict()
    if all_forecast_map:
        quantile_cols = [f"forecast_p{int(q*100)}" for q in quantiles]
        for col in quantile_cols:
            if col in result.columns:
                result[col] = result[col].fillna(result["date"].map(lambda d: all_forecast_map.get(d, {}).get(col)))
        if "forecast" in result.columns:
            result["forecast"] = result["forecast"].fillna(result["date"].map(lambda d: all_forecast_map.get(d, {}).get("forecast")))
    # Merge in the classic forecast column
    if "forecast" in classic_result.columns:
        # Merge on date + grain columns
        classic_merge_cols = [col for col in (["date"] + (grain or [])) if col in result.columns and col in classic_result.columns]
        result = pd.merge(
            result,
            classic_result[[*classic_merge_cols, "forecast"]],
            on=classic_merge_cols,
            how="left",
            suffixes=(None, "_classic")
        )
        # If forecast already exists, prefer the classic forecast for the 'forecast' column
        result["forecast"] = result["forecast"].combine_first(result["forecast_classic"]) if "forecast_classic" in result.columns else result["forecast"]
        if "forecast_classic" in result.columns:
            result = result.drop(columns=["forecast_classic"])

    for q in quantiles:
        col = f"forecast_p{int(q*100)}"
        if col not in result.columns:
            result[col] = 0
    if grain is not None:
        for col in grain:
            if col not in result.columns:
                if col in actual_df.columns:
                    result[col] = actual_df[col]
                elif col in forecast_df_renamed.columns:
                    result[col] = forecast_df_renamed[col]
                else:
                    result[col] = ""
    main_cols = [col for col in (grain or []) if col in result.columns] + [c for c in ["date", "actual", "forecast"] + [f"forecast_p{int(q*100)}" for q in quantiles] if c in result.columns]
    other_cols = [c for c in result.columns if c not in main_cols]
    result = result[main_cols + other_cols]
    for q in quantiles:
        col = f"forecast_p{int(q*100)}"
        result[col] = result[col].fillna(0)
    result["actual"] = result["actual"].fillna(0)

    # Feature importance: use the median model (p60)
    feature_importance = dict(zip(FEATURES_COMBINED, quantile_models[0.6].feature_importances_))

    return result, feature_importance, driver_artifacts


def forecast_all(df, horizon):
    """
    Runs forecast for ALL item-store pairs.
    Returns DataFrame:
    date | item | store | actual | forecast
    """
    results = []

    for (item, store), df_g in df.groupby(["item", "store"]):
        print(item, store)
        df_g = df_g.sort_values("date").reset_index(drop=True)


        try:
            res = main_forecast(
                df=df_g,
                item=item,
                store=store,
                horizon=horizon
            )
            res["item"] = item
            res["store"] = store
            results.append(res)

        except Exception as e:
            print(f"Skipping item={item}, store={store}: {e}")

    return pd.concat(results, ignore_index=True)


def impute_oos_sales(df,grain,sales_col="sales",oos_col="out_of_stock",window=7):
    df = df.copy()
    # print("gg 1 forecast df unique country", df['Country'].unique())
    if sales_col not in df.columns or oos_col not in df.columns:
        return df

    # Ensure time order
    df = df.sort_values(grain + ["date"])

    def impute_group(g):
        g = g.copy()

        # Use only IN-STOCK sales for rolling demand
        in_stock_sales = g[sales_col].where(g[oos_col] == 0)

        rolling_demand = (
            in_stock_sales
            .shift(1)
            .rolling(window=window, min_periods=1)
            .mean()
        )

        mask = (g[oos_col] == 1) & (g[sales_col] == 0)
        # Ensure no non-finite values before astype
        safe_rolling = rolling_demand.replace([np.inf, -np.inf], np.nan).fillna(0)
        g.loc[mask, sales_col] = safe_rolling.loc[mask].astype(g[sales_col].dtype)
        return g

    if grain:
        df = (df.groupby(grain, group_keys=False, sort=False).apply(impute_group))
    else:
        df = impute_group(df)

    # Final safety
    df[sales_col] = df[sales_col].fillna(0)
    # print("gg 2 forecast df unique country", df['Country'].unique())
    return df





def forecast_all_combined(df, start_date=None, months=12, grain=None, extra_features=None, progress_callback=None):
    """
    Runs forecast for ALL items and stores together using the main() logic.
    Handles categorical encoding properly with pd.get_dummies.

    Args:
        df: DataFrame with columns [date, item, store, sales]
        horizon: Forecast horizon in days (default: 365)
        extra_features: List of additional column names from the data to use as features

    Returns:
        DataFrame with columns: date | item | store | actual | forecast
    """
    from config import GROUP_COLS, FORECAST_HORIZON_DAYS
    print("gg grain in line 595", grain)
    grain = [col for col in (grain or []) if col and col in df.columns]
    print("gg grain in line 597", grain)
    if not grain:
        grain = [col for col in (GROUP_COLS if 'GROUP_COLS' in locals() or 'GROUP_COLS' in globals() else ["item", "store"]) if col in df.columns]
    group_cols = grain[:]
    print(f"gg Using grain received for forecasting: {grain}")
    # Handle extra features from raw data
    extra_features = extra_features or []
    extra_features = [col for col in extra_features if col in df.columns and col not in ['date', 'sales']]
    # print(f"Using extra features for training: {extra_features}")
    if progress_callback:
        progress_callback(0.01, "Validating input data...")
    # print(f"Using group columns for forecasting: {group_cols}")
    
    # print(f"forecast_all_combined called with df shape: {df.shape}")
    # print(f"df columns: {df.columns.tolist()}")
    # print(f"df dtypes: {df.dtypes.to_dict()}")
    # print(f"First 2 rows of df:")
    # print(df.head(2))
    
    # Validate input DataFrame
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Input must be a pandas DataFrame")

    required_cols = {"date", "sales"}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Add item/store if missing (for ungrouped forecasts)
    added_default_grain = False
    if "item" not in df.columns:
        df = df.copy()
        df["item"] = "ALL"
        added_default_grain = True
        # print("[LOG] Added missing 'item' column with default value 'ALL'")
    if "store" not in df.columns:
        df = df.copy()
        df["store"] = "ALL"
        added_default_grain = True
        # print("[LOG] Added missing 'store' column with default value 'ALL'")

    # Update grain to include item/store if we added them
    if added_default_grain:
        print(f"[LOG] DEBUG: grain value before update: {grain}, type: {type(grain)}")
        # If grain is empty, set it to default item/store
        is_empty_grain = (grain is None or
                         len(grain) == 0 or
                         (isinstance(grain, list) and all(not str(g).strip() for g in grain)))
        if is_empty_grain:
            grain = ["item", "store"]
            # print("[LOG] Updated grain to include default item/store columns")

        # Recalculate group_cols now that item/store columns exist in df
        # This handles the case where grain was already set but columns didn't exist
        group_cols = [col for col in (grain if grain else []) if col in df.columns]
        # print(f"[LOG] Recalculated group_cols after adding columns: {group_cols}")

    # Ensure date column is datetime
    try:
        if not pd.api.types.is_datetime64_any_dtype(df["date"]):
            df = df.copy()
            df["date"] = pd.to_datetime(df["date"])
    except Exception as e:
        raise ValueError(f"Failed to parse date column: {str(e)}")
    

    # Sort data
    df = df.sort_values("date").reset_index(drop=True)
    if progress_callback:
        progress_callback(0.10, "Cleaning sales data...")
    

    # Clean sales column: remove NaN and inf values
    if "sales" in df.columns:
        # Replace inf with NaN
        df["sales"] = df["sales"].replace([np.inf, -np.inf], np.nan)
        # Impute NaN sales with rolling mean (or zero if not enough history)
        group_cols_for_clean = grain if grain else ["item", "store"]
        def clean_sales(group):
            # Impute NaN with rolling mean, then zero if still NaN
            rolling_mean = group["sales"].shift(1).rolling(7, min_periods=1).mean()
            group["sales"] = group["sales"].fillna(rolling_mean)
            group["sales"] = group["sales"].fillna(0)
            # Outlier capping removed - preserve extreme values
            # cap = group["sales"].quantile(0.99)
            # group["sales"] = np.where(group["sales"] > cap, cap, group["sales"])
            return group
        if group_cols_for_clean:
            try:
                df = df.groupby(group_cols_for_clean, group_keys=False).apply(clean_sales)
            except KeyError as e:
                # print(f"Warning: Skipping groupby cleaning due to missing columns: {e}")
                df = clean_sales(df)
        else:
            df = clean_sales(df)

    # Add features (respect seasonal flags from config)
    df_feat = add_features(df, seasonal_flags=_config.SEASONAL_FLAGS)
    if progress_callback:
        progress_callback(0.20, "Feature engineering...")
    # Preserve all grain columns (even if not used for grouping)
    grain_cols_to_preserve = [col for col in (grain or []) if col in df.columns]
    grain_df = df[grain_cols_to_preserve + ["date"]].copy() if grain_cols_to_preserve else None
    # Don't drop NaN values - the model can handle them
    # df_feat = df_feat.dropna()

    # Get feature columns and handle categorical encoding
    numeric_features, categorical_features = get_feature_columns(df_feat)

    # Add user-selected extra features from raw data
    extra_numeric = []
    extra_categorical = []
    for col in extra_features:
        if col in df_feat.columns and col not in numeric_features and col not in categorical_features:
            if pd.api.types.is_numeric_dtype(df_feat[col]):
                extra_numeric.append(col)
                numeric_features.append(col)
            else:
                extra_categorical.append(col)
                categorical_features.append(col)

    if extra_features:
        print(f"Added {len(extra_numeric)} numeric and {len(extra_categorical)} categorical extra features")

    # Remove duplicates from feature lists
    numeric_features = list(dict.fromkeys(numeric_features))
    categorical_features = list(dict.fromkeys(categorical_features))

    logging.info(f"Categorical features: {categorical_features}")

    # One-hot encode categorical features
    # Replace non-finite values before converting to int
    df_feat = df_feat.replace([np.inf, -np.inf], np.nan).fillna(0)
    df_feat = pd.get_dummies(
        df_feat,
        columns=categorical_features,
        drop_first=True,
        dtype=int
    )

    # Build FEATURES list including dummy columns
    FEATURES_COMBINED = numeric_features + [
        col for col in df_feat.columns
        if any(col.startswith(cat + "_") for cat in categorical_features)
    ]

    # Remove duplicates and ensure all features exist
    FEATURES_COMBINED = [col for col in dict.fromkeys(FEATURES_COMBINED) if col in df_feat.columns]
    print(f"Total features for training: {len(FEATURES_COMBINED)}")
    
    logging.info(f"Features used: {FEATURES_COMBINED}")
    
    # Split data for training
    split_idx = int(len(df_feat) * TRAIN_RATIO)
    train_df = df_feat.iloc[:split_idx]
    # print("Features gg ",FEATURES_COMBINED)
    # Train model
    if progress_callback:
        progress_callback(0.30, "Training model...")
    model = train_model(train_df, FEATURES_COMBINED)
    
    # Determine the last sales date
    last_sales_date = df["date"].max()
    
    # Use provided start_date or default to last_sales_date + 1
    if start_date is None:
        start_date = last_sales_date + pd.Timedelta(days=1)
    
    print(f"Last sales date: {last_sales_date}")
    print(f"Forecast start date: {start_date}")
    
    # Use data BEFORE the forecast start date as history for predictions
    df_feat_history = df_feat[df_feat["date"] < start_date].copy()
    
    print(f"Historical data shape: {df_feat_history.shape}")
    
    if df_feat_history.empty:
        # print("Warning: No historical data available for forecasting. Using all available data.")
        # Fall back to using all data if no historical data is available
        df_feat_history = df_feat.copy()
    
    # Custom recursive forecast with categorical encoding
    history = df_feat_history.copy()
    future_preds = []
    horizon = months * 30  # Convert months to days
    valid_group_cols = [col for col in group_cols if col in history.columns]
    observed_combinations = []
    grain_cols = [col for col in (grain or []) if col in df.columns]
    if grain_cols:
        observed_combinations = df[grain_cols].drop_duplicates().dropna().to_dict(orient="records")

    # Calculate historical statistics for dampening
    historical_sales = df_feat_history["sales"]
    hist_mean = historical_sales.mean()
    hist_std = historical_sales.std()
    hist_max = historical_sales.max()
    hist_median = historical_sales.median()

    # Adaptive dampening parameters
    # Use more lenient settings to allow model predictions within historical ranges
    MAX_DAILY_GROWTH_RATE = 1.20  # Allow up to 20% growth per day (more lenient)
    HISTORICAL_MAX_MULTIPLIER = 1.5  # Allow up to 1.5x historical max
    EXTREME_CAP_MULTIPLIER = 3.0  # Extreme safety cap at 3x historical max

    print(f"Dampening enabled - Hist Mean: {hist_mean:.2f}, Std: {hist_std:.2f}, Max: {hist_max:.2f}")
    print(f"Max daily growth: {(MAX_DAILY_GROWTH_RATE-1)*100:.1f}%, Historical max cap: {hist_max * HISTORICAL_MAX_MULTIPLIER:.2f}, Extreme cap: {hist_max * EXTREME_CAP_MULTIPLIER:.2f}")

    if history.empty:
        print("Error: No data available for forecasting")
    else:
        for step in range(horizon):
            if progress_callback and horizon > 0:
                progress_callback(0.35 + 0.6 * (step / horizon), f"Forecasting step {step+1}/{horizon}")
            if step == 0:
                print(f"Forecast start date: {history['date'].max()}")
            next_date = history["date"].max() + pd.Timedelta(days=1)
            step_rows = []
            # Only forecast for observed grain combinations
            if observed_combinations:
                if step == 0:
                    print("gg observed_combinations", observed_combinations)
                for combo_dict in observed_combinations:
                    # Find the most recent row in history for this combo
                    for col in grain_cols:
                        if col not in history.columns:
                            history[col] = np.nan
                    mask = (history["date"] == history["date"].max())
                    for col, val in combo_dict.items():
                        mask &= (history[col] == val)
                    last_row = history[mask].iloc[-1:] if mask.any() else None
                    if last_row is None or last_row.empty:
                        # If no history for this combo, use a template row with zeros
                        last_row = pd.DataFrame([{**combo_dict, "date": next_date, **{f: 0 for f in FEATURES_COMBINED}}])
                    else:
                        last_row = last_row.copy()
                        last_row["date"] = next_date
                        for col in grain_cols:
                            if col not in last_row.columns:
                                last_row[col] = combo_dict[col]
                    missing_features = [f for f in FEATURES_COMBINED if f not in last_row.columns]
                    for f in missing_features:
                        last_row[f] = 0
                    X = last_row[FEATURES_COMBINED].fillna(0)
                    y_pred_raw = max(0, model.predict(X)[0])

                    # Get historical sales for this combo
                    hist_mask = (history["date"] < next_date)
                    for col, val in combo_dict.items():
                        hist_mask &= (history[col] == val)
                    combo_historical_sales = history.loc[hist_mask, "sales"]
                    combo_mean = combo_historical_sales.mean() if len(combo_historical_sales) > 0 else hist_mean
                    combo_max = combo_historical_sales.max() if len(combo_historical_sales) > 0 else hist_max

                    # Apply smart prediction dampening
                    # Use 7-day rolling average as baseline instead of just last value
                    if len(combo_historical_sales) >= 7:
                        recent_avg = combo_historical_sales.iloc[-7:].mean()
                    elif len(combo_historical_sales) > 0:
                        recent_avg = combo_historical_sales.iloc[-min(3, len(combo_historical_sales)):].mean()
                    else:
                        recent_avg = combo_mean

                    # Only apply strict dampening if prediction exceeds historical max
                    if y_pred_raw <= combo_max:
                        # Within historical range - allow model prediction with light dampening
                        y_pred = y_pred_raw
                    else:
                        # Above historical max - apply progressive dampening
                        if y_pred_raw <= combo_max * HISTORICAL_MAX_MULTIPLIER:
                            # Between 1x and 1.5x historical max - moderate dampening
                            y_pred = min(y_pred_raw, recent_avg * MAX_DAILY_GROWTH_RATE, combo_max * HISTORICAL_MAX_MULTIPLIER)
                        else:
                            # Above 1.5x historical max - strong dampening
                            y_pred = min(y_pred_raw, combo_max * HISTORICAL_MAX_MULTIPLIER)

                    # Extreme safety cap
                    y_pred = min(y_pred, combo_max * EXTREME_CAP_MULTIPLIER)
                    y_pred = max(0, y_pred)  # Ensure non-negative
                    row_dict = {**combo_dict, "date": next_date, "sales": y_pred}
                    step_rows.append(row_dict)
                step_df = pd.DataFrame(step_rows)
            else:
                # No grain columns, treat all data as one group
                last_row = history.iloc[[-1]].copy()
                last_row["date"] = next_date
                X = last_row[FEATURES_COMBINED].fillna(0)
                y_pred_raw = max(0, model.predict(X)[0])

                # Apply smart prediction dampening
                # Use 7-day rolling average as baseline
                recent_sales = history["sales"].iloc[-min(7, len(history)):]
                recent_avg = recent_sales.mean()

                # Only apply strict dampening if prediction exceeds historical max
                if y_pred_raw <= hist_max:
                    # Within historical range - allow model prediction
                    y_pred = y_pred_raw
                else:
                    # Above historical max - apply progressive dampening
                    if y_pred_raw <= hist_max * HISTORICAL_MAX_MULTIPLIER:
                        # Between 1x and 1.5x historical max - moderate dampening
                        y_pred = min(y_pred_raw, recent_avg * MAX_DAILY_GROWTH_RATE, hist_max * HISTORICAL_MAX_MULTIPLIER)
                    else:
                        # Above 1.5x historical max - strong dampening
                        y_pred = min(y_pred_raw, hist_max * HISTORICAL_MAX_MULTIPLIER)

                # Extreme safety cap
                y_pred = min(y_pred, hist_max * EXTREME_CAP_MULTIPLIER)

                y_pred = max(0, y_pred)  # Ensure non-negative

                row_dict = {"date": next_date, "sales": y_pred}
                step_df = pd.DataFrame([row_dict])
            # Before one-hot encoding, ensure all grain columns are present in history
            missing_grain_cols = [col for col in grain_cols if col not in step_df.columns]
            if missing_grain_cols:
                step_df = step_df.assign(**{col: np.nan for col in missing_grain_cols})

            # Remove duplicate columns if any
            step_df = step_df.loc[:, ~step_df.columns.duplicated()]
            history = history.loc[:, ~history.columns.duplicated()]

            # Align columns order - add missing columns to history efficiently
            missing_cols = [col for col in step_df.columns if col not in history.columns]
            if missing_cols:
                history = history.assign(**{col: np.nan for col in missing_cols})

            # Add missing columns to step_df efficiently
            missing_cols = [col for col in history.columns if col not in step_df.columns]
            if missing_cols:
                step_df = step_df.assign(**{col: np.nan for col in missing_cols})
            # Reorder columns to match
            step_df = step_df[history.columns]
            history = pd.concat([history, step_df], ignore_index=True)
            history = add_features(history, seasonal_flags=_config.SEASONAL_FLAGS)
            # Only apply get_dummies if categorical columns exist
            existing_categoricals = [col for col in categorical_features if col in history.columns]
            if existing_categoricals:
                history = history.replace([np.inf, -np.inf], np.nan).fillna(0)
                history = pd.get_dummies(
                    history,
                    columns=existing_categoricals,
                    drop_first=True,
                    dtype=int
                )
            # Ensure all required columns exist
            for col in FEATURES_COMBINED:
                if col not in history.columns:
                    history[col] = 0
            future_preds.append(step_df)
            # Debug: Print progress every 10 steps
            if (step + 1) % 10 == 0 or step == 0:
                print(f"Completed step {step + 1}/{horizon}")
        if progress_callback:
            progress_callback(0.95, "Finalizing results...")
    
    forecast_df = pd.concat(future_preds) if future_preds else pd.DataFrame()
    # print("Forecast DF head:")
    # print(forecast_df.head(3))

    if forecast_df.empty:
        print("Warning: No forecast data generated. This may be due to insufficient historical data.")
        # Return empty result with proper structure
        actual_df = df[["date", "item", "store", "sales"]].copy()
        actual_df.rename(columns={"sales": "actual"}, inplace=True)
        result = actual_df.assign(forecast=0)
        feature_importance = {}
        return result, feature_importance
    
    # print(f"Forecast date range: {forecast_df['date'].min()} to {forecast_df['date'].max()}")
    grain_cols_in_forecast = [col for col in (grain if grain else ["item", "store"]) if col in forecast_df.columns]
    if grain_cols_in_forecast:
        print(f"Unique grain pairs in forecast: {forecast_df[grain_cols_in_forecast].drop_duplicates().shape[0]}")
    else:
        print(f"Unique forecast groups: {forecast_df.shape[0]}")
    
    # Prepare actual data
    actual_cols = [col for col in (["date", "sales"] + (grain or [])) if col in df.columns]
    # print("gg df columns",df.columns)
    actual_df = df[actual_cols].copy()
    if "sales" in actual_df.columns:
        actual_df.rename(columns={"sales": "actual"}, inplace=True)

    # print(f"Actual DF shape: {actual_df.shape}")
    # print(f"Actual date range: {actual_df['date'].min()} to {actual_df['date'].max()}")

    # Prepare forecast data
    forecast_cols = [col for col in (["date", "sales"] + (grain or [])) if col in forecast_df.columns]
    # print("gg forecast df columns",forecast_df.columns)
    forecast_df_renamed = forecast_df[forecast_cols].copy()
    if "sales" in forecast_df_renamed.columns:
        forecast_df_renamed.rename(columns={"sales": "forecast"}, inplace=True)
    # Merge all grain columns back if missing and preserved
    if grain_df is not None:
        for col in grain_cols_to_preserve:
            if col not in forecast_df_renamed.columns:
                forecast_df_renamed = pd.merge(forecast_df_renamed, grain_df[["date", col]], on="date", how="left")
    
    # Check for duplicates before merge
    actual_subset = [col for col in ["date", "item", "store"] if col in actual_df.columns]
    forecast_subset = [col for col in ["date", "item", "store"] if col in forecast_df_renamed.columns]
    actual_dupes = actual_df.duplicated(subset=actual_subset).sum() if actual_subset else 0
    forecast_dupes = forecast_df_renamed.duplicated(subset=forecast_subset).sum() if forecast_subset else 0
    print(f"Duplicates in actual_df: {actual_dupes}")
    print(f"Duplicates in forecast_df: {forecast_dupes}")
    print(f"checking actual and forecast:")
    print(actual_df.head(5))
    print(forecast_df_renamed.head(5))
    # Always merge on all grain columns that exist in both DataFrames (plus date)
    grain_merge_cols = [col for col in (grain or []) if col in actual_df.columns and col in forecast_df_renamed.columns]
    # Always include 'date' if present in both
    if "date" in actual_df.columns and "date" in forecast_df_renamed.columns and "date" not in grain_merge_cols:
        grain_merge_cols = ["date"] + grain_merge_cols
    print(f"Merging on columns: {grain_merge_cols}")
    if not grain_merge_cols:
        grain_merge_cols = ["date"] if "date" in actual_df.columns and "date" in forecast_df_renamed.columns else []
    result = pd.merge(
        actual_df,
        forecast_df_renamed,
        on=grain_merge_cols,
        how="outer"
    )
    
    if grain is not None:
        for col in grain:
            if col in result.columns:
                # If the column exists but has NaNs (common in outer joins), 
                # fill it with the default "ALL" or the first valid value
                result[col] = result[col].fillna("ALL").astype(str)

    # Ensure 'actual' and 'forecast' also don't have NaNs which can break the UI
    result["actual"] = result["actual"].fillna(0)
    result["forecast"] = result["forecast"].fillna(0)




    # Ensure all grain columns are present in the result for filtering, and fill with values from source DataFrames if possible
    if grain is not None:
        for col in grain:
            if col not in result.columns:
                # Try to fill from actual_df, forecast_df_renamed, or grain_df if available
                if col in actual_df.columns:
                    result[col] = actual_df[col]
                elif col in forecast_df_renamed.columns:
                    result[col] = forecast_df_renamed[col]
                elif grain_df is not None and col in grain_df.columns:
                    result[col] = grain_df[col]
                else:
                    result[col] = ""
    # Reorder columns: grain columns, date, actual, forecast, rest
    main_cols = [col for col in (grain or []) if col in result.columns] + [c for c in ["date", "actual", "forecast"] if c in result.columns]
    other_cols = [c for c in result.columns if c not in main_cols]
    result = result[main_cols + other_cols]
    
    print(f"Result DF shape after merge: {result.shape}")
    print(f"Sample of result:")
    print(result.head(10))
    print(result.tail(10))
    
    # Fill NaN values with 0
    result["actual"] = result["actual"].fillna(0)
    result["forecast"] = result["forecast"].fillna(0)
    
    # Get feature importance from the model
    feature_importance = dict(zip(FEATURES_COMBINED, model.feature_importances_))
    
    return result, feature_importance


def main_forecast(df, item, store, horizon):
    df = df[(df["item"] == item) & (df["store"] == store)]
    df = df.sort_values("date").reset_index(drop=True)

    df_feat = add_features(df, seasonal_flags=_config.SEASONAL_FLAGS).dropna()

    # Determine the last sales date
    last_sales_date = df["date"].max()

    # Calculate the forecast start date (4 months before the last sales month)
    # Using pd.DateOffset for reliable month subtraction
    forecast_start_date = last_sales_date - pd.DateOffset(months=0)

    # Filter df_feat to start forecasting from the desired date
    # Ensure 'date' column in df_feat is in datetime format if not already
    df_feat["date"] = pd.to_datetime(df_feat["date"])
    # print(df_feat['date'].min())
    df_feat_for_forecast = df_feat[df_feat["date"] == forecast_start_date].copy()
    # print(df_feat_for_forecast['date'].min())
    split_idx = int(len(df_feat) * 0.8)
    train_df = df_feat.iloc[:split_idx]

    model = train_model(train_df, FEATURES)

    forecast_df = recursive_forecast(
        model,
        df_feat_for_forecast,
        FEATURES
    )

    # print(forecast_df['date'].min())
    # merge actual + forecast for UI
    result = pd.concat([
        df[["date","sales"]].rename(columns={"sales":"actual"}),
        forecast_df.rename(columns={"sales":"forecast"})
    ])

    return result


def main():
    df = load_sales_data()

    # Example filter (can be parameterized)
    #df = df[(df["item"] == 50) & (df["store"] == 2)]
    df = df.sort_values("date").reset_index(drop=True)

    df_feat = add_features(df, seasonal_flags=_config.SEASONAL_FLAGS).dropna()
    numeric_features, categorical_features = get_feature_columns(df_feat)
    print(categorical_features)
    df_feat = pd.get_dummies(
        df_feat,
        columns=categorical_features,
        drop_first=True,
        dtype=int
    )
    FEATURES = numeric_features + [
        col for col in df_feat.columns
        if any(col.startswith(cat + "_") for cat in categorical_features)
    ]

    print(FEATURES)
    split_idx = int(len(df_feat) * TRAIN_RATIO)
    train_df = df_feat.iloc[:split_idx]

    model = train_model(train_df, FEATURES)

    # Set start_date and months for forecasting
    start_date = df["date"].max()
    months = 12  # Default forecast horizon

    forecast_df = recursive_forecast(model, df_feat, FEATURES, start_date=start_date, months=months)
    # forecast_df = recursive_forecast(model, df_feat, FEATURES)
    # forecast_df.to_csv("forecast_df.csv", index=False)

    plot_monthly(df, forecast_df)

if __name__ == "__main__":
    main()
