# forecasting.py
import pandas as pd
from config import GROUP_COLS, FORECAST_HORIZON_DAYS
import config as _config
from features import add_features

def recursive_forecast(model, history, features, start_date=None, months=12):
    if start_date is None:
        start_date = history["date"].max() + pd.Timedelta(days=1)
    horizon = months * 30  # Approximate days per month
    future_preds = []

    current_date = start_date

    for step in range(horizon):

        next_date = current_date + pd.Timedelta(days=step)
        step_rows = []

        for (item, store), hist_g in history.groupby(GROUP_COLS):
            # Use the last available data up to current_date
            available_hist = hist_g[hist_g["date"] < next_date]
            if available_hist.empty:
                continue
            last_row = available_hist.iloc[-1:]
            X = last_row[features]
            y_pred = max(0, model.predict(X)[0])

            step_rows.append({
                "date": next_date,
                "item": item,
                "store": store,
                "sales": y_pred
            })

        if step_rows:
            step_df = pd.DataFrame(step_rows)
            history = pd.concat([history, step_df], ignore_index=True)
            history = add_features(history, seasonal_flags=_config.SEASONAL_FLAGS)
            future_preds.append(step_df)

    return pd.concat(future_preds) if future_preds else pd.DataFrame()
