# evaluation.py
import matplotlib.pyplot as plt
import pandas as pd

def plot_monthly(actual_df, forecast_df):
    monthly_actual = (
        actual_df.groupby(pd.Grouper(key="date", freq="ME"))["sales"].sum()
    )

    monthly_forecast = (
        forecast_df.groupby(pd.Grouper(key="date", freq="ME"))["sales"].sum()
    )

    plt.figure(figsize=(14,7))
    plt.plot(monthly_actual.index, monthly_actual.values, label="Actual", marker="o")
    plt.plot(monthly_forecast.index, monthly_forecast.values, label="Forecast", marker="o")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
