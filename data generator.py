import pandas as pd
import numpy as np

np.random.seed(42)

dates = pd.date_range("2023-01-01", "2024-12-31")
stores = ["S1", "S2", "S3"]
products = ["P1", "P2", "P3"]

rows = []

for d in dates:
    for s in stores:
        for p in products:
            promo = np.random.binomial(1, 0.2)
            out = np.random.binomial(1, 0.05)
            base = np.random.randint(15, 40)

            sales = 0 if out else int(
                base * (1 + promo * 0.8) * np.random.uniform(0.7, 1.3)
            )

            rows.append({
                "date": d,
                "store_id": s,
                "product_id": p,
                "sales_qty": sales,
                "price": 10 if not promo else 8,
                "promo": promo,
                "promo_discount": 20 if promo else 0,
                "out_of_stock": out,
                "inventory_level": 0 if out else np.random.randint(50, 200),
                "holiday": np.random.binomial(1, 0.03),
                "day_of_week": d.day_name(),
                "weather_index": round(np.random.uniform(0.6, 1.4), 2),
                "competitor_price": round(np.random.uniform(9, 16), 2),
            })

df = pd.DataFrame(rows)
df.to_csv("sales_forecasting_data.csv", index=False)