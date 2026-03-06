
import pandas as pd
import numpy as np

np.random.seed(42)

# Parameters
n_rows = 2000
start_date = pd.to_datetime("2022-01-01")
end_date = pd.to_datetime("2023-12-31")

date_range = pd.date_range(start_date, end_date)

# Create dataset
sales = pd.DataFrame({
    "order_id": np.arange(1, n_rows + 1),
    "order_date": np.random.choice(date_range, n_rows),
    "customer_id": np.random.randint(1000, 1100, n_rows),
    "product": np.random.choice(
        ["Laptop", "Mobile", "Tablet", "Headphones", "Monitor"],
        n_rows
    ),
    "category": np.random.choice(
        ["Electronics", "Accessories"],
        n_rows
    ),
    "region": np.random.choice(
        ["North", "South", "East", "West"],
        n_rows
    ),
    "quantity": np.random.randint(1, 5, n_rows),
    "unit_price": np.random.choice([500, 1000, 1500, 2000, 3000], n_rows),
    "discount_pct": np.random.choice([0, 5, 10, 15], n_rows)
})

# Revenue calculation
sales["gross_amount"] = sales["quantity"] * sales["unit_price"]
sales["discount_amount"] = sales["gross_amount"] * sales["discount_pct"] / 100
sales["net_amount"] = sales["gross_amount"] - sales["discount_amount"]

# Introduce some missing values
sales.loc[np.random.choice(sales.index, 50), "region"] = np.nan

# Introduce duplicates
sales = pd.concat([sales, sales.sample(20)])

# Shuffle
sales = sales.sample(frac=1).reset_index(drop=True)

print(sales.head())

# Optional: Save to CSV
sales.to_csv("sales_dataset.csv", index=False)

