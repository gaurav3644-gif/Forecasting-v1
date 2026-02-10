import pandas as pd
import numpy as np

def generate_supply_plan(forecast_df, inventory_dict, lead_time_days=7, safety_stock=0):
    """
    Generate a supply plan based on forecasted demand, current inventory, lead time, and safety stock.
    Args:
        forecast_df: DataFrame with columns ['date', 'item', 'store', 'forecast']
        inventory_dict: dict {(item, store): current_inventory}
        lead_time_days: int, lead time in days
        safety_stock: int or float, safety stock units
    Returns:
        supply_plan_df: DataFrame with columns ['date', 'item', 'store', 'forecast', 'projected_inventory', 'recommended_order']
    """
    df = forecast_df.copy()
    df = df.sort_values(['item', 'store', 'date']).reset_index(drop=True)
    # Ensure columns are float dtype to avoid FutureWarning
    df['projected_inventory'] = np.full(len(df), np.nan, dtype=float)
    df['recommended_order'] = np.zeros(len(df), dtype=float)
    # Group by item/store
    for (item, store), group in df.groupby(['item', 'store']):
        inv = inventory_dict.get((item, store), 0)
        proj_inv = []
        orders = []
        for idx, row in group.iterrows():
            # Projected inventory before order
            proj_inv.append(float(inv))
            # Place order if projected inventory after lead time minus safety stock is below forecast
            if idx + lead_time_days < len(group):
                future_demand = group.iloc[idx:idx+lead_time_days]['forecast'].sum()
            else:
                future_demand = group.iloc[idx:]['forecast'].sum()
            reorder_point = float(future_demand) + float(safety_stock)
            if inv < reorder_point:
                order_qty = reorder_point - inv
            else:
                order_qty = 0.0
            orders.append(float(order_qty))
            # Update inventory for next day
            inv = float(inv) - float(row['forecast']) + float(order_qty)
        # Use .values to avoid dtype warnings
        df.loc[group.index, 'projected_inventory'] = np.array(proj_inv, dtype=float).reshape(-1)
        df.loc[group.index, 'recommended_order'] = np.array(orders, dtype=float).reshape(-1)
    return df
