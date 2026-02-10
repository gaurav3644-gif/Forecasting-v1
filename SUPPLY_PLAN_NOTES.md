# Supply Planning Notes (Monthly Time‑Phased “Sawtooth”)

This document explains how the **monthly time‑phased supply plan** is calculated in this repo, including reorder logic, lead time usage, and key output columns.

Relevant code:
- `supply_planner.py` → `generate_time_phased_supply_plan()`
- `app.py` → `/supply_plan` endpoint (builds inputs + applies overrides)

---

## 1) What problem this planner solves

For each `(sku_id, location)` and each month in the forecast horizon, the planner simulates:
- starting inventory
- incoming receipts (orders placed earlier that arrive after lead time)
- demand consumption during the month
- whether to place an order now, and how much

The result is a time series that looks like a “sawtooth”:
inventory rises on receipts, falls as demand is consumed, and orders are placed to avoid stockouts while respecting constraints (MOQ, order multiple, capacity).

---

## 2) Inputs (what you provide / what gets auto‑filled)

The time‑phased planner expects monthly buckets:

### Forecast (`forecast_df`)
Required columns:
- `sku_id`, `location`
- `period_start` (month start timestamps)
- `forecast_demand` (units for that month)

### Inventory snapshot (`inventory_df`)
Required columns:
- `sku_id`, `location`
- `on_hand`, `allocated`, `backorders`

This is treated as a **single snapshot as of the start date**, not a month‑by‑month series.

### Constraints (`constraints_df`)
Required columns:
- `sku_id`, `lead_time_days`, `moq`, `order_multiple`
Optional:
- `max_capacity_per_week`

### Policy (`policy_df`)
Required columns:
- `sku_id`, `service_level` (0–1)

---

## 3) Horizon & monthly grid

The planner creates a monthly index:
- `months_index = [start_month, start_month+1M, ..., start_month+(months-1)M]`

It then builds a complete `(sku_id, location, period_start)` grid for the horizon and fills missing forecast months with `0`.

This ensures every SKU/location has every month represented even if the input forecast is sparse.

---

## 4) Inventory snapshot → usable starting inventory

The planner repeats these snapshot values on every month row for context:
- `input_on_hand`
- `input_allocated`
- `input_backorders`
- `starting_net_on_hand`

Meaning:
- `input_on_hand`: raw on-hand inventory from `inventory_df`
- `input_allocated`: committed/reserved units (not available for new demand)
- `input_backorders`: already-owed units (past unmet demand)
- `starting_net_on_hand = max(0, on_hand - allocated - backorders)`

Only `starting_net_on_hand` is used to initialize the month‑by‑month simulation.

---

## 5) Lead time (days) in a monthly model

Because the simulation is monthly, lead time is bucketed to months:

1. `lead_time_days` comes from `constraints_df`
2. `lead_time_months = ceil(lead_time_days / 30)`
3. `lead_time_months = max(1, lead_time_months)`

So any lead time from `1–30 days` behaves as `1 month` in this monthly model.

Where lead time affects calculations:
- **Receipt timing**: orders placed now are scheduled to arrive `lead_time_months` months later.
- **Lead‑time demand**: reorder point includes forecast demand over the next `lead_time_months` months.
- **Safety stock scaling**: safety stock increases with `sqrt(lead_time_months)`.

---

## 6) Safety stock (service level driven)

The planner uses a Normal approximation:

1. Convert `service_level` (e.g., 0.95) to a z‑value:
   - `z = NormalDist().inv_cdf(service_level)`
2. Estimate monthly demand variability:
   - `monthly_std = stddev(forecast_demand across months in horizon for that sku/location)`
3. Safety stock:
   - `safety_stock = z * monthly_std * sqrt(lead_time_months)`

Notes:
- If `monthly_std` is missing or ~0, the code uses a small fallback to avoid zero/NaN.
- This is a simplified approach; real implementations often separate forecast error from demand variability and use historical error distributions.

---

## 7) Pipeline & receipts (how orders “arrive”)

The planner tracks on-order inventory with a `pipeline` list:
- length = `lead_time_months`
- `pipeline[i]` represents quantity scheduled to arrive in `i` months

Each month:
1. `receipts = pipeline.pop(0)` (arrivals this month)
2. `on_hand += receipts`

When you place an order this month:
- it is added to the far end: `pipeline[-1] += order_qty`

---

## 8) Inventory position

The reorder decision uses **inventory position** (not just on-hand):

`inventory_position = on_hand + sum(pipeline)`

This means “what you have + what’s already on the way”.

---

## 9) Reorder point (ROP)

For each month `m`:

1. Lead‑time demand (sum of future forecast):
   - `lead_time_demand = sum_{t=m..m+lead_time_months-1} forecast_demand[t]`
2. Reorder point:
   - `reorder_point = safety_stock + lead_time_demand`

If inventory position drops below what you want to cover (ROP / target), the planner orders.

---

## 10) Order‑up‑to (“target level”) policy

This planner uses an order‑up‑to target that covers:
- lead time demand **plus one review period** (this month)

1. `cover_demand = sum_{t=m..m+lead_time_months} forecast_demand[t]`  (note the `+1` month)
2. `target_level = safety_stock + cover_demand`
3. Raw order:
   - `raw_order = max(0, target_level - inventory_position)`

Interpretation:
- If you already have enough inventory position to cover that window + safety stock, order 0.
- Otherwise, order the shortfall.

---

## 11) Constraints applied to the order quantity

Once `raw_order > 0`, these constraints apply:

### MOQ
If `moq > 0`:
- `raw_order = max(raw_order, moq)`

### Order multiple
Round up to a multiple:
- `raw_order = ceil(raw_order / order_multiple) * order_multiple`

### Capacity (optional)
If `max_capacity_per_week` exists:
- approximate monthly cap: `cap_month = max_capacity_per_week * 4`
- limit: `raw_order = min(raw_order, cap_month)`
- then re-round to `order_multiple`

Final:
- `order_qty = raw_order`

---

## 12) Demand consumption & ending inventory

For each month:
1. `beginning_on_hand` is recorded after receipts are added.
2. Demand is consumed:
   - if `on_hand >= demand_m`: `on_hand -= demand_m`
   - else: stockout occurs, `stockout_qty = demand_m - on_hand`, and `on_hand = 0`
3. `ending_on_hand` is recorded.

The plan labels the month:
- `risk_flag = "OK"` if no stockout
- `risk_flag = "STOCKOUT"` if stockout occurred

---

## 13) Key output columns (quick reference)

Inputs (constant across months for a sku/location):
- `lead_time_days`: from `constraints_df` (your override)
- `lead_time_months`: `ceil(lead_time_days/30)` (used by the monthly simulation)
- `moq`, `order_multiple`, `max_capacity_per_week`: from `constraints_df`
- `service_level`: from `policy_df`
- `input_on_hand`, `input_allocated`, `input_backorders`: raw inventory snapshot inputs
- `starting_net_on_hand`: usable starting inventory after adjustments

Simulation (varies by month):
- `forecast_demand`: demand for the month (monthly bucket)
- `beginning_on_hand`: on-hand at month start after receipts
- `receipts`: quantity arriving this month from prior orders
- `order_qty`: new order placed this month (arrives later)
- `ending_on_hand`: on-hand after consuming demand
- `inventory_position`: `on_hand + pipeline`
- `reorder_point`: `safety_stock + lead_time_demand`
- `safety_stock`: buffer level derived from service level
- `stockout_qty`: how much demand couldn’t be satisfied
- `explanation`: readable summary of that month’s computation

---

## 14) Important limitations (so you interpret results correctly)

1. **Monthly granularity**: a 21‑day lead time cannot be represented precisely; it becomes 1 month.
2. **Single inventory snapshot**: inventory isn’t time‑phased (no planned receipts other than those generated by the model).
3. **Simplified safety stock**: uses monthly std dev over the horizon as a proxy.
4. **Capacity approximation**: weekly capacity × 4 is a rough monthly conversion.

---

## 15) How to make lead times like 21 days “matter” more

If you want 21 vs 7 vs 28 days to behave differently, you need a finer time step:
- **Weekly planning**: convert monthly forecast to weeks (or forecast weekly directly), and schedule receipts in weeks.
- **Daily planning**: most accurate but more complex (calendar, business days, etc.).

Alternative (still monthly): support fractional months (e.g., `lead_time_days/30`) and adjust pipeline/demand sums, but then you must decide how to allocate “partial-month” receipts and demand.

