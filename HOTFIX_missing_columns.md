# Hotfix for Missing item/store Columns

## Problem
The forecast function requires `item` and `store` columns, but they're being removed from the dataframe before forecasting runs.

## Root Cause
When no grain columns are selected, somewhere in the data flow the `item` and `store` columns are being filtered out, but the forecast function still validates that they must exist.

## Quick Fix

Add this code in `app.py` in the `forecast_task` function, right after retrieving the dataframe:

### Location: app.py, line ~1315 (in forecast_task function)

**BEFORE:**
```python
df = data_store[session_id]["df"]
if df is None or df.empty:
    ...
```

**AFTER:**
```python
df = data_store[session_id]["df"]

# HOTFIX: Ensure item and store columns exist
if "item" not in df.columns:
    logging.warning("[HOTFIX] Adding missing 'item' column")
    df["item"] = "ALL"  # Default value when no item column
if "store" not in df.columns:
    logging.warning("[HOTFIX] Adding missing 'store' column")
    df["store"] = "ALL"  # Default value when no store column

if df is None or df.empty:
    ...
```

## Alternative Fix (Better)

The real issue is that `item` and `store` columns exist in the uploaded CSV but are being dropped. To fix this properly, we need to find where they're being removed.

### Debug Steps:

1. Add logging right after upload in `upload_file` endpoint (~line 1189):
   ```python
   session_id = "default"
   print(f"[UPLOAD DEBUG] Storing df with columns: {df.columns.tolist()}")
   data_store[session_id] = {"df": df}
   ```

2. Add logging right before forecast in `forecast_task` (~line 1315):
   ```python
   df = data_store[session_id]["df"]
   print(f"[FORECAST DEBUG] Retrieved df with columns: {df.columns.tolist()}")
   ```

3. Check logs to see where columns disappear

## Permanent Solution

The forecast functions should be more flexible about grain columns. Modify `run_forecast2.py` line 59 to make item/store optional:

**CHANGE FROM:**
```python
required_cols = {"date", "item", "store", "sales"}
missing_cols = required_cols - set(df.columns)
if missing_cols:
    raise ValueError(f"Missing required columns: {missing_cols}")
```

**CHANGE TO:**
```python
required_cols = {"date", "sales"}
missing_cols = required_cols - set(df.columns)
if missing_cols:
    raise ValueError(f"Missing required columns: {missing_cols}")

# Add item/store if missing (for ungrouped forecasts)
if "item" not in df.columns:
    df = df.copy()
    df["item"] = "ALL"
if "store" not in df.columns:
    df = df.copy()
    df["store"] = "ALL"
```

Do the same for the second occurrence at line 554.

## Deploy the Fix

1. Make the changes above
2. Commit to git:
   ```bash
   git add app.py run_forecast2.py
   git commit -m "Fix missing item/store columns issue"
   git push
   ```
3. Render will auto-deploy

## Why This Works

The forecast code assumes `item` and `store` columns always exist because they're defined as GROUP_COLS. When they're missing, we add them with default values ("ALL") so the code can run ungrouped forecasts correctly.
