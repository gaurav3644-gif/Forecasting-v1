import pandas as pd
from typing import Optional, List, Dict, Any


def _ensure_datetime(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    return df


def compute_monthly_rollup(
    df: pd.DataFrame,
    date_col: str = "date",
    value_col: str = "sales",
    group_cols: Optional[List[str]] = None,
    months: int = 24,
    end_date: Optional[pd.Timestamp] = None,
) -> List[Dict[str, Any]]:
    """Return last `months` monthly totals as a JSON-serializable list.

    If `group_cols` is provided, returns a list of groups each with a
    `monthly` array of {month: "YYYY-MM", total: float}.
    If no `group_cols`, returns a flat list of {month: "YYYY-MM", total_sales: float}.
    """
    if df is None or df.shape[0] == 0:
        return []

    df = _ensure_datetime(df, date_col)
    if end_date is None:
        max_dt = df[date_col].max()
    else:
        max_dt = pd.to_datetime(end_date)
    if pd.isna(max_dt):
        return []

    df[value_col] = pd.to_numeric(df.get(value_col, 0), errors="coerce").fillna(0.0)

    last_month = max_dt.to_period("M").to_timestamp()
    start_month = (last_month.to_period("M") - (months - 1)).to_timestamp()
    months_idx = pd.date_range(start=start_month, end=last_month, freq="MS")

    if group_cols:
        grouped = (
            df.groupby(group_cols + [pd.Grouper(key=date_col, freq="MS")])[value_col]
            .sum()
            .reset_index()
        )
        out: List[Dict[str, Any]] = []
        for _, g in grouped.groupby(group_cols):
            grp_keys = {c: g.iloc[0][c] for c in group_cols}
            series = g.set_index(date_col)[value_col].reindex(months_idx, fill_value=0.0)
            monthly = [{"month": d.strftime("%Y-%m"), "total": float(v)} for d, v in series.items()]
            out.append({"group": grp_keys, "monthly": monthly})
        return out

    series = df.groupby(pd.Grouper(key=date_col, freq="MS"))[value_col].sum().reindex(months_idx, fill_value=0.0)
    return [{"month": d.strftime("%Y-%m"), "total_sales": float(v)} for d, v in series.items()]


def compute_period_total(
    df: pd.DataFrame,
    start: str,
    end: str,
    date_col: str = "date",
    value_col: str = "sales",
    group_cols: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """Compute total `value_col` between start and end (inclusive start, exclusive end).

    Returns list of dicts per-group (or single dict) with `total` float.
    """
    if df is None or df.shape[0] == 0:
        return []

    df = _ensure_datetime(df, date_col)
    start_ts = pd.to_datetime(start, errors="coerce")
    end_ts = pd.to_datetime(end, errors="coerce")
    if pd.isna(start_ts) or pd.isna(end_ts):
        return []

    df[value_col] = pd.to_numeric(df.get(value_col, 0), errors="coerce").fillna(0.0)
    mask = (df[date_col] >= start_ts) & (df[date_col] < end_ts)
    sliced = df.loc[mask]
    if group_cols:
        grouped = sliced.groupby(group_cols)[value_col].sum().reset_index()
        return [{**{c: row[c] for c in group_cols}, "total": float(row[value_col])} for _, row in grouped.iterrows()]
    total = float(sliced[value_col].sum())
    return [{"total": total}]


def compute_custom_aggregate(
    df: pd.DataFrame,
    date_col: str,
    value_col: str,
    freq: str = "MS",
    agg: str = "sum",
    group_cols: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """Generic aggregator over `freq` periods using `agg` (sum/mean/min/max).

    Returns JSON-serializable list similar to `compute_monthly_rollup`.
    """
    if df is None or df.shape[0] == 0:
        return []
    df = _ensure_datetime(df, date_col)
    df[value_col] = pd.to_numeric(df.get(value_col, 0), errors="coerce").fillna(0.0)

    if group_cols:
        grouped = df.groupby(group_cols + [pd.Grouper(key=date_col, freq=freq)])[value_col]
        if agg == "sum":
            res = grouped.sum()
        elif agg == "mean":
            res = grouped.mean()
        elif agg == "min":
            res = grouped.min()
        elif agg == "max":
            res = grouped.max()
        else:
            raise ValueError("unsupported agg")
        res = res.reset_index()
        out: List[Dict[str, Any]] = []
        for _, g in res.groupby(group_cols):
            grp_keys = {c: g.iloc[0][c] for c in group_cols}
            series = g.set_index(date_col)[value_col].sort_index()
            out.append({"group": grp_keys, "series": [{"period": d.strftime("%Y-%m-%d"), "value": float(v)} for d, v in series.items()]})
        return out

    grouped = df.groupby(pd.Grouper(key=date_col, freq=freq))[value_col]
    if agg == "sum":
        res = grouped.sum()
    elif agg == "mean":
        res = grouped.mean()
    elif agg == "min":
        res = grouped.min()
    elif agg == "max":
        res = grouped.max()
    else:
        raise ValueError("unsupported agg")
    res = res.reset_index().sort_values(by=date_col)
    return [{"period": pd.to_datetime(row[date_col]).strftime("%Y-%m-%d"), "value": float(row[value_col])} for _, row in res.iterrows()]
