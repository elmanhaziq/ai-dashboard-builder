"""Number formatting helpers shared by the UI, KPIs and the query engine."""
from __future__ import annotations

import math

import pandas as pd


def format_value(v, decimals: int = 2, compact: bool = False, percent: bool = False) -> str:
    """Format a number for display. ``percent`` expects a fraction (0.25 -> 25.0%)."""
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "—"
    if isinstance(v, pd.Timestamp):
        return v.strftime("%Y-%m-%d")
    try:
        x = float(v)
    except (TypeError, ValueError):
        return str(v)
    if percent:
        return f"{x * 100:,.1f}%"
    if compact and abs(x) >= 10_000:
        for div, suffix in ((1e12, "T"), (1e9, "B"), (1e6, "M"), (1e3, "K")):
            if abs(x) >= div:
                return f"{x / div:,.2f}".rstrip("0").rstrip(".") + suffix
    if x.is_integer() and abs(x) < 1e15:
        return f"{int(x):,}"
    if abs(x) >= 100:
        decimals = min(decimals, 1) if compact else decimals
    return f"{x:,.{decimals}f}"


def format_delta(pct: float | None) -> str | None:
    if pct is None or (isinstance(pct, float) and (math.isnan(pct) or math.isinf(pct))):
        return None
    return f"{pct:+.1f}%"
