"""Deterministic, calculation-backed insights about a dataset.

Every number in an insight is computed with pandas from the (filtered) data.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .chart_generator import agg_label
from .data_profiler import DatasetProfile
from .formatting import format_value


@dataclass
class Insight:
    kind: str  # overview | quality | concentration | trend | outlier | correlation
    title: str
    text: str


def _fmt(v: float) -> str:
    return format_value(v, compact=True)


def period_series(df: pd.DataFrame, date_col: str, metric: str | None, agg: str) -> tuple[pd.Series, str, bool]:
    """Aggregate a metric by period. Returns (series indexed by Period, grain, last_period_partial)."""
    d = df[[date_col] + ([metric] if metric else [])].dropna(subset=[date_col])
    span = (d[date_col].max() - d[date_col].min()).days if len(d) else 0
    grain = "D" if span <= 31 else "W" if span <= 120 else "M" if span <= 1100 else "Q" if span <= 3650 else "Y"
    periods = d[date_col].dt.to_period(grain)
    if metric and agg != "count":
        s = d.groupby(periods)[metric].agg(agg)
    else:
        s = d.groupby(periods).size().astype(float)
    s = s.sort_index()
    partial = bool(len(s)) and d[date_col].max() < s.index.max().end_time.normalize()
    return s, grain, partial


def period_label(p: pd.Period) -> str:
    f = p.freqstr[0] if p.freqstr else "D"
    if f == "M":
        return p.strftime("%b %Y")
    if f == "Q":
        return f"Q{p.quarter} {p.year}"
    if f in ("Y", "A"):
        return str(p.year)
    if f == "W":
        return "week of " + p.start_time.strftime("%b %d, %Y")
    return p.strftime("%b %d, %Y")


def trend_insight(df: pd.DataFrame, profile: DatasetProfile, metric: str | None, date_col: str) -> Insight | None:
    agg = profile.columns[metric].aggregation if metric else "count"
    s, grain, partial = period_series(df, date_col, metric, agg)
    label = agg_label(agg, metric)
    if partial and len(s) > 3:
        s = s.iloc[:-1]  # do not compare against an incomplete final period
    s = s.dropna()
    if len(s) < 3:
        return None
    first, last = s.iloc[0], s.iloc[-1]
    x = np.arange(len(s))
    slope = np.polyfit(x, s.values.astype(float), 1)[0]
    mean = float(s.mean()) or 1.0
    rel = slope * (len(s) - 1) / abs(mean) * 100
    direction = "upward" if rel > 10 else "downward" if rel < -10 else "broadly flat"
    peak = s.idxmax()
    trough = s.idxmin()
    text = (
        f"{label} shows {'an' if direction == 'upward' else 'a'} {direction} trend across {len(s)} {dict(D='days', W='weeks', M='months', Q='quarters', Y='years')[grain]} "
        f"({period_label(s.index[0])}: {_fmt(first)} → {period_label(s.index[-1])}: {_fmt(last)}). "
        f"Highest: {period_label(peak)} ({_fmt(s[peak])}); lowest: {period_label(trough)} ({_fmt(s[trough])})."
    )
    if partial:
        text += " The most recent period is incomplete and was excluded."
    return Insight("trend", f"Trend in {label}", text)


def outlier_insight(df: pd.DataFrame, metric: str) -> Insight | None:
    v = pd.to_numeric(df[metric], errors="coerce").dropna()
    if len(v) < 20:
        return None
    q1, q3 = np.percentile(v, [25, 75])
    iqr = q3 - q1
    if iqr <= 0:
        return None
    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    high, low = int((v > hi).sum()), int((v < lo).sum())
    if high + low == 0:
        return None
    parts = []
    if high:
        parts.append(f"{high:,} unusually high (above {_fmt(hi)}, max {_fmt(v.max())})")
    if low:
        parts.append(f"{low:,} unusually low (below {_fmt(lo)}, min {_fmt(v.min())})")
    return Insight("outlier", f"Outliers in {metric}",
                   f"{' and '.join(parts)} value(s) of '{metric}' out of {len(v):,} — {(high + low) / len(v):.1%} of rows, "
                   "using the 1.5 × IQR rule.")


def generate_insights(df: pd.DataFrame, profile: DatasetProfile, max_items: int = 8) -> list[Insight]:
    out: list[Insight] = []
    n = len(df)
    if n == 0:
        return [Insight("overview", "No data", "No rows match the current filters.")]

    dates = [d for d in profile.date_columns if d in df.columns and not profile.columns[d].constant]
    text = f"{n:,} rows and {profile.n_cols} columns"
    if dates:
        d = df[dates[0]].dropna()
        if len(d):
            text += f"; '{dates[0]}' runs from {d.min():%b %d, %Y} to {d.max():%b %d, %Y}"
    roles = f"{len(profile.metric_columns)} measures, {len(profile.dimension_columns)} categories, {len(profile.date_columns)} date fields"
    out.append(Insight("overview", "Dataset at a glance", f"{text}. Detected {roles}."))

    metrics = [m for m in profile.metric_columns if m in df.columns and not profile.columns[m].constant]
    dims = [d for d in profile.chartable_dimensions if d in df.columns]
    primary = metrics[0] if metrics else None

    # Concentration: share of the headline additive measure held by the top category.
    if primary and dims and profile.columns[primary].aggregation == "sum":
        dim = dims[0]
        g = df.groupby(df[dim].astype(str), observed=True)[primary].sum(min_count=1).dropna().sort_values(ascending=False)
        total = g.sum()
        if len(g) >= 2 and total > 0 and (g >= 0).all():
            share = g.iloc[0] / total
            out.append(Insight("concentration", f"Largest {dim}",
                               f"'{g.index[0]}' contributes {share:.1%} of the total '{primary}' ({_fmt(g.iloc[0])} of {_fmt(total)}), "
                               f"ahead of '{g.index[1]}' ({g.iloc[1] / total:.1%}). {len(g)} '{dim}' values in total."))
            if len(g) >= 10:
                k = max(1, int(round(len(g) * 0.2)))
                out.append(Insight("concentration", "Concentration",
                                   f"The top 20% of '{dim}' values ({k}) account for {g.iloc[:k].sum() / total:.1%} of total '{primary}'."))
    elif dims:
        dim = dims[0]
        vc = df[dim].astype(str).value_counts()
        if len(vc) >= 2:
            out.append(Insight("concentration", f"Most common {dim}",
                               f"'{vc.index[0]}' is the most frequent '{dim}' ({vc.iloc[0]:,} rows, {vc.iloc[0] / n:.1%})."))

    if dates:
        ins = trend_insight(df, profile, primary, dates[0])
        if ins:
            out.append(ins)

    for m in metrics[:3]:
        ins = outlier_insight(df, m)
        if ins:
            out.append(ins)

    if len(metrics) >= 2 and n >= 10:
        corr = df[metrics[:8]].apply(pd.to_numeric, errors="coerce").corr(min_periods=10)
        best, best_v = None, 0.6
        for i, a in enumerate(corr.columns):
            for b in corr.columns[i + 1:]:
                v = corr.loc[a, b]
                if pd.notna(v) and abs(v) >= best_v and abs(v) < 0.999:
                    best, best_v = (a, b, v), abs(v)
        if best:
            a, b, v = best
            out.append(Insight("correlation", "Related measures",
                               f"'{a}' and '{b}' move {'together' if v > 0 else 'in opposite directions'} "
                               f"(correlation r = {v:.2f}). Correlation does not imply that one causes the other."))

    quality = []
    if profile.duplicate_rows:
        quality.append(f"{profile.duplicate_rows:,} duplicate rows")
    worst = sorted((c for c in profile.columns.values() if c.missing), key=lambda c: -c.missing)
    if worst:
        quality.append(f"'{worst[0].name}' is missing {worst[0].missing_pct:.1f}% of values")
    if quality:
        out.append(Insight("quality", "Data quality", "; ".join(quality).capitalize() + " (whole dataset)."))
    return out[:max_items]
