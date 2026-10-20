"""Dashboard configuration, filtering and KPI calculation.

The original dataset is never modified: filters always produce a new frame.
"""
from __future__ import annotations

import re
import uuid
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

from .chart_generator import (
    AGGREGATIONS, ChartConfigError, attach_correlations, recommend_charts, recommend_kpis,
    validate_chart_spec, validate_kpi_spec,
)
from .data_profiler import DATE, DIMENSION, DatasetProfile
from .formatting import format_delta, format_value
from .themes import clean_theme

SCHEMA_VERSION = 1
MAX_FILTER_OPTIONS = 300
MAX_CHARTS = 16
MAX_KPIS = 8
BLANK = "(blank)"
_ID_RE = re.compile(r"^[a-f0-9]{8,32}$")


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def new_dashboard_id() -> str:
    return uuid.uuid4().hex[:16]


# --------------------------------------------------------------------------- creation

def build_default_dashboard(profile: DatasetProfile, df: pd.DataFrame, dataset_id: str, dataset_name: str,
                            sheet: str, header_row: int | None = None) -> dict:
    attach_correlations(profile, df)
    stem = re.sub(r"\.(xlsx|xls|csv)$", "", dataset_name, flags=re.IGNORECASE)
    name = stem if sheet in (None, "", "CSV data") else f"{stem} · {sheet}"
    return {
        "schema_version": SCHEMA_VERSION,
        "id": new_dashboard_id(),
        "name": name[:120],
        "dataset_id": dataset_id,
        "dataset_name": dataset_name,
        "sheet": sheet,
        "header_row": header_row,
        "created_at": now_iso(),
        "updated_at": now_iso(),
        "kpis": recommend_kpis(profile),
        "charts": recommend_charts(profile),
        "filters": {"categorical": {}, "date": {}},
        "filter_columns": default_filter_columns(profile),
        "column_overrides": {},
    }


def regenerate(config: dict, profile: DatasetProfile, df: pd.DataFrame) -> dict:
    """Reset KPIs and charts to the automatic recommendations, keeping identity and name."""
    attach_correlations(profile, df)
    return dict(config, kpis=recommend_kpis(profile), charts=recommend_charts(profile),
                filter_columns=default_filter_columns(profile), filters={"categorical": {}, "date": {}},
                updated_at=now_iso())


def default_filter_columns(profile: DatasetProfile, limit: int = 4) -> list[str]:
    cols: list[str] = []
    dates = [d for d in profile.date_columns if not profile.columns[d].constant]
    if dates:
        cols.append(dates[0])
    cols += [d for d in profile.chartable_dimensions if profile.columns[d].unique <= 50][: limit - len(cols)]
    return cols


def filter_candidates(profile: DatasetProfile) -> list[str]:
    """Columns that can be offered as filters."""
    out = [d for d in profile.date_columns if not profile.columns[d].constant]
    out += [c.name for c in profile.columns.values()
            if c.role == DIMENSION and 2 <= c.unique <= MAX_FILTER_OPTIONS]
    return out


# --------------------------------------------------------------------------- filtering

def filter_options(df: pd.DataFrame, col: str) -> list[str]:
    s = df[col]
    labels = s.map(lambda v: BLANK if pd.isna(v) else str(v))
    counts = labels.value_counts()
    return list(counts.index[:MAX_FILTER_OPTIONS])


def apply_filters(df: pd.DataFrame, filters: dict | None, profile: DatasetProfile | None = None) -> pd.DataFrame:
    """Return the rows of ``df`` matching ``filters``. Never mutates ``df``."""
    if not filters:
        return df
    mask = pd.Series(True, index=df.index)
    for col, values in (filters.get("categorical") or {}).items():
        if col not in df.columns or not values:
            continue
        wanted = {str(v) for v in values}
        labels = df[col].map(lambda v: BLANK if pd.isna(v) else str(v))
        mask &= labels.isin(wanted)
    for col, rng in (filters.get("date") or {}).items():
        if col not in df.columns or not rng or len(rng) != 2:
            continue
        s = df[col]
        if not pd.api.types.is_datetime64_any_dtype(s):
            continue
        start, end = (pd.to_datetime(v, errors="coerce") if v else None for v in rng)
        if start is not None and not pd.isna(start):
            mask &= s >= start
        if end is not None and not pd.isna(end):
            # Inclusive of the whole end day.
            mask &= s < (end.normalize() + pd.Timedelta(days=1))
    for col, rng in (filters.get("numeric") or {}).items():
        if col not in df.columns or not rng or len(rng) != 2:
            continue
        lo, hi = rng
        s = pd.to_numeric(df[col], errors="coerce")
        if lo is not None:
            mask &= s >= float(lo)
        if hi is not None:
            mask &= s <= float(hi)
    return df[mask]


def describe_filters(filters: dict | None) -> str:
    if not filters:
        return "No filters"
    parts = []
    for col, values in (filters.get("categorical") or {}).items():
        if values:
            parts.append(f"{col}: {', '.join(map(str, values[:3]))}{'…' if len(values) > 3 else ''}")
    for col, rng in (filters.get("date") or {}).items():
        if rng and any(rng):
            parts.append(f"{col}: {rng[0] or '…'} → {rng[1] or '…'}")
    return "; ".join(parts) or "No filters"


# --------------------------------------------------------------------------- KPIs

def _agg(series: pd.Series, agg: str) -> float | None:
    if agg == "nunique":
        return float(series.nunique(dropna=True))
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return None
    return float(getattr(s, agg)())


def _period_compare(df: pd.DataFrame, value_col: str, agg: str, date_col: str) -> tuple[float | None, str | None]:
    d = df[[date_col, value_col]].dropna(subset=[date_col])
    if d.empty:
        return None, None
    span = (d[date_col].max() - d[date_col].min()).days
    grain = "M" if span >= 60 else "W"
    periods = d[date_col].dt.to_period(grain)
    latest = periods.max()
    # If the newest period is incomplete, compare the two most recent complete periods instead.
    if d[date_col].max() < latest.end_time.normalize():
        latest = latest - 1
    prev = latest - 1
    if prev < periods.min():
        return None, None
    cur_v = _agg(d.loc[periods == latest, value_col], agg)
    prev_v = _agg(d.loc[periods == prev, value_col], agg)
    if cur_v is None or prev_v is None or prev_v == 0:
        return None, None
    fmt = "%b %Y" if grain == "M" else "wk of %b %d"
    label = f"{latest.start_time.strftime(fmt)} vs {prev.start_time.strftime(fmt)}"
    return (cur_v - prev_v) / abs(prev_v) * 100, label


def compute_kpi(df: pd.DataFrame, kpi: dict, profile: DatasetProfile) -> dict:
    """Evaluate one KPI on (filtered) data. Returns display-ready values."""
    kind = kpi["kind"]
    out = {"label": kpi.get("label", ""), "value": None, "display": "—", "delta": None, "delta_label": None,
           "help": kpi.get("description", "")}
    if kind == "count":
        out["value"] = float(len(df))
        out["display"] = format_value(len(df), compact=True)
        return out
    if kind == "agg":
        col, agg = kpi["column"], kpi["agg"]
        if col not in df.columns:
            return out
        v = _agg(df[col], agg)
        cp = profile.columns.get(col)
        out["value"] = v
        pct = bool(cp and cp.is_percent and cp.stats.get("max", 0) <= 1.0 and agg not in ("nunique",))
        out["display"] = format_value(v, compact=True, percent=pct)
        if kpi.get("compare_date") and kpi["compare_date"] in df.columns and agg != "nunique":
            delta, label = _period_compare(df, col, agg, kpi["compare_date"])
            out["delta"], out["delta_label"] = format_delta(delta), label
        return out
    if kind == "ratio":
        n, d = kpi["numerator"], kpi["denominator"]
        if n not in df.columns or d not in df.columns:
            return out
        num = pd.to_numeric(df[n], errors="coerce").sum()
        den = pd.to_numeric(df[d], errors="coerce").sum()
        if den and not np.isnan(den):
            v = float(num / den)
            out["value"] = v
            out["display"] = format_value(v, decimals=2, percent=kpi.get("format") == "percent", compact=True)
        return out
    return out


# --------------------------------------------------------------------------- config validation

def validate_dashboard_config(raw: Any, profile: DatasetProfile | None = None) -> tuple[dict, list[str]]:
    """Validate a dashboard config loaded from disk.

    Structural problems raise ``ChartConfigError``. Individual charts/KPIs that no longer fit
    the dataset are dropped with a warning instead of breaking the whole dashboard.
    """
    if not isinstance(raw, dict):
        raise ChartConfigError("Dashboard file is not a JSON object.")
    warnings: list[str] = []
    did = str(raw.get("id", ""))
    if not _ID_RE.match(did):
        raise ChartConfigError("Dashboard has an invalid id.")
    dataset_id = str(raw.get("dataset_id", ""))
    if not _ID_RE.match(dataset_id):
        raise ChartConfigError("Dashboard references an invalid dataset id.")
    charts_in = raw.get("charts") if isinstance(raw.get("charts"), list) else []
    kpis_in = raw.get("kpis") if isinstance(raw.get("kpis"), list) else []
    header_row = raw.get("header_row")
    cfg = {
        "schema_version": SCHEMA_VERSION,
        "id": did,
        "name": str(raw.get("name") or "Untitled dashboard")[:120],
        "dataset_id": dataset_id,
        "dataset_name": str(raw.get("dataset_name") or "")[:255],
        "sheet": str(raw.get("sheet") or "")[:255] or None,
        "header_row": header_row if isinstance(header_row, int) and -1 <= header_row < 10_000 else None,
        "created_at": str(raw.get("created_at") or now_iso())[:40],
        "updated_at": str(raw.get("updated_at") or now_iso())[:40],
        "charts": charts_in[:MAX_CHARTS],
        "kpis": kpis_in[:MAX_KPIS],
        "filters": _clean_filters(raw.get("filters")),
        "filter_columns": [str(c) for c in (raw.get("filter_columns") or []) if isinstance(c, str)][:10],
        "column_overrides": _clean_overrides(raw.get("column_overrides")),
        # Set by the AI analyst: replayed on load so the dashboard reopens with the approved cleaning and names.
        "cleaning_ops": _clean_ops(raw.get("cleaning_ops")),
        "renames": _clean_str_map(raw.get("renames"), 80),
        "column_notes": _clean_str_map(raw.get("column_notes"), 300),
        "analysis": _clean_analysis(raw.get("analysis")),
        "theme": clean_theme(raw.get("theme")),
    }
    if profile is None:
        return cfg, warnings

    charts = []
    for i, spec in enumerate(cfg["charts"]):
        try:
            charts.append(validate_chart_spec(spec, profile))
        except ChartConfigError as exc:
            title = spec.get("title") if isinstance(spec, dict) else None
            warnings.append(f"Chart '{title or i + 1}' was skipped: {exc}")
    kpis = []
    for k in cfg["kpis"]:
        try:
            kpis.append(validate_kpi_spec(k, profile))
        except ChartConfigError as exc:
            warnings.append(f"KPI skipped: {exc}")
    cfg["charts"], cfg["kpis"] = charts, kpis
    cfg["filter_columns"] = [c for c in cfg["filter_columns"] if c in profile.columns]
    f = cfg["filters"]
    f["categorical"] = {c: v for c, v in f["categorical"].items() if c in profile.columns}
    f["date"] = {c: v for c, v in f["date"].items() if c in profile.columns and profile.columns[c].role == DATE}
    return cfg, warnings


_OVERRIDE_ROLES = {"date", "metric", "dimension", "identifier", "text", "coordinate", "empty", "hidden"}


def _clean_ops(raw: Any) -> list[dict]:
    """Structural check of stored cleaning operations (full validation happens against the data on load)."""
    from .agent.cleaning import OPERATIONS

    out = []
    for op in (raw if isinstance(raw, list) else [])[:50]:
        if isinstance(op, dict) and op.get("op") in OPERATIONS:
            col = op.get("column")
            params = op.get("params") if isinstance(op.get("params"), dict) else {}
            out.append({"op": op["op"], "column": str(col)[:255] if col else None, "params": params})
    return out


def _clean_str_map(raw: Any, limit: int) -> dict[str, str]:
    if not isinstance(raw, dict):
        return {}
    return {str(k)[:255]: str(v)[:limit] for k, v in list(raw.items())[:500] if isinstance(v, str) and v.strip()}


def _clean_analysis(raw: Any) -> dict | None:
    if not isinstance(raw, dict):
        return None
    out = {"dataset_id": str(raw.get("dataset_id") or "")[:32], "sheet": str(raw.get("sheet"))[:255] if raw.get("sheet") else None,
           "objective": str(raw.get("objective") or "")[:300], "summary": str(raw.get("summary") or "")[:400],
           "questions": [str(q)[:200] for q in (raw.get("questions") or [])[:6] if isinstance(q, str)]}
    return out if _ID_RE.match(out["dataset_id"]) else None


def _clean_overrides(raw: Any) -> dict:
    out: dict = {}
    if not isinstance(raw, dict):
        return out
    for col, ov in list(raw.items())[:500]:
        if not isinstance(col, str) or not isinstance(ov, dict):
            continue
        clean = {}
        if ov.get("role") in _OVERRIDE_ROLES:
            clean["role"] = ov["role"]
        if ov.get("aggregation") in ("sum", "mean"):
            clean["aggregation"] = ov["aggregation"]
        if clean:
            out[col[:255]] = clean
    return out


def _clean_filters(raw: Any) -> dict:
    out: dict = {"categorical": {}, "date": {}}
    if not isinstance(raw, dict):
        return out
    for col, values in (raw.get("categorical") or {}).items():
        if isinstance(col, str) and isinstance(values, list):
            out["categorical"][col] = [str(v)[:500] for v in values[:MAX_FILTER_OPTIONS]]
    for col, rng in (raw.get("date") or {}).items():
        if isinstance(col, str) and isinstance(rng, list) and len(rng) == 2:
            out["date"][col] = [str(v)[:40] if v else None for v in rng]
    return out


__all__ = [
    "AGGREGATIONS", "apply_filters", "build_default_dashboard", "compute_kpi", "describe_filters",
    "filter_candidates", "filter_options", "regenerate", "validate_dashboard_config",
]
