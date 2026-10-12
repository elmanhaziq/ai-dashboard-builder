"""Deterministic chart and KPI recommendations plus chart-spec validation.

Charts are stored as plain JSON-able dictionaries (``ChartSpec``), never as code.
Every spec is validated against the dataset profile before it is rendered.
"""
from __future__ import annotations

import re
import uuid
from typing import Any

import pandas as pd

from .data_profiler import COORDINATE, DATE, DIMENSION, IDENTIFIER, METRIC, DatasetProfile

CHART_TYPES = {
    "line": "Line (trend over time)",
    "area": "Area (trend over time)",
    "bar": "Bar (vertical)",
    "hbar": "Bar (horizontal)",
    "donut": "Donut (share of total)",
    "scatter": "Scatter (two measures)",
    "histogram": "Histogram (distribution)",
    "box": "Box plot (distribution)",
    "table": "Summary table",
    "heatmap": "Matrix heatmap (two categories)",
    "map": "Map (coordinates)",
}
# Short names for the visual gallery in the builder.
CHART_SHORT = {
    "line": "Line", "area": "Area", "bar": "Column", "hbar": "Bar", "donut": "Donut", "scatter": "Scatter",
    "heatmap": "Matrix", "histogram": "Histogram", "box": "Box plot", "table": "Table", "map": "Map",
}
BARMODES = {"group": "Side by side", "stack": "Stacked", "percent": "100% stacked"}
MULTI_MEASURE_TYPES = ("line", "area", "bar", "hbar")
MAX_EXTRA_MEASURES = 3
MAX_CHART_FILTER_VALUES = 50
AGGREGATIONS = {"sum": "Sum", "mean": "Average", "median": "Median", "min": "Minimum", "max": "Maximum", "count": "Count of records", "nunique": "Distinct count"}
SORTS = {"value_desc": "Largest first", "value_asc": "Smallest first", "label": "By label", "none": "Natural order"}
TIME_GRAINS = {"auto": "Automatic", "D": "Day", "W": "Week", "M": "Month", "Q": "Quarter", "Y": "Year"}
MAX_TOP_N = 50
MAX_COLOR_SERIES = 6  # extra series fold into "Other" (palette has 8 fixed slots)


class ChartConfigError(ValueError):
    """A chart configuration is invalid for the current dataset."""


def new_id() -> str:
    return uuid.uuid4().hex[:12]


def agg_label(agg: str, column: str | None) -> str:
    if agg == "count" or not column:
        return "Records"
    return f"{AGGREGATIONS.get(agg, agg.title())} of {column}"


# --------------------------------------------------------------------------- spec validation

def _role(profile: DatasetProfile, col: str | None) -> str | None:
    if col is None:
        return None
    cp = profile.columns.get(col)
    return cp.role if cp else None


def _is_time_axis(profile: DatasetProfile, col: str | None) -> bool:
    cp = profile.columns.get(col) if col else None
    return bool(cp and (cp.role == DATE or cp.is_time_part))


def _groupable(profile: DatasetProfile, col: str | None) -> bool:
    return _role(profile, col) in (DIMENSION, IDENTIFIER) or _is_time_axis(profile, col)


def validate_chart_spec(spec: Any, profile: DatasetProfile) -> dict:
    """Return a cleaned copy of ``spec`` or raise :class:`ChartConfigError`."""
    if not isinstance(spec, dict):
        raise ChartConfigError("Chart configuration must be an object.")
    ctype = spec.get("type")
    if ctype not in CHART_TYPES:
        raise ChartConfigError(f"Unsupported chart type: {ctype!r}.")

    def col(key: str) -> str | None:
        v = spec.get(key)
        if v in (None, ""):
            return None
        if not isinstance(v, str) or v not in profile.columns:
            raise ChartConfigError(f"Column {v!r} is not in this dataset.")
        return v

    x, y, color = col("x"), col("y"), col("color")
    agg = spec.get("agg") or "sum"
    if agg not in AGGREGATIONS:
        raise ChartConfigError(f"Unsupported aggregation: {agg!r}.")
    sort = spec.get("sort") or "none"
    if sort not in SORTS:
        sort = "none"
    grain = spec.get("time_grain") or "auto"
    if grain not in TIME_GRAINS:
        grain = "auto"
    try:
        top_n = int(spec.get("top_n") or 15)
    except (TypeError, ValueError):
        top_n = 15
    top_n = max(1, min(MAX_TOP_N, top_n))

    if agg not in ("count", "nunique") and y is not None and _role(profile, y) != METRIC and ctype not in ("scatter", "map"):
        raise ChartConfigError(f"'{y}' is not a numeric measure; use a count instead.")
    if agg == "nunique" and y is None:
        raise ChartConfigError("A distinct count needs a column to count.")
    if agg not in ("count",) and y is None and ctype in ("line", "area", "bar", "hbar", "donut", "heatmap"):
        agg = "count"

    if ctype in ("line", "area"):
        if not _is_time_axis(profile, x):
            raise ChartConfigError("Trend charts need a date or period column on the X axis.")
    elif ctype in ("bar", "hbar", "table"):
        if not _groupable(profile, x):
            raise ChartConfigError("Bar charts and tables need a category column to group by.")
    elif ctype == "donut":
        if _role(profile, x) != DIMENSION:
            raise ChartConfigError("Donut charts need a category column.")
        if agg not in ("sum", "count"):
            raise ChartConfigError("Donut charts show parts of a whole, so they only support sums or counts.")
        if y and profile.columns[y].stats.get("negatives", 0):
            raise ChartConfigError(f"'{y}' contains negative values, which cannot be shown as shares.")
    elif ctype == "scatter":
        if _role(profile, x) != METRIC or _role(profile, y) != METRIC or x == y:
            raise ChartConfigError("Scatter plots need two different numeric measures.")
    elif ctype == "histogram":
        if _role(profile, x) != METRIC:
            raise ChartConfigError("Histograms need a numeric measure.")
    elif ctype == "box":
        if _role(profile, y) != METRIC:
            raise ChartConfigError("Box plots need a numeric measure.")
        if x is not None and _role(profile, x) != DIMENSION:
            raise ChartConfigError("Box plots can only be split by a category column.")
    elif ctype == "map":
        if _role(profile, x) != COORDINATE or _role(profile, y) != COORDINATE:
            raise ChartConfigError("Maps need latitude and longitude columns.")
    elif ctype == "heatmap":
        x2 = col("x2")
        if not _groupable(profile, x) or not _groupable(profile, x2) or x == x2:
            raise ChartConfigError("A matrix heatmap needs two different category columns.")
    if color is not None and (_role(profile, color) != DIMENSION or color in (x, y)):
        raise ChartConfigError("Legend grouping must use a different category column.")

    # Additional measures on the same axis (Power BI "Values" well). Never a second y-axis.
    extra: list[dict] = []
    raw_extra = spec.get("extra") or []
    if raw_extra and ctype not in MULTI_MEASURE_TYPES:
        raw_extra = []
    if not isinstance(raw_extra, list):
        raise ChartConfigError("Additional values must be a list.")
    for item in raw_extra[:MAX_EXTRA_MEASURES]:
        if not isinstance(item, dict):
            raise ChartConfigError("Additional values are malformed.")
        ey, eagg = item.get("y"), item.get("agg") or "sum"
        if ey not in profile.columns or _role(profile, ey) != METRIC:
            raise ChartConfigError(f"Additional value {ey!r} is not a numeric measure.")
        if eagg not in AGGREGATIONS or eagg in ("count", "nunique"):
            raise ChartConfigError(f"Unsupported calculation {eagg!r} for '{ey}'.")
        if (ey, eagg) == (y, agg) or any(e["y"] == ey and e["agg"] == eagg for e in extra):
            continue
        extra.append({"y": ey, "agg": eagg})
    if extra and y is None:
        raise ChartConfigError("Choose a primary value before adding more values.")
    if extra and color:
        raise ChartConfigError("A legend can't be combined with several values. Remove the legend or the extra values.")

    x2 = col("x2") if ctype == "heatmap" else None
    size = col("size") if ctype == "scatter" else None
    if size is not None:
        if _role(profile, size) != METRIC:
            raise ChartConfigError("Bubble size must be a numeric measure.")
        if profile.columns[size].stats.get("negatives", 0):
            raise ChartConfigError(f"'{size}' has negative values and can't be used as bubble size.")

    barmode = spec.get("barmode") if spec.get("barmode") in BARMODES else "group"
    if barmode == "percent" and agg not in ("sum", "count") and ctype in ("bar", "hbar", "area"):
        raise ChartConfigError("100% stacking shows shares of a total, so it needs sums or counts.")

    chart_filters: dict[str, list[str]] = {}
    raw_filters = spec.get("chart_filters") or {}
    if not isinstance(raw_filters, dict):
        raise ChartConfigError("Visual filters are malformed.")
    for fcol, values in raw_filters.items():
        if fcol not in profile.columns or not (_groupable(profile, fcol)):
            raise ChartConfigError(f"Visual filter column {fcol!r} is not a category in this dataset.")
        if not isinstance(values, list):
            raise ChartConfigError("Visual filter values must be a list.")
        vals = [str(v)[:500] for v in values[:MAX_CHART_FILTER_VALUES]]
        if vals:
            chart_filters[fcol] = vals

    origin = spec.get("origin") if isinstance(spec.get("origin"), dict) else None
    if origin:
        origin = {k: v for k, v in origin.items() if k != "origin"}

    title = str(spec.get("title") or "").strip()[:120] or default_title(ctype, x, y, agg, color, extra, x2)
    return {
        "id": str(spec.get("id") or new_id())[:32],
        "type": ctype,
        "title": title,
        "description": str(spec.get("description") or "")[:600],
        "x": x, "y": y, "color": color, "agg": agg, "sort": sort,
        "top_n": top_n, "time_grain": grain,
        "extra": extra, "x2": x2, "size": size, "barmode": barmode,
        "labels": bool(spec.get("labels", False)),
        "title_custom": bool(spec.get("title_custom", False)),
        "width": "full" if spec.get("width") == "full" else "half",
        "chart_filters": chart_filters,
        "auto": bool(spec.get("auto", False)),
        "origin": origin,
    }


def default_title(ctype: str, x: str | None, y: str | None, agg: str, color: str | None = None,
                  extra: list[dict] | None = None, x2: str | None = None) -> str:
    measure = agg_label(agg, y)
    if extra:
        names = [y] + [e["y"] for e in extra]
        measure = ", ".join(names[:-1]) + f" and {names[-1]}"
    if ctype == "heatmap":
        return f"{measure} by {x} and {x2}"
    if ctype in ("line", "area"):
        return f"{measure} over time" + (f" by {color}" if color else "")
    if ctype in ("bar", "hbar", "donut"):
        return f"{measure} by {x}"
    if ctype == "scatter":
        return f"{y} vs {x}"
    if ctype == "histogram":
        return f"Distribution of {x}"
    if ctype == "box":
        return f"Spread of {y}" + (f" by {x}" if x else "")
    if ctype == "table":
        return f"Summary by {x}"
    if ctype == "map":
        return "Locations"
    return measure


def describe_chart(spec: dict, profile: DatasetProfile) -> str:
    """Plain-language explanation of exactly how a chart is calculated."""
    t, x, y, agg = spec["type"], spec.get("x"), spec.get("y"), spec.get("agg", "sum")
    measure = "the number of records" if agg == "count" or not y else f"the {AGGREGATIONS[agg].lower()} of '{y}'"
    if t in ("line", "area"):
        grain = TIME_GRAINS.get(spec.get("time_grain", "auto"), "Automatic").lower()
        text = f"Shows {measure} per {grain if grain != 'automatic' else 'period'} of '{x}'."
    elif t in ("bar", "hbar"):
        text = f"Shows {measure} for each value of '{x}' (top {spec.get('top_n', 15)})."
    elif t == "donut":
        text = f"Shows each '{x}' value's share of {measure}; smaller groups are combined into 'Other'."
    elif t == "scatter":
        text = f"Each point is one row, plotting '{y}' against '{x}'."
    elif t == "histogram":
        text = f"Shows how values of '{x}' are distributed across ranges."
    elif t == "box":
        text = f"Shows the median, quartiles and outliers of '{y}'" + (f" for each '{x}'." if x else ".")
    elif t == "table":
        text = f"Record counts and each measure's default aggregation, grouped by '{x}'."
    elif t == "heatmap":
        text = f"Shows {measure} for every combination of '{x}' (columns) and '{spec.get('x2')}' (rows); darker cells are higher."
    else:
        text = "Each point is one row positioned by its coordinates."
    if spec.get("extra"):
        parts = ", ".join(f"the {AGGREGATIONS[e['agg']].lower()} of '{e['y']}'" for e in spec["extra"])
        text += f" Also shows {parts} on the same axis."
    if spec.get("size"):
        text += f" Bubble size shows '{spec['size']}'."
    if spec.get("color"):
        text += f" Coloured by '{spec['color']}'."
    if spec.get("barmode") in ("stack", "percent") and t in ("bar", "hbar", "area") and (spec.get("color") or spec.get("extra")):
        text += " Stacked" + (" to 100% so each bar shows shares." if spec["barmode"] == "percent" else ".")
    if spec.get("chart_filters"):
        conds = "; ".join(f"{c} = {', '.join(v[:3])}{'…' if len(v) > 3 else ''}" for c, v in spec["chart_filters"].items())
        text += f" Visual filter: {conds}."
    if y and agg not in ("count",) and profile.columns.get(y) and profile.columns[y].missing:
        text += f" {profile.columns[y].missing:,} missing value(s) in '{y}' are excluded."
    return text


def compatible_types(profile: DatasetProfile) -> list[str]:
    """Chart types that can be built at all from this dataset."""
    out = []
    has_metric = bool(profile.metric_columns)
    if profile.date_columns or any(c.is_time_part for c in profile.columns.values()):
        out += ["line", "area"]
    if any(_groupable(profile, c) for c in profile.columns):
        out += ["bar", "hbar", "table"]
    if any(c.role == DIMENSION for c in profile.columns.values()):
        out.append("donut")
    if sum(1 for c in profile.columns if _groupable(profile, c)) >= 2:
        out.append("heatmap")
    if len(profile.metric_columns) >= 2:
        out.append("scatter")
    if has_metric:
        out += ["histogram", "box"]
    if profile.coordinate_columns():
        out.append("map")
    return out


def adapt_spec(spec: dict, new_type: str, profile: DatasetProfile) -> dict:
    """Switch a chart to ``new_type``, keeping compatible columns and filling the rest sensibly."""
    s = dict(spec, type=new_type)
    metrics = profile.metric_columns
    dims = profile.chartable_dimensions or profile.dimension_columns
    times = profile.date_columns + [c.name for c in profile.columns.values() if c.is_time_part]
    y = s.get("y") if s.get("y") in metrics else (metrics[0] if metrics else None)
    if new_type in ("line", "area"):
        s["x"] = s["x"] if _is_time_axis(profile, s.get("x")) else (times[0] if times else None)
        s["y"] = y
    elif new_type in ("bar", "hbar", "table", "donut"):
        pool = [d for d in dims if new_type != "donut" or profile.columns[d].role == DIMENSION]
        s["x"] = s["x"] if s.get("x") in pool or (new_type != "donut" and _groupable(profile, s.get("x"))) else (pool[0] if pool else None)
        s["y"] = y
        if new_type == "donut" and s.get("agg") not in ("sum", "count"):
            s["agg"] = "sum" if y else "count"
    elif new_type == "scatter":
        s["x"] = s["x"] if s.get("x") in metrics else (metrics[0] if metrics else None)
        others = [m for m in metrics if m != s["x"]]
        s["y"] = s["y"] if s.get("y") in others else (others[0] if others else None)
    elif new_type == "histogram":
        s["x"] = s["x"] if s.get("x") in metrics else y
        s["y"] = None
    elif new_type == "box":
        s["y"] = y
        s["x"] = s["x"] if profile.columns.get(s.get("x") or "") and profile.columns[s["x"]].role == DIMENSION else None
    elif new_type == "heatmap":
        pool = [d for d in dims if profile.columns[d].unique <= 50] or dims
        s["x"] = s["x"] if s.get("x") in pool else (pool[0] if pool else None)
        rest = [d for d in pool if d != s["x"]]
        s["x2"] = s.get("x2") if s.get("x2") in rest else (s.get("color") if s.get("color") in rest else (rest[0] if rest else None))
        s["y"] = y
        s["color"] = None
    elif new_type == "map":
        coords = profile.coordinate_columns()
        if coords:
            s["y"], s["x"] = coords
    if new_type not in MULTI_MEASURE_TYPES:
        s["extra"] = []
    if new_type != "scatter":
        s["size"] = None
    if new_type != "heatmap":
        s["x2"] = None
    if s.get("y") is None and new_type in ("line", "area", "bar", "hbar", "donut", "heatmap"):
        s["agg"] = "count"
    elif s.get("agg") == "count" and s.get("y") and new_type in ("line", "area", "bar", "hbar", "heatmap"):
        s["agg"] = profile.columns[s["y"]].aggregation if s["y"] in profile.columns else "sum"
    if s.get("color") in (s.get("x"), s.get("y")) or s.get("extra"):
        s["color"] = None
    if new_type == "donut" and s.get("agg") not in ("sum", "count"):
        s["agg"] = "sum" if s.get("y") else "count"
    s["title"] = ""
    return s


# --------------------------------------------------------------------------- recommendations

def _time_grain_for(profile: DatasetProfile, date_col: str) -> str:
    span = profile.columns[date_col].stats.get("span_days", 0) or 0
    if span <= 45:
        return "D"
    if span <= 200:
        return "W"
    if span <= 1100:
        return "M"
    if span <= 3650:
        return "Q"
    return "Y"


def recommend_charts(profile: DatasetProfile, max_charts: int = 8) -> list[dict]:
    """Rule-based chart recommendations. Deterministic for a given profile."""
    charts: list[dict] = []
    seen: set[tuple] = set()

    def add(spec: dict) -> None:
        key = (spec["type"], spec.get("x"), spec.get("y"), spec.get("color"))
        if key in seen or len(charts) >= max_charts:
            return
        try:
            clean = validate_chart_spec(dict(spec, auto=True), profile)
        except ChartConfigError:
            return
        clean["description"] = clean["description"] or describe_chart(clean, profile)
        seen.add(key)
        charts.append(clean)

    metrics = [m for m in profile.metric_columns if not profile.columns[m].constant]
    dims = profile.chartable_dimensions
    dates = [d for d in profile.date_columns if not profile.columns[d].constant and profile.columns[d].unique >= 3]
    time_parts = [c.name for c in profile.columns.values() if c.is_time_part and c.unique >= 2]
    primary = metrics[0] if metrics else None
    p_agg = profile.columns[primary].aggregation if primary else "count"
    secondary = metrics[1] if len(metrics) > 1 else None

    # 1. Trend of the headline measure (or record count) over time.
    time_axis = dates[0] if dates else (time_parts[0] if time_parts else None)
    if time_axis:
        grain = _time_grain_for(profile, time_axis) if time_axis in dates else "auto"
        add({"type": "line", "x": time_axis, "y": primary, "agg": p_agg if primary else "count", "time_grain": grain})

    # 2. Headline measure by the best categories.
    for d in dims[:2]:
        if primary:
            card = profile.columns[d].unique
            add({"type": "hbar" if card > 8 else "bar", "x": d, "y": primary, "agg": p_agg, "sort": "value_desc", "top_n": 15})
        else:
            add({"type": "hbar", "x": d, "agg": "count", "sort": "value_desc", "top_n": 15})

    # 3. Share of total for a low-cardinality category (only for additive measures).
    donut_dim = next((d for d in dims if 2 <= profile.columns[d].unique <= 6), None)
    if donut_dim:
        if primary and p_agg == "sum" and not profile.columns[primary].stats.get("negatives"):
            add({"type": "donut", "x": donut_dim, "y": primary, "agg": "sum"})
        elif not primary or p_agg != "sum":
            add({"type": "donut", "x": donut_dim, "agg": "count"})

    # 4. Second measure by the top category.
    if secondary and dims:
        s_agg = profile.columns[secondary].aggregation
        add({"type": "bar" if profile.columns[dims[0]].unique <= 8 else "hbar", "x": dims[0], "y": secondary, "agg": s_agg, "sort": "value_desc", "top_n": 15})

    # 5. Relationship between the two most correlated measures.
    pair = _best_scatter_pair(profile, metrics)
    if pair:
        add({"type": "scatter", "x": pair[0], "y": pair[1], "color": dims[0] if dims and profile.columns[dims[0]].unique <= 6 else None})

    # 5b. Two small categories against each other.
    small = [d for d in dims if profile.columns[d].unique <= 12]
    if len(small) >= 2:
        add({"type": "heatmap", "x": small[0], "x2": small[1], "y": primary, "agg": p_agg if primary else "count"})

    # 6. Distribution of the headline measure.
    if primary:
        if dims and profile.columns[dims[0]].unique <= 8:
            add({"type": "box", "x": dims[0], "y": primary})
        add({"type": "histogram", "x": primary})

    # 7. Trend split by the leading category.
    if time_axis and dims and profile.columns[dims[0]].unique <= 8 and primary:
        add({"type": "line", "x": time_axis, "y": primary, "agg": p_agg, "color": dims[0],
             "time_grain": _time_grain_for(profile, time_axis) if time_axis in dates else "auto"})

    # 8. Coordinates map.
    coords = profile.coordinate_columns()
    if coords:
        add({"type": "map", "x": coords[1], "y": coords[0], "color": dims[0] if dims else None})

    # 9. Summary table.
    if dims:
        add({"type": "table", "x": dims[0], "y": primary, "agg": p_agg if primary else "count", "sort": "value_desc", "top_n": 20})

    # Fallback for datasets with only high-cardinality text/ids.
    if not charts:
        high = [d for d in profile.dimension_columns if profile.columns[d].unique >= 2]
        if high:
            add({"type": "hbar", "x": high[0], "agg": "count", "sort": "value_desc", "top_n": 15})
        elif primary:
            add({"type": "histogram", "x": primary})
    return charts


def _best_scatter_pair(profile: DatasetProfile, metrics: list[str]) -> tuple[str, str] | None:
    corr = profile.__dict__.get("_correlations")
    if not corr or len(metrics) < 2:
        return None
    best, best_val = None, 0.3
    for (a, b), r in corr.items():
        if a in metrics and b in metrics and abs(r) > best_val:
            best, best_val = (a, b), abs(r)
    return best


def attach_correlations(profile: DatasetProfile, df: pd.DataFrame, limit: int = 8) -> None:
    """Pre-compute pairwise correlations between the top measures (used by recommendations)."""
    metrics = [m for m in profile.metric_columns if not profile.columns[m].constant][:limit]
    corr: dict[tuple[str, str], float] = {}
    if len(metrics) >= 2 and len(df) >= 10:
        m = df[metrics].apply(pd.to_numeric, errors="coerce").corr(min_periods=10)
        for i, a in enumerate(metrics):
            for b in metrics[i + 1:]:
                v = m.loc[a, b]
                if pd.notna(v):
                    corr[(a, b)] = float(v)
    profile.__dict__["_correlations"] = corr


# --------------------------------------------------------------------------- KPIs

_SNAPSHOT_DATE_RE = re.compile(r"last|next|due|expir|birth|\bdob\b|hire|joined|start|end|creat|updat|modif|restock", re.IGNORECASE)

_RATIO_RULES = [
    ("Click-through rate", r"clicks?", r"impressions?", "percent", "Sum of {n} ÷ sum of {d}"),
    ("Conversion rate", r"conversions?", r"clicks?", "percent", "Sum of {n} ÷ sum of {d}"),
    ("Cost per click", r"(cost|spend|ad spend|amount spent)", r"clicks?", "number", "Sum of {n} ÷ sum of {d}"),
    ("Cost per conversion", r"(cost|spend|ad spend|amount spent)", r"conversions?", "number", "Sum of {n} ÷ sum of {d}"),
    ("Profit margin", r"(gross |net )?profit", r"(revenue|sales|net sales)", "percent", "Sum of {n} ÷ sum of {d}"),
]


def _norm_name(name: str) -> str:
    return re.sub(r"\s*[\(\[].*?[\)\]]\s*", " ", name).strip().lower().replace("_", " ")


def recommend_kpis(profile: DatasetProfile, max_kpis: int = 6) -> list[dict]:
    kpis: list[dict] = [{
        "id": new_id(), "kind": "count", "label": "Records", "column": None, "agg": "count",
        "description": "Number of rows after filters.",
    }]
    metrics = [m for m in profile.metric_columns if not profile.columns[m].constant]
    # Period-over-period deltas only on event-style dates (not "Last restock", "Due date", "Hire date"...).
    date_col = next((d for d in profile.date_columns if profile.columns[d].stats.get("span_days", 0) >= 14
                     and not _SNAPSHOT_DATE_RE.search(d)), None)

    for m in metrics[:3]:
        cp = profile.columns[m]
        agg = cp.aggregation
        kpis.append({
            "id": new_id(), "kind": "agg", "label": agg_label(agg, m), "column": m, "agg": agg,
            "compare_date": date_col if agg == "sum" else None,
            "description": f"{AGGREGATIONS[agg]} of '{m}' across filtered rows"
                           + (f"; {cp.missing:,} missing values excluded." if cp.missing else "."),
        })

    # Ratios only when both columns are unambiguously identified.
    norm = {m: _norm_name(m) for m in metrics if profile.columns[m].aggregation == "sum"}
    for label, num_re, den_re, fmt, desc in _RATIO_RULES:
        num = [m for m, n in norm.items() if re.fullmatch(num_re, n)]
        den = [m for m, n in norm.items() if re.fullmatch(den_re, n)]
        if len(num) == 1 and len(den) == 1 and num[0] != den[0]:
            kpis.append({
                "id": new_id(), "kind": "ratio", "label": label, "numerator": num[0], "denominator": den[0],
                "format": fmt, "description": desc.format(n=num[0], d=den[0]) + " (filtered rows).",
            })

    # A distinct count of the leading category or identifier.
    distinct = next((d for d in profile.identifier_columns if profile.columns[d].unique < profile.n_rows), None) \
        or next(iter(profile.chartable_dimensions), None)
    if distinct:
        kpis.append({
            "id": new_id(), "kind": "agg", "label": f"Distinct {distinct}", "column": distinct, "agg": "nunique",
            "description": f"Number of different '{distinct}' values in the filtered rows.",
        })
    # Keep ratios ahead of the trailing distinct count when trimming.
    return kpis[:max_kpis]


def validate_kpi_spec(kpi: Any, profile: DatasetProfile) -> dict:
    if not isinstance(kpi, dict):
        raise ChartConfigError("KPI configuration must be an object.")
    kind = kpi.get("kind")
    label = str(kpi.get("label") or "")[:80]
    base = {"id": str(kpi.get("id") or new_id())[:32], "kind": kind, "label": label,
            "description": str(kpi.get("description") or "")[:300]}
    if kind == "count":
        return dict(base, label=label or "Records", agg="count", column=None)
    if kind == "agg":
        col, agg = kpi.get("column"), kpi.get("agg")
        if col not in profile.columns or agg not in AGGREGATIONS or agg == "count":
            raise ChartConfigError(f"KPI '{label}' references an unknown column or aggregation.")
        if agg != "nunique" and profile.columns[col].role != METRIC:
            raise ChartConfigError(f"KPI '{label}' needs a numeric column.")
        cmp = kpi.get("compare_date")
        cmp = cmp if cmp in profile.columns and profile.columns[cmp].role == DATE else None
        return dict(base, label=label or agg_label(agg, col), column=col, agg=agg, compare_date=cmp)
    if kind == "ratio":
        n, d = kpi.get("numerator"), kpi.get("denominator")
        if n not in profile.columns or d not in profile.columns or METRIC not in (profile.columns[n].role,) or profile.columns[d].role != METRIC:
            raise ChartConfigError(f"KPI '{label}' references unknown columns.")
        return dict(base, numerator=n, denominator=d, format="percent" if kpi.get("format") == "percent" else "number")
    raise ChartConfigError(f"Unsupported KPI type: {kind!r}.")
