"""Turn validated chart specs into aggregated data frames and Plotly figures."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from .chart_generator import AGGREGATIONS, MAX_COLOR_SERIES, agg_label
from .config import SERIES_COLORS
from .data_profiler import DATE, DatasetProfile
from .formatting import format_value

INK = "#1c1f24"
INK_2 = "#5b616b"
INK_3 = "#8a9099"
GRID = "#eceef1"
SURFACE = "#ffffff"
OTHER_COLOR = "#b8bdc6"
OTHER = "Other"
SERIES = "series"  # long-format column holding the measure name for multi-value charts
# Sequential single-hue ramp (light → dark) for magnitude in heatmaps.
SEQUENTIAL = [[0.0, "#eef4fc"], [0.35, "#a9c8f0"], [0.7, "#4a8fe0"], [1.0, "#174a91"]]
SCALE_WARNING_RATIO = 20


class ChartDataError(Exception):
    """Raised when a chart has nothing meaningful to show for the current data."""


def _resolve_grain(df: pd.DataFrame, x: str, grain: str) -> str:
    if grain != "auto":
        return grain
    v = df[x].dropna()
    span = (v.max() - v.min()).days if len(v) else 0
    return "D" if span <= 45 else "W" if span <= 200 else "M" if span <= 1100 else "Q" if span <= 3650 else "Y"


def _aggregate(frame: pd.DataFrame, keys: list[str], y: str | None, agg: str) -> pd.DataFrame:
    g = frame.groupby(keys, dropna=False, observed=True, sort=False)
    if agg == "count" or y is None:
        return g.size().rename("value").reset_index()
    return g[y].agg(agg).rename("value").reset_index()


def measures_of(spec: dict) -> list[tuple[str | None, str]]:
    """All (column, aggregation) pairs a chart plots, primary first."""
    y, agg = spec.get("y"), spec.get("agg", "sum")
    first = [(None, "count")] if (y is None or agg == "count") else [(y, agg)]
    return first + [(e["y"], e["agg"]) for e in (spec.get("extra") or [])]


def _aggregate_long(frame: pd.DataFrame, keys: list[str], measures: list[tuple[str | None, str]]) -> pd.DataFrame:
    parts = []
    for col, agg in measures:
        part = _aggregate(frame, keys, col, agg)
        part[SERIES] = agg_label(agg, col)
        parts.append(part)
    return pd.concat(parts, ignore_index=True)


def _fold_other(frame: pd.DataFrame, col: str, y: str | None, agg: str, keep: int) -> tuple[pd.DataFrame, int]:
    """Keep the ``keep`` largest groups of ``col`` (by sum/count) and fold the rest into 'Other'."""
    weight = frame.groupby(col, observed=True)[y].sum(min_count=1).abs() if (y and agg in ("sum", "mean")) else frame.groupby(col, observed=True).size()
    weight = weight.sort_values(ascending=False)
    if len(weight) <= keep:
        return frame, 0
    top = set(weight.index[:keep])
    out = frame.copy()
    out[col] = out[col].where(out[col].isin(top), OTHER)
    return out, len(weight) - keep


def _labels(series: pd.Series) -> pd.Series:
    return series.map(lambda v: "(blank)" if pd.isna(v) else str(v))


def apply_chart_filters(df: pd.DataFrame, spec: dict) -> pd.DataFrame:
    """Visual-level filters: apply only to this chart, on top of the page filters."""
    filters = spec.get("chart_filters") or {}
    if not filters:
        return df
    mask = pd.Series(True, index=df.index)
    for col, values in filters.items():
        if col in df.columns and values:
            mask &= _labels(df[col]).isin(set(values))
    return df[mask]


def _category_order(totals: pd.Series, sort: str, natural: bool) -> list:
    if sort == "label" or (sort == "none" and natural):
        return sorted(totals.index, key=_natural_key)
    if sort == "value_asc":
        return list(totals.sort_values().index)
    if sort == "none":
        return list(totals.index)
    return list(totals.sort_values(ascending=False).index)


def _scale_note(out: pd.DataFrame) -> list[str]:
    if SERIES not in out.columns or out[SERIES].nunique() < 2:
        return []
    peaks = out.groupby(SERIES)["value"].apply(lambda v: v.abs().max()).replace(0, np.nan).dropna()
    if len(peaks) >= 2 and peaks.max() / peaks.min() > SCALE_WARNING_RATIO:
        return ["These values have very different scales, so smaller ones may look flat. Consider separate charts."]
    return []


def prepare_chart_data(df: pd.DataFrame, spec: dict, profile: DatasetProfile) -> tuple[pd.DataFrame, list[str]]:
    """Aggregate ``df`` for ``spec``. Returns (data, notes shown under the chart)."""
    t, x, y, color, agg = spec["type"], spec.get("x"), spec.get("y"), spec.get("color"), spec.get("agg", "sum")
    extra = spec.get("extra") or []
    notes: list[str] = []
    if df.empty:
        raise ChartDataError("No rows match the current filters.")
    needed = [c for c in [x, y, color, spec.get("x2"), spec.get("size")] + [e["y"] for e in extra] if c]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ChartDataError(f"Column(s) not found: {', '.join(missing)}.")
    if spec.get("chart_filters"):
        df = apply_chart_filters(df, spec)
        if df.empty:
            raise ChartDataError("No rows match this visual's own filters.")
    cols = list(dict.fromkeys(needed))

    if t in ("line", "area"):
        frame = df[cols].copy()
        frame = frame[frame[x].notna()]
        if y and agg != "count" and not extra:
            frame = frame[frame[y].notna()]
        if frame.empty:
            raise ChartDataError("No rows have both a date and a value.")
        if profile.columns[x].role == DATE:
            grain = _resolve_grain(frame, x, spec.get("time_grain", "auto"))
            frame["period"] = frame[x].dt.to_period(grain).dt.start_time
            notes.append(f"Grouped by {dict(D='day', W='week', M='month', Q='quarter', Y='year')[grain]}.")
        else:
            frame["period"] = frame[x]
        if extra:
            out = _aggregate_long(frame, ["period"], measures_of(spec)).sort_values("period")
            return out, notes + _scale_note(out)
        keys = ["period"]
        if color:
            frame, folded = _fold_other(frame, color, y, agg, MAX_COLOR_SERIES)
            if folded:
                notes.append(f"{folded} smaller '{color}' groups are combined into 'Other'.")
            frame[color] = _labels(frame[color])
            keys.append(color)
        return _aggregate(frame, keys, y, agg).sort_values("period"), notes

    if t == "table":
        metrics = [m for m in profile.metric_columns if m in df.columns][:6]
        frame = df[[x] + metrics].copy()
        frame[x] = _labels(frame[x])
        g = frame.groupby(x, observed=True)
        out = g.size().rename("Records").to_frame()
        for m in metrics:
            a = profile.columns[m].aggregation
            out[agg_label(a, m)] = g[m].agg(a)
        sort_col = agg_label(spec.get("agg"), y) if y and agg_label(spec.get("agg"), y) in out.columns else "Records"
        out = out.sort_values(sort_col, ascending=spec.get("sort") == "value_asc").reset_index()
        if len(out) > spec.get("top_n", 20):
            notes.append(f"Showing {spec.get('top_n', 20)} of {len(out):,} groups.")
            out = out.head(spec.get("top_n", 20))
        return out, notes

    if t in ("bar", "hbar", "donut"):
        frame = df[cols].copy()
        if y and agg not in ("count",) and not extra:
            frame = frame[frame[y].notna()]
            if frame.empty:
                raise ChartDataError(f"'{y}' has no values in the filtered rows.")
        frame[x] = _labels(frame[x])
        if t == "donut":
            frame, folded = _fold_other(frame, x, y, agg, 7)
            if folded:
                notes.append(f"{folded} smaller groups are combined into 'Other'.")
            out = _aggregate(frame, [x], y, agg)
            out = out[out["value"] > 0].sort_values("value", ascending=False)
            if out.empty:
                raise ChartDataError("There are no positive values to show as shares.")
            return out, notes
        if extra:
            out = _aggregate_long(frame, [x], measures_of(spec))
            first_col, first_agg = measures_of(spec)[0]
            first = agg_label(first_agg, first_col)
            totals = out[out[SERIES] == first].set_index(x)["value"]
            notes += _scale_note(out)
        else:
            keys = [x]
            if color:
                frame, folded = _fold_other(frame, color, y, agg, MAX_COLOR_SERIES)
                if folded:
                    notes.append(f"{folded} smaller '{color}' groups are combined into 'Other'.")
                frame[color] = _labels(frame[color])
                keys.append(color)
            out = _aggregate(frame, keys, y, agg)
            totals = out.groupby(x)["value"].sum() if color else out.set_index(x)["value"]
        order = _category_order(totals, spec.get("sort", "value_desc"), profile.columns[x].is_time_part)
        top_n = spec.get("top_n", 15)
        if len(order) > top_n:
            notes.append(f"Showing {top_n} of {len(order):,} '{x}' values.")
            order = order[:top_n]
        out = out[out[x].isin(order)].copy()
        out[x] = pd.Categorical(out[x], categories=order, ordered=True)
        return out.sort_values(x), notes

    if t == "heatmap":
        x2 = spec["x2"]
        frame = df[cols].copy()
        if y and agg != "count":
            frame = frame[frame[y].notna()]
        if frame.empty:
            raise ChartDataError(f"'{y}' has no values in the filtered rows.")
        frame[x] = _labels(frame[x])
        frame[x2] = _labels(frame[x2])
        orders = {}
        for col, keep in ((x, spec.get("top_n", 15)), (x2, 20)):
            w = frame.groupby(col, observed=True)[y].sum() if (y and agg in ("sum", "mean")) else frame.groupby(col, observed=True).size()
            order = _category_order(w, spec.get("sort", "value_desc"), profile.columns[col].is_time_part)
            if len(order) > keep:
                notes.append(f"Showing the top {keep} of {len(order):,} '{col}' values.")
                order = order[:keep]
            orders[col] = order
        frame = frame[frame[x].isin(orders[x]) & frame[x2].isin(orders[x2])]
        out = _aggregate(frame, [x2, x], y, agg)
        out[x] = pd.Categorical(out[x], categories=orders[x], ordered=True)
        out[x2] = pd.Categorical(out[x2], categories=orders[x2], ordered=True)
        return out.sort_values([x2, x]), notes

    if t == "scatter":
        frame = df[cols].dropna(subset=[x, y] + ([spec["size"]] if spec.get("size") else []))
        if frame.empty:
            raise ChartDataError("No rows have values for both measures.")
        if len(frame) > 5000:
            notes.append(f"Showing a random sample of 5,000 of {len(frame):,} points.")
            frame = frame.sample(5000, random_state=0)
        if color:
            frame, _ = _fold_other(frame, color, None, "count", MAX_COLOR_SERIES)
            frame = frame.assign(**{color: _labels(frame[color])})
        return frame, notes

    if t == "histogram":
        frame = df[[x]].dropna()
        if frame.empty:
            raise ChartDataError(f"'{x}' has no values in the filtered rows.")
        return frame, notes

    if t == "box":
        frame = df[[c for c in (x, y) if c]].dropna(subset=[y])
        if x:
            frame, folded = _fold_other(frame, x, None, "count", 12)
            if folded:
                notes.append(f"{folded} smaller groups are combined into 'Other'.")
            frame = frame.assign(**{x: _labels(frame[x])})
        if frame.empty:
            raise ChartDataError(f"'{y}' has no values in the filtered rows.")
        return frame, notes

    if t == "map":
        frame = df[cols].dropna(subset=[x, y])
        if frame.empty:
            raise ChartDataError("No rows have coordinates.")
        if len(frame) > 5000:
            notes.append(f"Showing a random sample of 5,000 of {len(frame):,} points.")
            frame = frame.sample(5000, random_state=0)
        return frame, notes

    raise ChartDataError(f"Unsupported chart type {t!r}.")


def _natural_key(v):
    s = str(v)
    try:
        return (0, float(s), "")
    except ValueError:
        return (1, 0.0, s.lower())


def _style(fig: go.Figure, height: int, show_legend: bool) -> go.Figure:
    fig.update_layout(
        height=height,
        margin=dict(l=8, r=8, t=8, b=8),
        paper_bgcolor=SURFACE, plot_bgcolor=SURFACE,
        font=dict(family="Inter, -apple-system, Segoe UI, Roboto, sans-serif", size=12, color=INK_2),
        hoverlabel=dict(bgcolor="#ffffff", bordercolor="#d9dce1", font=dict(color=INK, size=12)),
        showlegend=show_legend,
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="left", x=0, title_text="", font=dict(color=INK_2)),
        colorway=SERIES_COLORS,
        bargap=0.28,
        barcornerradius=4,
        uniformtext=dict(minsize=9, mode="hide"),
    )
    fig.update_xaxes(showgrid=False, linecolor="#d9dce1", tickfont=dict(color=INK_3), title_font=dict(color=INK_2), zeroline=False, automargin=True)
    fig.update_yaxes(gridcolor=GRID, linecolor="rgba(0,0,0,0)", tickfont=dict(color=INK_3), title_font=dict(color=INK_2), zeroline=False, automargin=True)
    return fig


def _value_format(profile: DatasetProfile, y: str | None, agg: str) -> str:
    if agg in ("count", "nunique") or not y:
        return ",.0f"
    cp = profile.columns.get(y)
    if cp and cp.stats.get("integer_valued") and agg in ("sum", "min", "max"):
        return ",.0f"
    return ",.2f"


def _color_map(data: pd.DataFrame, col: str) -> dict:
    cats = [c for c in pd.unique(data[col]) if c != OTHER]
    cmap = {c: SERIES_COLORS[i % len(SERIES_COLORS)] for i, c in enumerate(cats)}
    cmap[OTHER] = OTHER_COLOR
    return cmap


def _label_last_points(fig: go.Figure, vfmt: str) -> None:
    """Direct-label the final point of each line instead of labelling every point."""
    for tr in fig.data:
        xs, ys = list(tr.x), list(tr.y)
        if not xs:
            continue
        fig.add_annotation(x=xs[-1], y=ys[-1], text=format_value(ys[-1], compact=True), showarrow=False,
                           xanchor="left", xshift=6, font=dict(size=11, color=INK_2))


def build_figure(data: pd.DataFrame, spec: dict, profile: DatasetProfile, height: int = 340,
                 highlight: list[str] | None = None) -> go.Figure:
    t, x, y, color, agg = spec["type"], spec.get("x"), spec.get("y"), spec.get("color"), spec.get("agg", "sum")
    multi = bool(spec.get("extra")) and SERIES in data.columns
    legend_col = SERIES if multi else color
    measure = "Value" if multi else agg_label(agg, y)
    vfmt = _value_format(profile, y, agg)
    labels = bool(spec.get("labels"))
    barmode = spec.get("barmode", "group")
    color_map = _color_map(data, legend_col) if legend_col and legend_col in data.columns else None

    if t in ("line", "area"):
        if t == "area":
            fig = px.area(data, x="period", y="value", color=legend_col, color_discrete_map=color_map,
                          groupnorm="percent" if barmode == "percent" and legend_col else None)
            if legend_col and barmode == "group":
                fig.update_traces(stackgroup=None, fill="tozeroy")
        else:
            fig = px.line(data, x="period", y="value", color=legend_col, color_discrete_map=color_map, markers=len(data) <= 40)
        name = "%{fullData.name}" if legend_col else ""
        fig.update_traces(line=dict(width=2), marker=dict(size=7),
                          hovertemplate=f"%{{x|%b %d, %Y}}<br>{measure}: %{{y:{vfmt}}}<extra>{name}</extra>")
        if not legend_col:
            fig.update_traces(line_color=SERIES_COLORS[0])
        fig.update_layout(hovermode="x unified" if legend_col else "closest")
        fig.update_xaxes(title_text=x)
        pct = t == "area" and barmode == "percent" and legend_col
        fig.update_yaxes(title_text="% of total" if pct else measure,
                         rangemode="tozero" if agg in ("sum", "count") and data["value"].min() >= 0 else "normal")
        if labels and t == "line":
            _label_last_points(fig, vfmt)
        return _style(fig, height, bool(legend_col))

    if t in ("bar", "hbar"):
        horizontal = t == "hbar"
        plot = data
        value_title = measure
        if legend_col and barmode == "percent":
            plot = data.copy()
            totals = plot.groupby(x, observed=True)["value"].transform("sum").replace(0, np.nan)
            plot["value"] = plot["value"] / totals * 100
            value_title = "% of total"
        kwargs = dict(x="value", y=x, orientation="h") if horizontal else dict(x=x, y="value")
        mode = "group" if not legend_col or barmode == "group" else "relative"
        fig = px.bar(plot, color=legend_col, color_discrete_map=color_map, barmode=mode, **kwargs)
        if not legend_col:
            fig.update_traces(marker_color=SERIES_COLORS[0])
        cat_axis = "%{y}" if horizontal else "%{x}"
        val_axis = "%{x" if horizontal else "%{y"
        fmt = ".1f}%" if value_title == "% of total" else f"{vfmt}}}"
        name = "%{fullData.name}" if legend_col else ""
        fig.update_traces(hovertemplate=f"{cat_axis}<br>{value_title}: {val_axis}:{fmt}<extra>{name}</extra>", marker_line_width=0)
        if labels:
            tfmt = ".0f}%" if value_title == "% of total" else ".3~s}"
            stacked = mode == "relative"
            fig.update_traces(texttemplate=f"{val_axis}:{tfmt}", textposition="inside" if stacked else "outside",
                              textfont=dict(size=11, color="#ffffff" if stacked else INK_2), cliponaxis=False)
        n_cats = data[x].nunique()
        gap = 0.7 if n_cats == 1 else 0.55 if n_cats == 2 else 0.4 if n_cats <= 4 else 0.28
        if horizontal:
            fig.update_yaxes(autorange="reversed", title_text="", type="category", showgrid=False)
            fig.update_xaxes(title_text=value_title, gridcolor=GRID, showgrid=True)
            height = max(height, min(900, 28 * n_cats * (1 if mode == "relative" or not legend_col else max(1, data[legend_col].nunique() // 2)) + 80))
        else:
            fig.update_xaxes(title_text="", type="category")
            fig.update_yaxes(title_text=value_title)
        fig = _style(fig, height, bool(legend_col))
        fig.update_layout(bargap=gap)
        if highlight:
            wanted = {str(v) for v in highlight}
            for tr in fig.data:  # selected categories stay solid, the rest fade (Power BI-style highlight)
                cats = tr.y if horizontal else tr.x
                tr.marker.opacity = [1.0 if str(c) in wanted else 0.25 for c in cats]
        return fig

    if t == "heatmap":
        x2 = spec["x2"]
        pivot = data.pivot_table(index=x2, columns=x, values="value", aggfunc="first", observed=False)
        z = pivot.to_numpy(dtype=float)
        fig = go.Figure(go.Heatmap(
            z=z, x=[str(c) for c in pivot.columns], y=[str(r) for r in pivot.index],
            colorscale=SEQUENTIAL, xgap=2, ygap=2, hoverongaps=False,
            colorbar=dict(thickness=10, outlinewidth=0, tickfont=dict(color=INK_3, size=10)),
            hovertemplate=f"{x}: %{{x}}<br>{x2}: %{{y}}<br>{measure}: %{{z:{vfmt}}}<extra></extra>",
            texttemplate="%{z:.3~s}" if labels or z.size <= 120 else None,
            textfont=dict(size=11),
        ))
        fig.update_xaxes(type="category", title_text=x, side="bottom")
        fig.update_yaxes(type="category", title_text=x2, autorange="reversed", gridcolor="rgba(0,0,0,0)")
        height = max(height, min(800, 34 * len(pivot.index) + 110))
        return _style(fig, height, False)

    if t == "donut":
        colors = [SERIES_COLORS[i % len(SERIES_COLORS)] if v != OTHER else OTHER_COLOR for i, v in enumerate(data[x])]
        fig = go.Figure(go.Pie(
            labels=data[x], values=data["value"], hole=0.62, sort=False, direction="clockwise",
            marker=dict(colors=colors, line=dict(color=SURFACE, width=2)),
            textinfo="percent+label" if labels else "percent", textfont=dict(color="#ffffff", size=11),
            insidetextorientation="horizontal",
            hovertemplate=f"%{{label}}<br>{measure}: %{{value:{vfmt}}}<br>%{{percent}} of total<extra></extra>",
        ))
        fig = _style(fig, height, True)
        fig.update_layout(legend=dict(orientation="v", x=1.02, y=0.5, yanchor="middle", xanchor="left"))
        return fig

    if t == "scatter":
        size = spec.get("size")
        fig = px.scatter(data, x=x, y=y, color=color, color_discrete_map=color_map, opacity=0.75,
                         size=size, size_max=28 if size else None)
        fig.update_traces(marker=dict(line=dict(width=1, color=SURFACE)))
        if not size:
            fig.update_traces(marker=dict(size=8))
        if not color:
            fig.update_traces(marker_color=SERIES_COLORS[0])
        fig.update_xaxes(title_text=x, showgrid=True, gridcolor=GRID)
        fig.update_yaxes(title_text=y)
        return _style(fig, height, bool(color))

    if t == "histogram":
        fig = px.histogram(data, x=x, nbins=min(40, max(10, int(np.sqrt(len(data))))))
        fig.update_traces(marker_color=SERIES_COLORS[0], marker_line=dict(color=SURFACE, width=1),
                          hovertemplate=f"{x}: %{{x}}<br>Rows: %{{y:,}}<extra></extra>")
        if labels:
            fig.update_traces(texttemplate="%{y}", textposition="outside", textfont=dict(size=10, color=INK_2))
        fig.update_xaxes(title_text=x)
        fig.update_yaxes(title_text="Rows")
        fig = _style(fig, height, False)
        fig.update_layout(bargap=0.04, barcornerradius=2)
        return fig

    if t == "box":
        fig = px.box(data, x=x, y=y, points="outliers")
        fig.update_traces(marker_color=SERIES_COLORS[0], line=dict(width=1.5), fillcolor="rgba(42,120,214,0.15)")
        fig.update_xaxes(title_text=x or "", type="category" if x else None)
        fig.update_yaxes(title_text=y)
        return _style(fig, height, False)

    if t == "map":
        fig = px.scatter_map(data, lat=y, lon=x, color=color, color_discrete_map=color_map, zoom=1, map_style="carto-positron")
        if not color:
            fig.update_traces(marker=dict(color=SERIES_COLORS[0], size=8))
        return _style(fig, height, bool(color))

    raise ChartDataError(f"Unsupported chart type {t!r}.")


def format_table(data: pd.DataFrame, profile: DatasetProfile) -> pd.DataFrame:
    """Format numeric columns of a summary table for display (strings with separators)."""
    out = data.copy()
    for c in out.columns:
        if pd.api.types.is_numeric_dtype(out[c]):
            out[c] = out[c].map(lambda v: format_value(v, decimals=0 if float(v).is_integer() else 2) if pd.notna(v) else "—")
    return out


__all__ = ["ChartDataError", "apply_chart_filters", "build_figure", "format_table", "measures_of",
           "prepare_chart_data", "AGGREGATIONS"]
