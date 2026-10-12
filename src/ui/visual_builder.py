"""Power BI-style visual builder (modal dialog with live preview) and the Fields editor."""
from __future__ import annotations

import copy
import uuid

import pandas as pd
import streamlit as st

from ..chart_generator import (
    AGGREGATIONS, BARMODES, CHART_SHORT, CHART_TYPES, MAX_EXTRA_MEASURES, MULTI_MEASURE_TYPES, SORTS, TIME_GRAINS,
    ChartConfigError, adapt_spec, compatible_types, default_title, describe_chart, new_id, validate_chart_spec,
)
from ..chart_renderer import ChartDataError, build_figure, prepare_chart_data
from ..dashboard_service import MAX_FILTER_OPTIONS, filter_options
from ..data_profiler import (
    COORDINATE, DATE, DIMENSION, IDENTIFIER, METRIC, ROLE_LABELS, DatasetProfile, allowed_roles, apply_overrides,
)
from .theme import chart_heading, esc

BUILDER = "visual_builder"
_NONE = "__none__"
_COUNT_TYPES = ("line", "area", "bar", "hbar", "donut", "heatmap", "table")
_PLOTLY_CONFIG = {"displaylogo": False, "displayModeBar": False}


# --------------------------------------------------------------------------- helpers

def _sel(label: str, options: list, current, key: str, fmt=None, help: str | None = None, disabled: bool = False):
    """Selectbox supporting a 'none' choice (Streamlit treats a literal None as 'no selection')."""
    if not options:
        return None
    opts = [_NONE if o is None else o for o in options]
    cur = _NONE if current is None else current
    show = fmt or (lambda v: "—" if v is None else str(v))
    value = st.selectbox(label, opts, index=opts.index(cur) if cur in opts else 0, key=key, help=help,
                         format_func=lambda v: show(None if v == _NONE else v), disabled=disabled)
    return None if value == _NONE else value


def _fields(profile: DatasetProfile) -> dict[str, list[str]]:
    cols = profile.columns
    dims = profile.dimension_columns
    times = profile.date_columns + [c.name for c in cols.values() if c.is_time_part and c.name not in profile.date_columns]
    return {
        "metrics": profile.metric_columns,
        "dims": dims,
        "small_dims": [d for d in dims if cols[d].unique <= 50],
        "times": times,
        "ids": [c.name for c in cols.values() if c.role == IDENTIFIER],
        "coords": [c.name for c in cols.values() if c.role == COORDINATE],
        "filterable": [c.name for c in cols.values()
                       if (c.role in (DIMENSION, IDENTIFIER) or c.is_time_part) and 1 <= c.unique <= MAX_FILTER_OPTIONS],
    }


def _wells(t: str, f: dict) -> dict:
    """Which field wells a visual type has, and the columns each accepts."""
    if t in ("line", "area"):
        return {"x": ("X axis (date or period)", f["times"]), "y": True, "color": f["small_dims"]}
    if t in ("bar", "hbar"):
        return {"x": ("Axis (category)", f["dims"] + [c for c in f["times"] if c not in f["dims"]] + f["ids"]),
                "y": True, "color": f["small_dims"]}
    if t == "donut":
        return {"x": ("Category", f["dims"]), "y": True}
    if t == "heatmap":
        pool = f["dims"] + [c for c in f["times"] if c not in f["dims"]]
        return {"x": ("Columns (category)", pool), "x2": ("Rows (category)", pool), "y": True}
    if t == "table":
        return {"x": ("Group rows by", f["dims"] + f["ids"]), "y": True}
    if t == "scatter":
        return {"x": ("X axis (measure)", f["metrics"]), "y_metric": ("Y axis (measure)", f["metrics"]),
                "color": f["small_dims"], "size": f["metrics"]}
    if t == "histogram":
        return {"x": ("Measure", f["metrics"])}
    if t == "box":
        return {"y_metric": ("Measure", f["metrics"]), "x_opt": ("Split by (optional)", f["small_dims"])}
    if t == "map":
        return {"x": ("Longitude", f["coords"]), "y_metric": ("Latitude", f["coords"]), "color": f["small_dims"]}
    return {}


def _is_custom_title(spec: dict) -> bool:
    title = spec.get("title") or ""
    auto = default_title(spec["type"], spec.get("x"), spec.get("y"), spec.get("agg", "sum"), spec.get("color"),
                         spec.get("extra"), spec.get("x2"))
    return bool(title) and (spec.get("title_custom") or title != auto)


# --------------------------------------------------------------------------- open / close

def start_builder(chart_id: str | None, profile: DatasetProfile, preset_type: str | None = None) -> None:
    cfg = st.session_state["dashboard"]
    if chart_id:
        spec = copy.deepcopy(next(c for c in cfg["charts"] if c["id"] == chart_id))
    else:
        types = compatible_types(profile)
        t = preset_type if preset_type in types else ("bar" if "bar" in types else types[0])
        spec = adapt_spec({"id": new_id(), "type": t, "x": None, "y": None, "color": None, "agg": "sum",
                           "sort": "value_desc", "top_n": 15, "time_grain": "auto"}, t, profile)
    spec["title_custom"] = _is_custom_title(spec) if chart_id else False
    st.session_state[BUILDER] = {"chart_id": chart_id, "spec": spec, "nonce": uuid.uuid4().hex[:6]}


def _bump(b: dict) -> None:
    b["nonce"] = uuid.uuid4().hex[:6]


def _on_type(profile: DatasetProfile) -> None:
    b = st.session_state[BUILDER]
    new_type = st.session_state.get(f"vb_{b['nonce']}_type")
    if new_type and new_type != b["spec"]["type"]:
        custom, title = b["spec"].get("title_custom"), b["spec"].get("title")
        b["spec"] = adapt_spec(b["spec"], new_type, profile)
        if custom:
            b["spec"]["title"], b["spec"]["title_custom"] = title, True
    _bump(b)  # recreate the wells with the adapted fields (also restores a de-selected pill)


# --------------------------------------------------------------------------- dialog

@st.dialog("Visual builder", width="large")
def builder_dialog(data: pd.DataFrame, profile: DatasetProfile) -> None:
    b = st.session_state.get(BUILDER)
    if not b:
        st.write("Nothing to edit.")
        return
    spec, n = b["spec"], b["nonce"]

    def k(name: str) -> str:
        return f"vb_{n}_{name}"

    f = _fields(profile)
    types = compatible_types(profile)
    if spec["type"] not in types:
        types.insert(0, spec["type"])
    st.pills("Visual type", types, default=spec["type"], format_func=lambda t: CHART_SHORT[t], key=k("type"),
             on_change=_on_type, args=(profile,), selection_mode="single")
    st.caption(CHART_TYPES[spec["type"]])

    left, right = st.columns([1, 1.45], gap="large")
    with left:
        new = _render_wells(spec, f, profile, data, k)
    with right:
        clean, err = _preview(new, data, profile)

    st.divider()
    a, c, r, _ = st.columns([1.3, 1, 1.6, 1.5])
    editing = b["chart_id"] is not None
    if a.button("Update visual" if editing else "Add to dashboard", type="primary", disabled=clean is None,
                width="stretch", icon=":material/check:"):
        _commit(b, clean)
        st.rerun()
    if c.button("Cancel", width="stretch"):
        st.session_state.pop(BUILDER, None)
        st.rerun()
    if editing and spec.get("origin") and r.button("Revert to recommendation", width="stretch", icon=":material/undo:"):
        b["spec"] = dict(copy.deepcopy(spec["origin"]), id=spec["id"], origin=spec["origin"], title_custom=False)
        _bump(b)
        st.rerun(scope="fragment")
    b["spec"] = new


def _render_wells(spec: dict, f: dict, profile: DatasetProfile, data: pd.DataFrame, k) -> dict:
    t = spec["type"]
    w = _wells(t, f)
    new = dict(spec)
    st.markdown("**Fields**")

    if "x" in w:
        label, options = w["x"]
        new["x"] = _sel(label, options, spec.get("x"), k("x"))
    if "x_opt" in w:
        label, options = w["x_opt"]
        new["x"] = _sel(label, [None] + options, spec.get("x"), k("x"))
    if "x2" in w:
        label, options = w["x2"]
        new["x2"] = _sel(label, [o for o in options if o != new.get("x")], spec.get("x2"), k("x2"))
    if "y_metric" in w:
        label, options = w["y_metric"]
        new["y"] = _sel(label, options, spec.get("y"), k("y"))

    if w.get("y"):
        allow_count = t in _COUNT_TYPES
        y_opts = ([None] if allow_count else []) + f["metrics"]
        c1, c2 = st.columns([1.5, 1])
        with c1:
            new["y"] = _sel("Values", y_opts, spec.get("y"), k("y"), fmt=lambda v: "Count of records" if v is None else v)
        with c2:
            if new["y"] is None:
                st.selectbox("Calculation", ["Count"], key=k("agg_count"), disabled=True)
                new["agg"] = "count"
            else:
                aggs = ["sum", "count"] if t == "donut" else ["sum", "mean", "median", "min", "max", "count", "nunique"]
                cur = spec.get("agg") if spec.get("y") == new["y"] else profile.columns[new["y"]].aggregation
                new["agg"] = _sel("Calculation", aggs, cur if cur in aggs else aggs[0], k("agg"), fmt=lambda a: AGGREGATIONS[a])

        if t in MULTI_MEASURE_TYPES and new["y"] is not None and new["agg"] not in ("count", "nunique"):
            more_opts = [m for m in f["metrics"] if m != new["y"]]
            current_extra = [e["y"] for e in (spec.get("extra") or []) if e["y"] in more_opts]
            more = st.multiselect("More values (same axis)", more_opts, default=current_extra, key=k("extra"),
                                  max_selections=MAX_EXTRA_MEASURES, placeholder="Add a value to compare",
                                  help="Plot several measures together. They share one axis; there is never a second y-axis.")
            prev = {e["y"]: e["agg"] for e in (spec.get("extra") or [])}
            extra = []
            for i, m in enumerate(more):
                default = prev.get(m, profile.columns[m].aggregation)
                agg = st.selectbox(f"Calculation for {m}", ["sum", "mean", "median", "min", "max"],
                                   index=["sum", "mean", "median", "min", "max"].index(default) if default in ("sum", "mean", "median", "min", "max") else 0,
                                   format_func=lambda a: AGGREGATIONS[a], key=k(f"extra_agg_{m}"))
                extra.append({"y": m, "agg": agg})
            new["extra"] = extra
        else:
            new["extra"] = []

    if "color" in w:
        disabled = bool(new.get("extra"))
        new["color"] = _sel("Legend (split by)", [None] + [c for c in w["color"] if c not in (new.get("x"), new.get("y"))],
                            None if disabled else spec.get("color"), k("color"), disabled=disabled,
                            help="Not available when several values are shown." if disabled else None)
        if disabled:
            new["color"] = None
    else:
        new["color"] = None
    if "size" in w:
        new["size"] = _sel("Bubble size (optional)", [None] + [m for m in w["size"] if m not in (new.get("x"), new.get("y"))],
                           spec.get("size"), k("size"))

    with st.expander("Filters on this visual", icon=":material/filter_alt:", expanded=bool(spec.get("chart_filters"))):
        st.caption("Apply only to this visual, on top of the dashboard filters.")
        current = spec.get("chart_filters") or {}
        fcols = st.multiselect("Filter by", f["filterable"], default=[c for c in current if c in f["filterable"]],
                               key=k("fcols"), max_selections=4)
        chart_filters = {}
        for c in fcols:
            opts = filter_options(data, c) if c in data.columns else []
            vals = st.multiselect(c, opts, default=[v for v in current.get(c, []) if v in opts], key=k(f"fv_{c}"),
                                  placeholder="Choose values")
            if vals:
                chart_filters[c] = vals
        new["chart_filters"] = chart_filters

    with st.expander("Format", icon=":material/palette:"):
        auto_title = default_title(t, new.get("x"), new.get("y"), new.get("agg", "sum"), new.get("color"), new.get("extra"), new.get("x2"))
        title = st.text_input("Title", value=spec.get("title", "") if spec.get("title_custom") else "",
                              placeholder=auto_title, key=k("title"), max_chars=120, help="Leave empty to use an automatic title.")
        new["title"], new["title_custom"] = (title.strip(), True) if title.strip() else ("", False)
        if t in ("bar", "hbar", "area") and (new.get("color") or new.get("extra")):
            new["barmode"] = st.segmented_control("Series layout", list(BARMODES), default=spec.get("barmode", "group"),
                                                  format_func=lambda m: BARMODES[m], key=k("barmode")) or "group"
        if t in ("bar", "hbar", "table", "heatmap"):
            c1, c2 = st.columns(2)
            with c1:
                new["sort"] = _sel("Sort", list(SORTS), spec.get("sort", "value_desc"), k("sort"), fmt=lambda s: SORTS[s])
            with c2:
                new["top_n"] = st.number_input("Show top", 1, 50, int(spec.get("top_n", 15)), key=k("top"))
        if t in ("line", "area") and new.get("x") in profile.date_columns:
            new["time_grain"] = _sel("Time grouping", list(TIME_GRAINS), spec.get("time_grain", "auto"), k("grain"),
                                     fmt=lambda g: TIME_GRAINS[g])
        c1, c2 = st.columns(2)
        with c1:
            if t in ("bar", "hbar", "line", "donut", "heatmap", "histogram"):
                new["labels"] = st.toggle("Data labels", value=bool(spec.get("labels")), key=k("labels"))
        with c2:
            new["width"] = st.segmented_control("Width", ["half", "full"], default=spec.get("width", "half"),
                                                format_func=lambda v: "Half" if v == "half" else "Full", key=k("width")) or "half"
    return new


def _preview(new: dict, data: pd.DataFrame, profile: DatasetProfile) -> tuple[dict | None, str | None]:
    st.markdown("**Preview**")
    try:
        clean = validate_chart_spec(new, profile)
    except ChartConfigError as exc:
        st.info(str(exc), icon=":material/info:")
        return None, str(exc)
    clean["description"] = describe_chart(clean, profile)
    clean["title_custom"] = bool(new.get("title_custom"))
    with st.container(border=True):
        chart_heading(clean["title"], clean["description"])
        try:
            agg, notes = prepare_chart_data(data, clean, profile)
        except ChartDataError as exc:
            st.info(str(exc), icon=":material/info:")
            return clean, None
        if clean["type"] == "table":
            st.dataframe(agg, hide_index=True, width="stretch", height=300)
        else:
            st.plotly_chart(build_figure(agg, clean, profile, height=320), width="stretch", config=_PLOTLY_CONFIG,
                            key=f"vb_preview_{st.session_state[BUILDER]['nonce']}")
        if notes:
            st.caption(" ".join(notes))
    st.caption(f"Preview uses the {len(data):,} rows that match the dashboard filters.")
    return clean, None


def _commit(b: dict, clean: dict) -> None:
    cfg = st.session_state["dashboard"]
    clean = dict(clean, auto=False)
    if b["chart_id"]:
        i = next((i for i, c in enumerate(cfg["charts"]) if c["id"] == b["chart_id"]), None)
        if i is not None:
            old = cfg["charts"][i]
            origin = old.get("origin") or ({k: v for k, v in old.items() if k != "origin"} if old.get("auto") else None)
            cfg["charts"][i] = dict(clean, origin=origin)
    else:
        cfg["charts"].append(dict(clean, id=new_id(), origin=None))
    st.session_state.pop(BUILDER, None)
    st.session_state.setdefault("chart_errors", {}).pop(clean["id"], None)


# --------------------------------------------------------------------------- fields (column roles)

_ROLE_ORDER = ["metric", "dimension", "identifier", "date", "text", "coordinate", "hidden", "empty"]
_AGG_LABELS = {"sum": "Sum", "mean": "Average"}


@st.dialog("Fields and data types", width="large")
def fields_dialog(profile: DatasetProfile) -> None:
    cfg = st.session_state["dashboard"]
    st.caption("Correct how each column is treated. Changes apply to every visual, KPI, filter and answer on this "
               "dashboard. Detected types stay available, so you can always reset.")
    rows = []
    for cp in profile.columns.values():
        detected = cp.detected_role or cp.role
        rows.append({
            "Column": cp.name,
            "Detected as": ROLE_LABELS[detected],
            "Use as": ROLE_LABELS[cp.role],
            "Calculation": _AGG_LABELS.get(cp.aggregation, "—") if cp.role == "metric" else "—",
            "Allowed": ", ".join(ROLE_LABELS[r] for r in allowed_roles(cp)),
        })
    df = pd.DataFrame(rows)
    edited = st.data_editor(
        df, hide_index=True, width="stretch", key="fields_editor", height=min(460, 38 + 35 * len(df)),
        column_config={
            "Column": st.column_config.TextColumn(disabled=True),
            "Detected as": st.column_config.TextColumn(disabled=True),
            "Use as": st.column_config.SelectboxColumn(options=[ROLE_LABELS[r] for r in _ROLE_ORDER], required=True),
            "Calculation": st.column_config.SelectboxColumn(options=["Sum", "Average", "—"], required=True,
                                                            help="Default calculation for measures."),
            "Allowed": st.column_config.TextColumn("Allowed types", disabled=True),
        },
    )
    by_label = {v: k for k, v in ROLE_LABELS.items()}
    overrides: dict = {}
    problems = []
    for _, row in edited.iterrows():
        cp = profile.columns[row["Column"]]
        role = by_label.get(row["Use as"], cp.role)
        if role not in allowed_roles(cp):
            problems.append(f"'{cp.name}' can't be a {row['Use as']} (allowed: {row['Allowed']}).")
            continue
        ov = {}
        if role != (cp.detected_role or cp.role):
            ov["role"] = role
        agg = {"Sum": "sum", "Average": "mean"}.get(row["Calculation"])
        if role == "metric" and agg and agg != _detected_agg(cp):
            ov["aggregation"] = agg
        if ov:
            overrides[cp.name] = ov
    for p in problems:
        st.warning(p, icon=":material/warning:")

    a, b, _ = st.columns([1.2, 1.2, 2])
    if a.button("Apply", type="primary", width="stretch", disabled=bool(problems), icon=":material/check:"):
        _apply_overrides(cfg, overrides)
        st.rerun()
    if b.button("Reset to detected", width="stretch", icon=":material/restart_alt:"):
        _apply_overrides(cfg, {})
        st.rerun()


def _detected_agg(cp) -> str:
    """Aggregation the profiler chose before any user override."""
    return cp.detected_aggregation if cp.detected_aggregation != "none" else "sum"


def _apply_overrides(cfg: dict, overrides: dict) -> None:
    from . import state  # local import to avoid a cycle
    # Visuals that no longer fit are kept (and flagged on the dashboard) so resetting the fields restores them.
    cfg["column_overrides"] = overrides
    _, _, base = state.load_for(cfg["dataset_id"], cfg.get("sheet"), cfg.get("header_row"))
    _, warnings = apply_overrides(base, overrides)
    st.session_state["open_warnings"] = warnings
    st.session_state["filters_initialised_for"] = None


__all__ = ["BUILDER", "builder_dialog", "fields_dialog", "start_builder", "esc"]
