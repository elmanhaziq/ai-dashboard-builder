"""Per-visual menu actions and the KPI manager."""
from __future__ import annotations

import copy

import streamlit as st

from ..chart_generator import AGGREGATIONS, ChartConfigError, agg_label, describe_chart, new_id, validate_chart_spec, validate_kpi_spec
from ..data_profiler import METRIC, DatasetProfile


def _charts() -> list[dict]:
    return st.session_state["dashboard"]["charts"]


def _index(chart_id: str) -> int:
    return next((i for i, c in enumerate(_charts()) if c["id"] == chart_id), -1)


def _move(chart_id: str, step: int) -> None:
    charts = _charts()
    i = _index(chart_id)
    j = i + step
    if 0 <= i < len(charts) and 0 <= j < len(charts):
        charts[i], charts[j] = charts[j], charts[i]


def _remove(chart_id: str) -> None:
    i = _index(chart_id)
    if i >= 0:
        del _charts()[i]


def _duplicate(chart_id: str) -> None:
    i = _index(chart_id)
    if i >= 0:
        copy_spec = copy.deepcopy(_charts()[i])
        copy_spec.update(id=new_id(), auto=False, origin=None)
        if copy_spec.get("title_custom"):
            copy_spec["title"] = f"{copy_spec['title']} (copy)"[:120]
        _charts().insert(i + 1, copy_spec)


def _toggle_width(chart_id: str) -> None:
    i = _index(chart_id)
    if i >= 0:
        c = _charts()[i]
        c["width"] = "half" if c.get("width") == "full" else "full"


def _revert(chart_id: str, profile: DatasetProfile) -> None:
    i = _index(chart_id)
    if i < 0 or not _charts()[i].get("origin"):
        return
    origin = copy.deepcopy(_charts()[i]["origin"])
    try:
        clean = validate_chart_spec(dict(origin, id=chart_id, auto=True), profile)
    except ChartConfigError as exc:
        st.session_state.setdefault("chart_errors", {})[chart_id] = f"Can't revert: {exc}"
        return
    clean["description"] = describe_chart(clean, profile)
    clean["origin"] = None
    _charts()[i] = clean


def render_chart_menu(spec: dict, profile: DatasetProfile, on_edit) -> None:
    """Contents of a visual's ⋮ menu. ``on_edit`` opens the visual builder for this chart."""
    cid = spec["id"]
    if st.button("Edit visual", key=f"cm_edit_{cid}", icon=":material/edit:", width="stretch", type="primary"):
        on_edit(cid)
    st.button("Duplicate", key=f"cm_dup_{cid}", on_click=_duplicate, args=(cid,), icon=":material/content_copy:", width="stretch")
    full = spec.get("width") == "full"
    st.button("Half width" if full else "Full width", key=f"cm_w_{cid}", on_click=_toggle_width, args=(cid,),
              icon=":material/width_normal:" if full else ":material/width_full:", width="stretch")
    if spec.get("origin"):
        st.button("Revert to recommendation", key=f"cm_rev_{cid}", on_click=_revert, args=(cid, profile),
                  icon=":material/undo:", width="stretch")
    c1, c2 = st.columns(2)
    c1.button("Earlier", key=f"cm_up_{cid}", on_click=_move, args=(cid, -1), icon=":material/arrow_upward:", width="stretch")
    c2.button("Later", key=f"cm_dn_{cid}", on_click=_move, args=(cid, 1), icon=":material/arrow_downward:", width="stretch")
    st.button("Remove", key=f"cm_rm_{cid}", on_click=_remove, args=(cid,), icon=":material/delete:", width="stretch")


def render_kpi_manager(profile: DatasetProfile) -> None:
    cfg = st.session_state["dashboard"]
    st.markdown("**Shown KPIs**")
    st.caption("Untick to remove; edit a label to rename the card.")
    keep, changed = [], False
    for k in cfg["kpis"]:
        c1, c2 = st.columns([0.15, 1], vertical_alignment="center")
        on = c1.checkbox("Show", value=True, key=f"kpi_keep_{k['id']}", label_visibility="collapsed")
        label = c2.text_input("Label", value=k["label"], key=f"kpi_label_{k['id']}", label_visibility="collapsed", max_chars=80)
        if on:
            if label.strip() and label.strip() != k["label"]:
                k = dict(k, label=label.strip())
                changed = True
            keep.append(k)
    if (len(keep) != len(cfg["kpis"]) or changed) and st.button("Apply changes", key="kpi_apply", type="primary", width="stretch"):
        cfg["kpis"] = keep
        st.rerun()
    st.divider()
    st.markdown("**Add a KPI**")
    candidates = profile.metric_columns + profile.dimension_columns + profile.identifier_columns
    if not candidates:
        st.caption("No columns available.")
        return
    col = st.selectbox("Column", candidates, key="kpi_add_col")
    is_metric = profile.columns[col].role == METRIC
    aggs = ["sum", "mean", "median", "min", "max", "nunique"] if is_metric else ["nunique"]
    default = profile.columns[col].aggregation if is_metric else "nunique"
    agg = st.selectbox("Calculation", aggs, index=aggs.index(default) if default in aggs else 0,
                       format_func=lambda a: AGGREGATIONS[a], key="kpi_add_agg")
    if st.button("Add KPI", type="primary", key="kpi_add_btn", width="stretch"):
        if len(cfg["kpis"]) >= 8:
            st.warning("A dashboard can show up to 8 KPIs.")
            return
        date_col = next(iter(profile.date_columns), None) if agg == "sum" else None
        try:
            kpi = validate_kpi_spec({"kind": "agg", "column": col, "agg": agg,
                                     "label": agg_label(agg, col) if agg != "nunique" else f"Distinct {col}",
                                     "compare_date": date_col,
                                     "description": f"{AGGREGATIONS[agg]} of '{col}' across filtered rows."}, profile)
        except ChartConfigError as exc:
            st.error(str(exc))
            return
        cfg["kpis"].append(kpi)
        st.rerun()


__all__ = ["render_chart_menu", "render_kpi_manager"]
