"""The interactive dashboard: header actions, filters, KPIs, charts, insights and Q&A."""
from __future__ import annotations

import re
from datetime import date

import pandas as pd
import plotly.express as px
import streamlit as st

from ..chart_renderer import ChartDataError, _style, build_figure, prepare_chart_data
from .. import config
from ..config import SERIES_COLORS
from ..dashboard_service import (
    MAX_FILTER_OPTIONS, apply_filters, compute_kpi, filter_candidates, filter_options, regenerate,
)
from ..data_loader import DataLoadError
from ..data_profiler import DATE, DIMENSION, DatasetProfile
from ..formatting import format_value
from ..insights import generate_insights
from ..report_export import build_html_report
from ..query_engine import answer_question, suggested_questions
from ..storage import StorageError
from . import state
from ..chart_generator import ChartConfigError, validate_chart_spec, validate_kpi_spec
from .chart_editor import render_chart_menu, render_kpi_manager
from .visual_builder import BUILDER, builder_dialog, fields_dialog, start_builder
from .theme import card, chart_heading, chips, date_column_config, empty_state, esc, insight_list, kpi_grid

PLOTLY_CONFIG = {"displaylogo": False, "modeBarButtonsToRemove": ["lasso2d", "select2d", "autoScale2d", "toggleSpikelines"],
                 "toImageButtonOptions": {"format": "png", "scale": 2}}
FULL_WIDTH = {"table", "map"}


def _fkey(did: str, col: str) -> str:
    return f"flt_{did}_{col}"


# --------------------------------------------------------------------------- header

def _render_header(cfg: dict, meta: dict, profile: DatasetProfile, filtered: pd.DataFrame) -> None:
    n_filtered = len(filtered)
    left, right = st.columns([5, 4], vertical_alignment="bottom")
    with left:
        st.markdown('<div class="page-eyebrow">Dashboard</div>', unsafe_allow_html=True)
        st.markdown(f"# {esc(cfg['name'])}")
        items = [(meta["original_name"], ""), ]
        if cfg.get("sheet") and cfg["sheet"] != "CSV data":
            items.append((f"Sheet: {cfg['sheet']}", ""))
        items.append((f"{profile.n_rows:,} rows · {profile.n_cols} columns", ""))
        if n_filtered != profile.n_rows:
            items.append((f"{n_filtered:,} rows after filters", "accent"))
        items.append(("Unsaved changes", "warn") if state.is_dirty() else ("Saved", ""))
        chips(items)
        analysis = cfg.get("analysis")
        if analysis and st.button("Back to AI analyst", icon=":material/chat:", key="back_to_analyst"):
            if state.is_dirty():
                state.save_current()
            st.session_state["analysis"] = {"dataset_id": analysis["dataset_id"], "sheet": analysis.get("sheet") or cfg.get("sheet"),
                                            "header_row": cfg.get("header_row")}
            st.switch_page(st.session_state["_pages"]["analyst"])
    with right:
        b1, b0, b2, b3 = st.columns([1, 1, 1.1, 1.3])
        if b0.button("Fields", icon=":material/dataset:", width="stretch", help="Correct column types and default calculations"):
            fields_dialog(profile)
        if b1.button("Save", type="primary", icon=":material/save:", width="stretch"):
            try:
                state.save_current()
                st.toast("Dashboard saved for this session. Download it from My Dashboards to keep it." if config.ONLINE else "Dashboard saved locally.", icon=":material/check_circle:")
                st.rerun()
            except StorageError as exc:
                st.error(str(exc))
        with b2.popover("Manage", icon=":material/more_horiz:", width="stretch"):
            _manage_menu(cfg, profile, filtered)
        if b3.button("New dataset", icon=":material/upload_file:", width="stretch"):
            state.close_dashboard()
            st.rerun()


def _manage_menu(cfg: dict, profile: DatasetProfile, filtered: pd.DataFrame) -> None:
    new_name = st.text_input("Dashboard name", value=cfg["name"], max_chars=120, key=f"rename_{cfg['id']}")
    if st.button("Rename", key="mm_rename", width="stretch", icon=":material/edit:"):
        if new_name.strip():
            cfg["name"] = new_name.strip()
            state.save_current()
            st.rerun()
    st.divider()
    if st.button("Download to edit later (.zip)", key="mm_dl_prep", width="stretch", icon=":material/download:",
                 help="Saves your changes, then packages the settings and the data file so you can import them later."):
        state.save_current()
        st.session_state["mm_dl_ready"] = cfg["id"]
    if st.session_state.get("mm_dl_ready") == cfg["id"]:
        try:
            fname, blob = state.get_storage().export_bundle(cfg["id"])
            st.download_button("Save file", blob, file_name=fname, mime="application/zip", key="mm_dl",
                               type="primary", width="stretch")
        except StorageError as exc:
            st.error(str(exc))
    if st.button("Download as web page (.html)", key="mm_html_prep", width="stretch", icon=":material/language:",
                 help="One file that opens in any browser, offline, with no setup. A snapshot using the current filters."):
        st.session_state["mm_html_ready"] = cfg["id"]
    if st.session_state.get("mm_html_ready") == cfg["id"]:
        html_doc = build_html_report(cfg, filtered, profile, profile.n_rows)
        stem = re.sub(r"[^A-Za-z0-9._-]+", "_", cfg["name"]).strip("_")[:60] or "dashboard"
        st.download_button("Save web page", html_doc.encode("utf-8"), file_name=f"{stem}.html", mime="text/html",
                           key="mm_html", type="primary", width="stretch")
    if st.button("Duplicate", key="mm_dup", width="stretch", icon=":material/content_copy:"):
        state.save_current()
        copy = state.get_storage().duplicate_dashboard(cfg["id"])
        state.open_dashboard(copy["id"])
        st.toast("Opened the copy.")
        st.rerun()
    confirm_regen = st.checkbox("Replace my charts and KPIs with fresh recommendations", key="mm_regen_ok")
    if st.button("Reset to recommendations", key="mm_regen", width="stretch", disabled=not confirm_regen, icon=":material/restart_alt:"):
        _, result, _ = state.load_for(cfg["dataset_id"], cfg["sheet"], cfg.get("header_row"))
        new = regenerate(cfg, profile, result.df)
        st.session_state["dashboard"] = new
        st.session_state["filters_initialised_for"] = None
        _reset_filter_widgets(cfg["id"])
        st.rerun()
    st.divider()
    confirm_del = st.checkbox("I want to delete this dashboard", key="mm_del_ok")
    if st.button("Delete dashboard", key="mm_del", width="stretch", disabled=not confirm_del, icon=":material/delete:"):
        state.get_storage().delete_dashboard(cfg["id"])
        state.close_dashboard()
        st.toast("Dashboard deleted. The uploaded file is kept (remove it in Settings).")
        st.rerun()


def _reset_filter_widgets(did: str) -> None:
    for k in list(st.session_state.keys()):
        if isinstance(k, str) and k.startswith(f"flt_{did}_"):
            del st.session_state[k]


# --------------------------------------------------------------------------- filters

def _parse_date(v, fallback: date) -> date:
    try:
        d = pd.to_datetime(v)
        return fallback if pd.isna(d) else d.date()
    except (TypeError, ValueError):
        return fallback


def _render_filters(cfg: dict, df: pd.DataFrame, profile: DatasetProfile) -> pd.DataFrame:
    did = cfg["id"]
    candidates = filter_candidates(profile)
    cols = [c for c in cfg.get("filter_columns", []) if c in candidates]
    first_time = st.session_state.get("filters_initialised_for") != did
    saved = cfg.get("filters") or {"categorical": {}, "date": {}}
    filters: dict = {"categorical": {}, "date": {}}

    with card(f"filters_{did}"):
        widths = [1.25 if profile.columns[c].role == DATE else 1 for c in cols] + [0.6]
        slots = st.columns(widths, vertical_alignment="bottom") if cols else [st.container()]
        for i, c in enumerate(cols):
            key = _fkey(did, c)
            with slots[i]:
                if profile.columns[c].role == DATE:
                    v = df[c].dropna()
                    lo, hi = v.min().date(), v.max().date()
                    if first_time or key not in st.session_state:
                        rng = saved["date"].get(c) or [None, None]
                        start, end = _parse_date(rng[0], lo), _parse_date(rng[1], hi)
                        st.session_state[key] = (max(lo, min(start, hi)), min(hi, max(end, lo)))
                    st.date_input(c, min_value=lo, max_value=hi, key=key, format="YYYY-MM-DD")
                    val = st.session_state.get(key)
                    if isinstance(val, (tuple, list)) and len(val) >= 1:
                        start = val[0]
                        end = val[1] if len(val) > 1 else None
                        if start > lo or (end is not None and end < hi):
                            filters["date"][c] = [start.isoformat(), end.isoformat() if end else None]
                else:
                    opts = filter_options(df, c)
                    if first_time or key not in st.session_state:
                        st.session_state[key] = [v for v in saved["categorical"].get(c, []) if v in opts]
                    else:
                        st.session_state[key] = [v for v in st.session_state[key] if v in opts]
                    st.multiselect(c, opts, key=key, placeholder="All")
                    if st.session_state.get(key):
                        filters["categorical"][c] = list(st.session_state[key])
        with slots[-1] if cols else slots[0]:
            with st.popover("Filters", icon=":material/filter_list:", width="stretch"):
                chosen = st.multiselect("Filter by", candidates, default=cols, key=f"fltcols_{did}",
                                        help="Choose which columns appear in the filter bar.")
                if st.button("Apply", key=f"fltcols_apply_{did}", width="stretch"):
                    cfg["filter_columns"] = chosen[:6]
                    st.rerun()
                if st.button("Clear all filters", key=f"flt_clear_{did}", width="stretch"):
                    _reset_filter_widgets(did)
                    cfg["filters"] = {"categorical": {}, "date": {}}
                    st.session_state["filters_initialised_for"] = None
                    st.rerun()
        if not cols:
            st.caption("No filters selected. Use the Filters button to add some.")

    st.session_state["filters_initialised_for"] = did
    cfg["filters"] = filters
    return apply_filters(df, filters, profile)


def _cross_filter(chart_key: str, spec: dict, did: str) -> None:
    """Clicking bars filters the whole dashboard by the clicked categories."""
    event = st.session_state.get(chart_key)
    try:
        points = event["selection"]["points"]
    except (KeyError, TypeError):
        return
    cfg = st.session_state["dashboard"]
    col = spec["x"]
    key = _fkey(did, col)
    current = sorted(st.session_state.get(key) or [])
    horizontal = spec["type"] == "hbar"
    values = sorted({str(p.get("y") if horizontal else p.get("x")) for p in points
                     if (p.get("y") if horizontal else p.get("x")) is not None})
    if not values or values == current:
        # Clicking empty space, double-clicking, or clicking the selected bar again clears the selection.
        _clear_selection(did, col)
        return
    if col not in cfg["filter_columns"]:
        cfg["filter_columns"] = (cfg["filter_columns"] + [col])[-6:]
    st.session_state[key] = values


def _clear_selection(did: str, col: str) -> None:
    st.session_state[_fkey(did, col)] = []
    # A new chart key also resets Plotly's own selection state on the clicked chart.
    st.session_state["select_epoch"] = st.session_state.get("select_epoch", 0) + 1


def _clear_all_filters(did: str) -> None:
    _reset_filter_widgets(did)
    st.session_state["dashboard"]["filters"] = {"categorical": {}, "date": {}}
    st.session_state["filters_initialised_for"] = None
    st.session_state["select_epoch"] = st.session_state.get("select_epoch", 0) + 1


# --------------------------------------------------------------------------- charts

def _open_builder(chart_id: str | None, data: pd.DataFrame, profile: DatasetProfile, preset: str | None = None) -> None:
    start_builder(chart_id, profile, preset)
    builder_dialog(data, profile)


def _render_chart(spec: dict, data: pd.DataFrame, profile: DatasetProfile, did: str,
                  highlight: list[str] | None = None) -> None:
    with card(f"chart_{spec['id']}"):
        head, tools = st.columns([7, 1], vertical_alignment="top")
        with head:
            chart_heading(spec["title"], spec.get("description", ""))
        with tools:
            with st.popover("", icon=":material/more_vert:", help="Edit, duplicate, resize or remove this visual"):
                render_chart_menu(spec, profile, lambda cid: _open_builder(cid, data, profile))
        err = st.session_state.get("chart_errors", {}).get(spec["id"])
        if err:
            st.warning(err, icon=":material/warning:")
        # Re-validate against the current field types: a visual can stop fitting after a Fields change.
        try:
            spec = dict(validate_chart_spec(spec, profile), description=spec.get("description", ""))
        except ChartConfigError as exc:
            empty_state("This visual no longer fits the data", f"{exc} Edit the visual, reset the fields, or remove it.")
            return
        try:
            agg, notes = prepare_chart_data(data, spec, profile)
        except ChartDataError as exc:
            empty_state("Nothing to show", str(exc))
            return
        if spec["type"] == "table":
            col_cfg = {c: st.column_config.NumberColumn(c, format="localized") for c in agg.columns if pd.api.types.is_numeric_dtype(agg[c])}
            st.dataframe(agg, width="stretch", hide_index=True, column_config=col_cfg, height=min(420, 38 + 35 * len(agg)))
        else:
            fig = build_figure(agg, spec, profile, highlight=highlight)
            key = f"chart_{did}_{spec['id']}_{st.session_state.get('select_epoch', 0)}"
            if _clickable(spec, profile):
                st.plotly_chart(fig, key=key, width="stretch", config=PLOTLY_CONFIG, on_select=lambda: _cross_filter(key, spec, did),
                                selection_mode="points")
                if highlight:
                    c1, c2 = st.columns([3, 2], vertical_alignment="center")
                    shown = ", ".join(highlight[:3]) + ("…" if len(highlight) > 3 else "")
                    c1.markdown(f'<div class="chart-note">Filtering the dashboard by {esc(spec["x"])} = {esc(shown)}. '
                                "Click the bar again to undo.</div>", unsafe_allow_html=True)
                    c2.button(f"Show all {spec['x']}"[:40], key=f"clr_{spec['id']}", icon=":material/close:",
                              on_click=_clear_selection, args=(did, spec["x"]), width="stretch")
                else:
                    notes = notes + ["Click a bar to filter the dashboard."]
            else:
                st.plotly_chart(fig, key=key, width="stretch", config=PLOTLY_CONFIG)
        if notes:
            st.markdown(f'<div class="chart-note">{esc(" ".join(notes))}</div>', unsafe_allow_html=True)


def _clickable(spec: dict, profile: DatasetProfile) -> bool:
    cp = profile.columns.get(spec.get("x") or "")
    return spec["type"] in ("bar", "hbar") and cp is not None and cp.role == DIMENSION and cp.unique <= MAX_FILTER_OPTIONS


def _layout_rows(charts: list[dict]) -> list[list[dict]]:
    rows, buf = [], []
    for c in charts:
        if c["type"] in FULL_WIDTH or c.get("width") == "full":
            if buf:
                rows.append(buf)
                buf = []
            rows.append([c])
        else:
            buf.append(c)
            if len(buf) == 2:
                rows.append(buf)
                buf = []
    if buf:
        rows.append(buf)
    return rows


# --------------------------------------------------------------------------- Q&A

def _qa_figure(chart: dict, table: pd.DataFrame):
    data = chart.get("data") if isinstance(chart.get("data"), pd.DataFrame) else table
    x, y = chart["x"], chart["y"]
    if data is None or x not in data.columns or y not in data.columns or len(data) < 2:
        return None
    data = data.head(25)
    if chart["kind"] == "line":
        fig = px.line(data, x=x, y=y, markers=True)
        fig.update_traces(line=dict(width=2, color=SERIES_COLORS[0]), marker=dict(size=7))
    elif chart["kind"] == "hbar":
        fig = px.bar(data, x=y, y=x, orientation="h")
        fig.update_yaxes(autorange="reversed", type="category", title_text="")
        fig.update_traces(marker_color=SERIES_COLORS[0])
    else:
        fig = px.bar(data, x=x, y=y)
        fig.update_xaxes(type="category", title_text="")
        fig.update_traces(marker_color=SERIES_COLORS[0])
    return _style(fig, 260, False)


def _render_qa(df: pd.DataFrame, profile: DatasetProfile, filtered: bool) -> None:
    st.markdown("### Ask a question")
    st.caption("Answers are calculated with pandas on the " + ("filtered" if filtered else "full") + " data. No AI service receives your data.")
    sugg = suggested_questions(profile)
    picked = st.pills("Suggestions", sugg, key="qa_pill", label_visibility="collapsed")
    with st.form("qa_form", clear_on_submit=True, border=False):
        c1, c2 = st.columns([5, 1], vertical_alignment="bottom")
        q = c1.text_input("Question", placeholder="e.g. Which region has the highest total sales?", label_visibility="collapsed")
        asked = c2.form_submit_button("Ask", type="primary", width="stretch")
    question = q.strip() if asked and q.strip() else (picked if picked and st.session_state.get("qa_last_pill") != picked else None)
    st.session_state["qa_last_pill"] = picked
    if question:
        res = answer_question(question, df, profile)
        st.session_state.setdefault("qa_history", []).insert(0, {"q": question, "res": res, "rows": len(df)})
        st.session_state["qa_history"] = st.session_state["qa_history"][:6]

    for i, item in enumerate(st.session_state.get("qa_history", [])):
        res = item["res"]
        with card(f"qa_{i}"):
            st.markdown(f"**{esc(item['q'])}**")
            if res.ok:
                st.markdown(res.answer)
            else:
                st.info(res.answer, icon=":material/help:")
            if res.chart and res.table is not None:
                fig = _qa_figure(res.chart, res.table)
                if fig is not None:
                    st.plotly_chart(fig, key=f"qa_chart_{i}_{abs(hash(item['q']))}", width="stretch", config=PLOTLY_CONFIG)
            if res.table is not None and len(res.table):
                with st.expander("Calculation details"):
                    st.dataframe(res.table, hide_index=True, width="stretch")
                    if res.method:
                        st.caption(f"Method: {res.method} · {item['rows']:,} rows")


# --------------------------------------------------------------------------- page

def render_dashboard(cfg: dict) -> None:
    try:
        meta, result, profile, _ = state.load_dashboard_data(cfg)
    except (StorageError, DataLoadError) as exc:
        st.error(f"This dashboard's data could not be loaded: {exc}")
        if st.button("Back to overview"):
            state.close_dashboard()
            st.rerun()
        return
    df = result.df  # the original (cleaned) dataset — never modified

    warns = st.session_state.get("open_warnings") or []
    filtered_header_placeholder = st.container()
    if warns:
        with st.expander(f"Some saved settings no longer fit the data ({len(warns)})", icon=":material/warning:"):
            for w in warns:
                st.markdown(f"- {esc(w)}")

    filtered = _render_filters(cfg, df, profile)
    with filtered_header_placeholder:
        _render_header(cfg, meta, profile, filtered)

    # KPIs
    k1, k2 = st.columns([6, 1], vertical_alignment="bottom")
    k1.markdown("### Key metrics")
    with k2.popover("KPIs", icon=":material/tune:", width="stretch"):
        render_kpi_manager(profile)
    cards = []
    for k in cfg["kpis"]:
        try:
            validate_kpi_spec(k, profile)
        except ChartConfigError:
            cards.append({"label": k.get("label", "KPI"), "display": "—", "subtitle": "No longer fits the field types",
                          "help": "Change the field back in Fields, or remove this KPI."})
            continue
        out = compute_kpi(filtered, k, profile)
        if k["kind"] == "count" and len(filtered) != len(df):
            out["subtitle"] = f"of {format_value(len(df))} total"
        cards.append(out)
    if cards:
        kpi_grid(cards)
    else:
        empty_state("No KPIs", "Use the KPIs menu to add one.")

    # Charts
    active = any(cfg["filters"].get("categorical", {}).values()) or any(cfg["filters"].get("date", {}).values())
    c1, c3, c2 = st.columns([4, 1.3, 1.3], vertical_alignment="bottom")
    c1.markdown("### Charts")
    if active:
        c3.button("Clear filters", icon=":material/filter_alt_off:", width="stretch", key="clear_all_filters",
                  on_click=_clear_all_filters, args=(cfg["id"],), help="Show all data again")
    if c2.button("Add visual", icon=":material/add_chart:", width="stretch", type="primary"):
        _open_builder(None, filtered, profile)
    if not cfg["charts"]:
        empty_state("No charts", "No suitable charts could be generated automatically. Use 'Add visual' to build one.")
    for row in _layout_rows(cfg["charts"]):
        cols = st.columns(len(row)) if len(row) > 1 else [st.container()]
        for col, spec in zip(cols, row):
            with col:
                # The chart a selection was made on keeps showing every category (the selection is highlighted);
                # all other visuals and the KPIs use the filtered rows.
                sel = (cfg["filters"].get("categorical") or {}).get(spec.get("x")) if _clickable(spec, profile) else None
                if sel:
                    others = {"categorical": {c: v for c, v in cfg["filters"]["categorical"].items() if c != spec["x"]},
                              "date": cfg["filters"].get("date") or {}}
                    _render_chart(spec, apply_filters(df, others, profile), profile, cfg["id"], highlight=list(sel))
                else:
                    _render_chart(spec, filtered, profile, cfg["id"])

    st.write("")
    left, right = st.columns([1, 1.15], gap="large")
    with left:
        st.markdown("### AI insights")
        st.caption("Rule-based findings computed from the " + ("filtered" if len(filtered) != len(df) else "full") + " data.")
        with card("insights"):
            insight_list(generate_insights(filtered, profile))
    with right:
        _render_qa(filtered, profile, len(filtered) != len(df))

    st.write("")
    with st.expander(f"Data preview · {len(filtered):,} rows", icon=":material/table:"):
        st.dataframe(filtered.head(1000), width="stretch", height=380, column_config=date_column_config(filtered))
        st.caption("Showing up to 1,000 rows. Open the Data Explorer for search, column profiles and export.")
