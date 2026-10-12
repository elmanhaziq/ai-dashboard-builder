"""Page entry points used by ``st.navigation``."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from .. import config
from ..dashboard_service import apply_filters, describe_filters
from ..data_loader import DataLoadError
from ..data_profiler import DATE, METRIC
from ..formatting import format_value
from ..storage import StorageError
from . import state
from .dashboard_view import render_dashboard
from .theme import card, chips, date_column_config, empty_state, esc, human_size, page_header
from .upload_view import render_landing, render_staging


def overview_page() -> None:
    if st.session_state.get("staging"):
        render_staging(st.session_state["staging"])
    elif state.current_dashboard():
        render_dashboard(state.current_dashboard())
    else:
        render_landing()


def dashboards_page() -> None:
    page_header("Library", "My dashboards", "Dashboards are saved as JSON configuration files on this computer.")
    st.write("")
    storage = state.get_storage()
    items = storage.list_dashboards()
    if not items:
        empty_state("No saved dashboards", "Go to Overview and upload a spreadsheet to create one.")
        return
    query = st.text_input("Search", placeholder="Filter by name or file", label_visibility="collapsed")
    if query:
        ql = query.lower()
        items = [d for d in items if ql in str(d.get("name", "")).lower() or ql in str(d.get("dataset_name", "")).lower()]
    current = state.current_dashboard()
    for d in items:
        with card(f"dash_{d['id']}"):
            if d.get("error"):
                c1, c2 = st.columns([5, 1], vertical_alignment="center")
                c1.markdown(f"**{esc(d['id'])}.json** — {esc(d['error'])}")
                if c2.button("Delete file", key=f"bad_del_{d['id']}"):
                    storage.delete_dashboard(d["id"])
                    st.rerun()
                continue
            c1, c2 = st.columns([5, 4], vertical_alignment="center")
            with c1:
                tag = " · open now" if current and current.get("id") == d["id"] else ""
                st.markdown(f"**{esc(d['name'])}**{tag}")
                sheet = f" · sheet {d['sheet']}" if d.get("sheet") and d["sheet"] != "CSV data" else ""
                st.caption(f"{d.get('dataset_name') or 'Dataset'}{sheet} · {d['n_kpis']} KPIs · {d['n_charts']} charts · "
                           f"updated {str(d.get('updated_at') or '')[:16].replace('T', ' ')} UTC")
            with c2:
                b1, b2, b3, b4 = st.columns(4)
                if b1.button("Open", key=f"open_{d['id']}", type="primary", width="stretch"):
                    try:
                        state.open_dashboard(d["id"])
                    except (StorageError, DataLoadError) as exc:
                        st.error(str(exc))
                    else:
                        st.switch_page(st.session_state["_pages"]["overview"])
                with b2.popover("Rename", width="stretch"):
                    name = st.text_input("New name", value=d["name"], key=f"rn_{d['id']}", max_chars=120)
                    if st.button("Save name", key=f"rn_btn_{d['id']}"):
                        try:
                            storage.rename_dashboard(d["id"], name)
                            if current and current.get("id") == d["id"]:
                                current["name"] = name.strip()
                                state.save_current()
                            st.rerun()
                        except StorageError as exc:
                            st.error(str(exc))
                if b3.button("Copy", key=f"dup_{d['id']}", width="stretch"):
                    storage.duplicate_dashboard(d["id"])
                    st.rerun()
                with b4.popover("Delete", width="stretch"):
                    st.write("Delete this dashboard? The uploaded file is kept.")
                    if st.button("Delete", key=f"del_{d['id']}", type="primary"):
                        storage.delete_dashboard(d["id"])
                        if current and current.get("id") == d["id"]:
                            state.close_dashboard()
                        st.rerun()


def explorer_page() -> None:
    cfg = state.current_dashboard()
    staging = st.session_state.get("staging")
    source = cfg or staging
    if not source:
        page_header("Data", "Data explorer")
        st.write("")
        empty_state("No dataset open", "Open a dashboard or upload a file on the Overview page.")
        return
    try:
        if cfg:
            meta, result, profile, _ = state.load_dashboard_data(cfg)
        else:
            meta, result, profile = state.load_for(source["dataset_id"], source.get("sheet"), source.get("header_row"))
    except (StorageError, DataLoadError) as exc:
        st.error(str(exc))
        return
    df = result.df
    page_header("Data", "Data explorer", f"{meta['original_name']} · {result.sheet}")
    use_filters = False
    if cfg and (cfg["filters"]["categorical"] or cfg["filters"]["date"]):
        use_filters = st.toggle(f"Apply dashboard filters ({describe_filters(cfg['filters'])})", value=False)
    data = apply_filters(df, cfg["filters"], profile) if use_filters else df
    chips([(f"{len(data):,} rows", "accent"), (f"{profile.n_cols} columns", ""),
           (f"{profile.duplicate_rows:,} duplicate rows", "warn" if profile.duplicate_rows else ""),
           (f"{profile.missing_pct:.1f}% cells missing", "warn" if profile.missing_pct > 5 else "")])
    st.write("")
    tab_data, tab_profile, tab_col, tab_notes = st.tabs(["Rows", "Column profile", "Column detail", "Cleaning log"])

    with tab_data:
        c1, c2, c3 = st.columns([2, 2, 1])
        search = c1.text_input("Search all text columns", placeholder="Contains…")
        cols = c2.multiselect("Columns", list(df.columns), placeholder="All columns")
        limit = c3.selectbox("Rows", [100, 500, 1000, 5000], index=1)
        view = data
        if search:
            text_cols = [c for c in data.columns if not pd.api.types.is_numeric_dtype(data[c]) and not pd.api.types.is_datetime64_any_dtype(data[c])]
            if text_cols:
                mask = pd.Series(False, index=data.index)
                for c in text_cols:
                    mask |= data[c].astype(str).str.contains(search, case=False, regex=False, na=False)
                view = data[mask]
        if cols:
            view = view[cols]
        st.dataframe(view.head(limit), width="stretch", height=460, column_config=date_column_config(view))
        st.caption(f"Showing {min(limit, len(view)):,} of {len(view):,} matching rows. Click a column header to sort.")
        st.download_button("Download these rows as CSV", view.to_csv(index=False).encode("utf-8"),
                           file_name="filtered_data.csv", mime="text/csv", icon=":material/download:")

    with tab_profile:
        st.dataframe(profile.summary_frame(), width="stretch", hide_index=True, height=460)
        st.caption("Role is inferred from the data and column names. Identifiers are never summed; rates and prices are averaged.")

    with tab_col:
        col = st.selectbox("Column", list(profile.columns))
        cp = profile.columns[col]
        m = st.columns(4)
        m[0].metric("Role", cp.role.title())
        m[1].metric("Missing", f"{cp.missing:,} ({cp.missing_pct:.1f}%)")
        m[2].metric("Unique values", f"{cp.unique:,}")
        m[3].metric("Default aggregation", cp.aggregation if cp.role == METRIC else "—")
        if cp.notes:
            st.caption(" · ".join(cp.notes))
        if cp.role == METRIC and cp.stats:
            s = cp.stats
            st.dataframe(pd.DataFrame({
                "Statistic": ["Min", "Q1", "Median", "Mean", "Q3", "Max", "Std dev", "Sum", "Zeros", "Negatives", "Outliers (1.5×IQR)"],
                "Value": [format_value(s[k]) for k in ("min", "q1", "median", "mean", "q3", "max", "std", "sum", "zeros", "negatives", "outliers")],
            }), hide_index=True, width="content")
        elif cp.role == DATE and cp.stats:
            st.write(f"From **{cp.stats['min']:%Y-%m-%d}** to **{cp.stats['max']:%Y-%m-%d}** ({cp.stats['span_days']:,} days).")
        elif cp.top_values:
            st.dataframe(pd.DataFrame(cp.top_values, columns=[col, "Rows"]), hide_index=True, width="content")

    with tab_notes:
        notes = result.warnings + profile.warnings
        if notes:
            for n in notes:
                st.markdown(f"- {esc(n)}")
        else:
            st.write("No cleaning was needed.")
        if result.conversions:
            st.markdown("**Type conversions**")
            st.dataframe(pd.DataFrame(list(result.conversions.items()), columns=["Column", "Converted to"]), hide_index=True, width="content")
        if not result.removed_rows.empty:
            st.markdown("**Rows excluded as totals/subtotals**")
            st.dataframe(result.removed_rows, width="stretch")
        st.markdown("**Original sheet (first 200 rows, as stored)**")
        try:
            st.dataframe(state.raw_preview(meta["id"], result.sheet if result.sheet != "CSV data" else None, meta["sha256"]), width="stretch", height=300)
        except DataLoadError as exc:
            st.warning(str(exc))


def settings_page() -> None:
    page_header("Workspace", "Settings", "Local storage, limits and privacy.")
    st.write("")
    storage = state.get_storage()
    with card("settings_storage"):
        st.markdown("### Storage")
        st.markdown(
            f"- Uploaded files: `{esc(storage.upload_dir)}`\n"
            f"- Dashboard configurations: `{esc(storage.dashboard_dir)}`\n"
            f"- Space used by uploads: **{human_size(storage.disk_usage())}**\n"
            f"- Upload limit: **{config.MAX_UPLOAD_MB} MB** · row limit: **{config.MAX_ROWS:,}** (change in `.env`)"
        )
        st.caption("Files are stored unencrypted on this computer and are not uploaded anywhere. Deleting removes the files "
                   "with normal file deletion; it is not a secure wipe.")

    with card("settings_datasets"):
        st.markdown("### Uploaded datasets")
        datasets = storage.list_datasets()
        dashboards = storage.list_dashboards()
        if not datasets:
            st.caption("No datasets stored.")
        for meta in datasets:
            used = [d for d in dashboards if d.get("dataset_id") == meta["id"]]
            c1, c2 = st.columns([4, 1], vertical_alignment="center")
            c1.markdown(f"**{esc(meta['original_name'])}** · {human_size(meta['size'])} · uploaded {meta['uploaded_at'][:10]} · "
                        f"used by {len(used)} dashboard(s)")
            with c2.popover("Delete", width="stretch"):
                st.write(f"Delete this file and the {len(used)} dashboard(s) built on it?")
                if st.button("Delete permanently", key=f"ds_del_{meta['id']}", type="primary"):
                    cur = state.current_dashboard()
                    storage.delete_dataset(meta["id"])
                    if (cur and cur.get("dataset_id") == meta["id"]) or (st.session_state.get("staging") or {}).get("dataset_id") == meta["id"]:
                        state.close_dashboard()
                    st.cache_data.clear()
                    st.rerun()

    with card("settings_clear"):
        st.markdown("### Clear all local data")
        st.caption("Removes every uploaded file and every saved dashboard from this computer.")
        ok = st.checkbox("I understand this cannot be undone")
        if st.button("Clear everything", disabled=not ok, icon=":material/delete_forever:"):
            n_ds, n_db = storage.clear_all()
            state.close_dashboard()
            st.cache_data.clear()
            st.success(f"Removed {n_ds} dataset(s) and {n_db} dashboard(s).")

    with card("settings_ai"):
        st.markdown("### AI features")
        st.markdown(
            "Insights and answers come from a **local, rule-based engine**: every number is calculated with pandas "
            "on your data and no data leaves this computer. No API key is needed.\n\n"
            "The query engine turns a question into a structured plan before calculating anything, so an optional "
            "LLM could later interpret questions while pandas still does all the maths. No external AI provider "
            "is configured in this version."
        )
