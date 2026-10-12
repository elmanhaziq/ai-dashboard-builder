"""Landing page: upload, worksheet selection, preview and profiling before generation."""
from __future__ import annotations

import streamlit as st

from .. import config
from ..data_loader import DataLoadError, choose_default_sheet, list_sheets, sheet_dimensions, validate_upload
from ..storage import StorageError
from . import state
from .theme import card, chips, date_column_config, empty_state, esc, human_size, page_header


@st.cache_data(show_spinner=False, max_entries=8)
def _sheets(dataset_id: str, sha: str) -> tuple[list[str], dict]:
    storage = state.get_storage()
    meta = storage.get_dataset(dataset_id)
    path = storage.dataset_file(dataset_id)
    return list_sheets(path, meta["ext"]), sheet_dimensions(path, meta["ext"])


def _handle_upload(uploaded) -> None:
    if st.session_state.get("last_upload_id") == uploaded.file_id:
        return
    st.session_state["last_upload_id"] = uploaded.file_id
    try:
        ext = validate_upload(uploaded.name, uploaded.size)
        data = uploaded.getvalue()
        # Validate that the file can be opened before keeping it.
        list_sheets(data, ext)
        meta = state.get_storage().save_upload(uploaded.name, data, ext)
    except (DataLoadError, StorageError) as exc:
        st.session_state["upload_error"] = str(exc)
        return
    st.session_state.pop("upload_error", None)
    st.session_state["staging"] = {"dataset_id": meta["id"], "sheet": None, "header_row": None}


def render_landing() -> None:
    page_header("Overview", "Build a dashboard from a spreadsheet",
                "Upload an Excel workbook or CSV. The file is analysed and stored locally on this computer.")
    st.write("")
    uploaded = st.file_uploader(
        "Upload a spreadsheet", type=["xlsx", "xls", "csv"],
        help=f"Excel (.xlsx, .xls) or CSV, up to {config.MAX_UPLOAD_MB} MB.",
    )
    if uploaded is not None:
        _handle_upload(uploaded)
        if st.session_state.get("staging"):
            st.rerun()
    if st.session_state.get("upload_error"):
        st.error(st.session_state["upload_error"], icon=":material/error:")

    storage = state.get_storage()
    dashboards = [d for d in storage.list_dashboards() if not d.get("error")][:6]
    st.write("")
    st.markdown("### Recent dashboards")
    if not dashboards:
        empty_state("No dashboards yet", "Upload a spreadsheet above to generate your first dashboard.")
        return
    cols = st.columns(3)
    for i, d in enumerate(dashboards):
        with cols[i % 3], card(f"recent_{d['id']}"):
            st.markdown(f"**{esc(d['name'])}**")
            st.caption(f"{d.get('dataset_name') or 'Dataset'} · {d['n_charts']} charts · updated {str(d.get('updated_at', ''))[:10]}")
            if st.button("Open", key=f"recent_open_{d['id']}", icon=":material/open_in_new:"):
                _open(d["id"])


def _open(dashboard_id: str) -> None:
    try:
        state.open_dashboard(dashboard_id)
    except (StorageError, DataLoadError) as exc:
        st.error(str(exc))
        return
    st.rerun()


_HEADER_AUTO = "Detect automatically"
_HEADER_NONE = "No header row"


def render_staging(staging: dict) -> None:
    storage = state.get_storage()
    try:
        meta = storage.get_dataset(staging["dataset_id"])
        sheets, dims = _sheets(meta["id"], meta["sha256"])
    except (StorageError, DataLoadError) as exc:
        st.error(str(exc))
        if st.button("Start over"):
            state.close_dashboard()
            st.rerun()
        return

    page_header("New dashboard", meta["original_name"], "Check the data below, then generate the dashboard.")
    chip_items = [(human_size(meta["size"]), ""), (meta["ext"][1:].upper(), "")]
    if len(sheets) > 1:
        chip_items.append((f"{len(sheets)} worksheets", "accent"))
    chips(chip_items)
    st.write("")

    c1, c2 = st.columns([2, 1])
    with c1:
        if len(sheets) > 1:
            default = staging.get("sheet") or choose_default_sheet({s: dims.get(s, (0, 0)) for s in sheets}) or sheets[0]
            labels = {s: f"{s}  ·  ~{dims[s][0]:,} rows" if s in dims else s for s in sheets}
            sheet = st.selectbox("Worksheet", sheets, index=sheets.index(default), format_func=lambda s: labels[s])
        else:
            sheet = sheets[0]
            st.markdown(f'<div class="stat-line">Worksheet: <b>{esc(sheet)}</b></div>', unsafe_allow_html=True)
    with c2:
        header_choices = [_HEADER_AUTO] + [f"Row {i}" for i in range(1, 31)] + [_HEADER_NONE]
        current = staging.get("header_row")
        idx = 0 if current is None else (len(header_choices) - 1 if current == -1 else current + 1)
        choice = st.selectbox("Header row", header_choices, index=idx,
                              help="Which spreadsheet row contains the column names. Usually detected automatically.")
    header_row = None if choice == _HEADER_AUTO else (-1 if choice == _HEADER_NONE else int(choice.split()[1]) - 1)
    staging.update(sheet=sheet, header_row=header_row)

    try:
        with st.spinner("Reading and profiling the sheet…"):
            _, result, profile = state.load_for(meta["id"], sheet, header_row)
    except DataLoadError as exc:
        st.error(str(exc), icon=":material/error:")
        st.caption("Try another worksheet or set the header row manually.")
        _staging_actions(can_generate=False)
        return

    m = st.columns(5)
    m[0].metric("Rows", f"{profile.n_rows:,}")
    m[1].metric("Columns", f"{profile.n_cols:,}")
    m[2].metric("Header row", "None" if result.header_row < 0 else f"Row {result.header_row + 1}")
    m[3].metric("Missing cells", f"{profile.missing_pct:.1f}%")
    m[4].metric("Duplicate rows", f"{profile.duplicate_rows:,}")

    roles = profile.to_dict()["roles"]
    chips([(f"{len(roles['metric'])} measures", "accent"), (f"{len(roles['dimension'])} categories", ""),
           (f"{len(roles['date'])} date fields", ""), (f"{len(roles['identifier'])} identifiers", ""),
           (f"{len(roles['text']) + len(roles['empty'])} text/empty", "")])

    notes = result.warnings + profile.warnings
    if notes:
        with st.expander(f"Cleaning notes and assumptions ({len(notes)})", expanded=True, icon=":material/info:"):
            for n in notes:
                st.markdown(f"- {esc(n)}")

    tab1, tab2, tab3 = st.tabs(["Preview", "Column profile", "Original sheet"])
    with tab1:
        st.dataframe(result.df.head(100), width="stretch", height=360, column_config=date_column_config(result.df))
        st.caption(f"First {min(100, len(result.df))} of {len(result.df):,} rows after cleaning.")
    with tab2:
        st.dataframe(profile.summary_frame(), width="stretch", hide_index=True, height=360)
    with tab3:
        try:
            st.dataframe(state.raw_preview(meta["id"], sheet, meta["sha256"]), width="stretch", height=360)
            st.caption("The first 200 rows exactly as stored in the file, before header detection and cleaning.")
        except DataLoadError as exc:
            st.warning(str(exc))
    _staging_actions(can_generate=True)


def _staging_actions(can_generate: bool) -> None:
    st.write("")
    a, b, _ = st.columns([1.3, 1, 4])
    if a.button("Generate dashboard", type="primary", disabled=not can_generate, icon=":material/auto_awesome:"):
        stg = st.session_state["staging"]
        with st.spinner("Analysing data and building charts…"):
            try:
                state.generate_dashboard(stg["dataset_id"], stg["sheet"], stg["header_row"])
            except (DataLoadError, StorageError) as exc:
                st.error(str(exc))
                return
        st.rerun()
    if b.button("Cancel", icon=":material/close:"):
        state.close_dashboard()
        st.session_state.pop("last_upload_id", None)
        st.rerun()
