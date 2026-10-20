"""Excel-style view of the uploaded data, with the cells behind each clean-up suggestion highlighted.

Columns get spreadsheet letters (A, B, C…) and rows are numbered, so people who know Excel recognise the layout.
During the clean-up step the buddy explains the colours; choosing one suggestion shows only its rows.
"""
from __future__ import annotations

import html

import pandas as pd
import streamlit as st

from ..agent.quality import issue_cells
from ..data_profiler import DATE, METRIC, ROLE_LABELS
from ..formatting import format_value
from . import buddy

MAX_ROWS = 400
COLORS = {"missing": "#fff3bf", "inconsistent_labels": "#e5dbff", "duplicates": "#ffd8a8",
          "possible_duplicates": "#ffe8cc", "outliers": "#ffc9c9"}
LABELS = {"missing": "Missing value", "inconsistent_labels": "Same value written differently",
          "duplicates": "Duplicate row", "possible_duplicates": "Repeated ID", "outliers": "Unusual number"}
DONE = {"approved": "you approved a fix", "rejected": "you chose to leave it", "pending": "waiting for your decision",
        "info": "for information only"}

_CSS = """<style>
.xl-ribbon { display:flex; align-items:center; gap:10px; background:#107c41; color:#fff; border-radius:10px 10px 0 0;
  padding:8px 12px; font-size:.84rem; font-weight:600; }
.xl-ribbon .xl-icon { background:#fff; color:#107c41; border-radius:4px; font-weight:800; padding:0 6px; font-size:.78rem; }
.xl-ribbon .xl-meta { margin-left:auto; font-weight:500; opacity:.9; font-size:.78rem; }
.xl-legend { display:flex; flex-wrap:wrap; gap:6px; margin:4px 0 6px 0; }
.xl-legend span { display:inline-flex; align-items:center; gap:6px; font-size:.76rem; color:#374151; border:1px solid #e3e6ea;
  border-radius:999px; padding:2px 9px; background:#fff; }
.xl-legend i { width:12px; height:12px; border-radius:3px; display:inline-block; border:1px solid rgba(0,0,0,.12); }
</style>"""


def _letters(n: int) -> list[str]:
    out = []
    for i in range(n):
        s, k = "", i + 1
        while k:
            k, r = divmod(k - 1, 26)
            s = chr(65 + r) + s
        out.append(s)
    return out


def _cells(ctx) -> list[dict]:
    """issue_cells is cached per analysis and set of issues (it scans the whole file)."""
    m = ctx.memory
    key = (m.dataset_id, m.sheet, m.header_row, tuple((i["id"], i.get("status")) for i in m.cleaning_issues))
    cache = st.session_state.setdefault("_sheet_cells", {})
    if key not in cache:
        cache.clear()
        cache[key] = issue_cells(ctx.original.df, m.cleaning_issues) if m.cleaning_issues else []
    return cache[key]


EMPTY = "(empty)"


def _blank(v) -> bool:
    return v is None or (not isinstance(v, str) and bool(pd.isna(v)))


def _fmt(profile, col):
    cp = profile.columns.get(col)
    if cp is not None and cp.role == DATE:
        return lambda v: EMPTY if _blank(v) else (v.strftime("%Y-%m-%d") if hasattr(v, "strftime") else str(v))
    if cp is not None and cp.role == METRIC:
        def num(v):
            if _blank(v):
                return EMPTY
            try:
                return format_value(v, decimals=0 if float(v).is_integer() else 2)
            except (TypeError, ValueError):
                return str(v)
        return num
    return lambda v: EMPTY if _blank(v) else str(v)


def _grid(df: pd.DataFrame, profile, active: list[dict], only_problem_rows: bool) -> tuple[object, int, int]:
    rows_mask = None
    if active and only_problem_rows:
        rows_mask = active[0]["rows"].copy()
        for c in active[1:]:
            rows_mask |= c["rows"]
    idx = df.index[rows_mask.to_numpy()] if rows_mask is not None else df.index
    total = len(idx)
    idx = idx[:MAX_ROWS]
    view = df.loc[idx]
    letters = _letters(len(df.columns))
    flagged = {c for a in active for c in (a["columns"] or [])}
    names = {c: f"{letters[i]} · {c}" + (" ⚠" if c in flagged else "") for i, c in enumerate(df.columns)}
    styles = pd.DataFrame("", index=view.index, columns=view.columns)
    # whole-row issues first, so cell-level colours win where both apply
    for a in sorted(active, key=lambda a: a["columns"] is not None):
        hit = a["rows"].reindex(view.index, fill_value=False).to_numpy()
        for c in a["columns"] or list(view.columns):
            styles.loc[hit, c] = f"background-color: {COLORS.get(a['kind'], '#fff3bf')}; color: #1c1f24"
    positions = df.index.get_indexer(view.index) + 1
    view = view.copy()
    view.index = pd.Index(positions, name="Row")
    styles.index = view.index
    # Cells become display text here: the grid would otherwise show blanks as "None".
    for c in view.columns:
        view[c] = view[c].map(_fmt(profile, c)).astype(str)
    view.columns = [names[c] for c in view.columns]
    styles.columns = view.columns
    styler = view.style.apply(lambda _: styles, axis=None)
    return styler, len(view), total


def _structure(df: pd.DataFrame, profile, cells: list[dict]) -> pd.DataFrame:
    letters = _letters(len(df.columns))
    rows = []
    for i, c in enumerate(df.columns):
        cp = profile.columns.get(c)
        probs = [f"{x['title']} ({x['count']:,})" for x in cells if x["columns"] and c in x["columns"]]
        rows.append({
            "": letters[i], "Column": c,
            "Type": ROLE_LABELS.get(cp.role, cp.role) if cp else "—",
            "Filled": f"{100 - cp.missing_pct:.0f}%" if cp else "—",
            "Different values": f"{cp.unique:,}" if cp else "—",
            "Most common": ", ".join(str(v) for v, _ in (cp.top_values or [])[:3]) if cp else "",
            "To check": "; ".join(probs) or "—",
        })
    return pd.DataFrame(rows)


def _buddy_text(m, cells: list[dict], focus: dict | None, issue_by_id: dict, review: bool) -> tuple[str, str]:
    if focus:
        issue = issue_by_id.get(focus["id"], {})
        rec = (issue.get("recommended") or {}).get("label", "")
        status = DONE.get(issue.get("status"), "")
        text = (f"<b>{html.escape(focus['title'])}</b> — {focus['count']:,} highlighted. {html.escape(issue.get('detail', ''))} "
                f"{html.escape(issue.get('why', ''))}" + (f"<br><b>My suggestion:</b> {html.escape(rec)}" if rec else "")
                + (f" <i>({status})</i>" if status else ""))
        return "present", text
    if not m.cleaning_issues:
        return "idle", ("Here's your data the way I read it: each column has a letter, just like Excel. When we reach the "
                        "clean-up step I'll colour every cell that needs attention.")
    if not cells:
        return "celebrate", "I checked every cell and found nothing that needs highlighting. Nice and tidy!"
    kinds = sorted({c["kind"] for c in cells}, key=list(COLORS).index)
    colours = ", ".join(f"<b>{LABELS[k].lower()}</b> = {_colour_name(k)}" for k in kinds)
    lead = ("I coloured the cells behind my clean-up suggestions" if review else "These are the cells I flagged earlier")
    return "present", f"{lead}: {colours}. Pick one in the list to see only its rows."


def _colour_name(kind: str) -> str:
    return {"missing": "yellow", "inconsistent_labels": "purple", "duplicates": "orange", "possible_duplicates": "peach",
            "outliers": "red"}[kind]


def render_sheet(ctx, review: bool) -> None:
    """The spreadsheet panel. ``review`` = the clean-up step is active (the panel opens by itself)."""
    m = ctx.memory
    r = ctx.original
    df, profile = r.df, ctx.original_profile
    cells = _cells(ctx)
    issue_by_id = {i["id"]: i for i in m.cleaning_issues}
    label = "Spreadsheet view — see what needs cleaning" if review else "Spreadsheet view of your data"
    with st.expander(label, expanded=review, icon=":material/grid_on:"):
        st.html(_CSS)
        st.html(f'<div class="xl-ribbon"><span class="xl-icon">X</span>{html.escape(m.dataset_name)}'
                f'{" · " + html.escape(m.sheet) if m.sheet and m.sheet != "CSV data" else ""}'
                f'<span class="xl-meta">{len(df):,} rows × {len(df.columns)} columns</span></div>')
        tabs = st.tabs(["Sheet", "Column structure"] + (["After clean-up"] if m.applied_ops else []))
        with tabs[0]:
            focus = None
            if cells:
                options = ["All highlighted problems"] + [f"{c['title']} · {c['count']:,} rows" for c in cells]
                a, b = st.columns([3, 1.3], vertical_alignment="bottom")
                pick = a.selectbox("Show", options, key=f"xl_focus_{m.dataset_id}", label_visibility="collapsed")
                only = b.toggle("Problem rows only", value=True, key=f"xl_only_{m.dataset_id}")
                focus = cells[options.index(pick) - 1] if pick != options[0] else None
            else:
                only = False
            mood, text = _buddy_text(m, cells, focus, issue_by_id, review)
            buddy.inline(mood, text)
            active = [focus] if focus else cells
            if active:
                st.html('<div class="xl-legend">' + "".join(
                    f'<span><i style="background:{COLORS[k]}"></i>{LABELS[k]}</span>'
                    for k in dict.fromkeys(a["kind"] for a in active)) + "</div>")
            styler, shown, total = _grid(df, profile, active, only or bool(focus))
            st.dataframe(styler, height=420, width="stretch")
            more = f" (first {shown:,} of {total:,})" if total > shown else ""
            st.caption(f"Showing {shown:,} row(s){more}. Your original file is never changed."
                       + (" Columns marked ⚠ have something to check." if active else ""))
        with tabs[1]:
            st.dataframe(_structure(df, profile, cells), hide_index=True, width="stretch")
            st.caption("Type = how I'll use the column: Measures are added up or averaged, Categories are used to "
                       "group, Dates make trends, Identifiers are labels and aren't added up.")
        if m.applied_ops:
            with tabs[2]:
                work, wprofile = ctx.working()
                buddy.inline("celebrate", f"This is your data after the {len(m.applied_ops)} change(s) you approved "
                                          f"({len(work):,} rows). The dashboard uses this version.")
                styler, shown, total = _grid(work, wprofile, [], False)
                st.dataframe(styler, height=380, width="stretch")
