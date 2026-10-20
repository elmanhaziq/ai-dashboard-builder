"""Table picker: shows the tables found on a worksheet, how sure the app is about each, and lets the user correct it.

Returns a sheet reference (see ``src.structure``) that every later step loads. Plain sheets return the sheet name
unchanged, so ordinary files look and behave exactly as before.
"""
from __future__ import annotations

import html

import pandas as pd
import streamlit as st

from .. import structure
from ..data_loader import DataLoadError, read_raw
from . import buddy, state

WHOLE_SHEET = "__whole__"


@st.cache_data(show_spinner=False, max_entries=8)
def _raw(path: str, ext: str, sheet: str | None, sha: str) -> pd.DataFrame:
    return read_raw(path, ext, sheet)[0]


@st.cache_data(show_spinner=False, max_entries=8)
def _detect(path: str, ext: str, sheet: str | None, sha: str) -> dict:
    s = structure.detect(_raw(path, ext, sheet, sha))
    return {"tables": [t.to_dict() for t in s.tables], "notes": s.notes, "simple": s.simple}


def confidence_label(c: float) -> tuple[str, str]:
    """(label, chip style) for a 0–1 confidence."""
    if c >= 0.85:
        return "High confidence", "accent"
    if c >= 0.65:
        return "Medium confidence", ""
    return "Low confidence — please check", "warn"


def _spec(t: dict) -> dict:
    return {k: t[k] for k in ("top", "left", "bottom", "right", "header_rows", "transpose", "unpivot")}


def _preview(raw: pd.DataFrame, spec: dict) -> pd.DataFrame | None:
    try:
        grid, _ = structure.reconstruct(raw, spec)
    except DataLoadError:
        return None
    if grid.empty:
        return None
    head = [str(h) if h not in (None, "") else f"Column {i + 1}" for i, h in enumerate(grid.iloc[0])]
    seen: dict[str, int] = {}
    cols = []
    for h in head:  # the preview only needs unique labels
        seen[h] = seen.get(h, 0) + 1
        cols.append(h if seen[h] == 1 else f"{h} ({seen[h]})")
    body = grid.iloc[1:9].copy()
    body.columns = cols
    return body.map(_cell_text)


def _cell_text(v) -> str:
    if v is None or (isinstance(v, float) and v != v):
        return ""
    if hasattr(v, "strftime") and getattr(v, "hour", 0) == 0 and getattr(v, "minute", 0) == 0:
        return v.strftime("%Y-%m-%d")
    return str(v)


def table_picker(dataset_id: str, sheet: str | None, key: str) -> str | None:
    """Show the structure of ``sheet`` and return the sheet reference to load (None = nothing usable chosen)."""
    storage = state.get_storage()
    meta = storage.get_dataset(dataset_id)
    path = str(storage.dataset_file(dataset_id))
    try:
        info = _detect(path, meta["ext"], sheet, meta["sha256"])
    except DataLoadError:
        return sheet
    tables = info["tables"]
    if info["simple"] or not tables:
        return sheet
    real = sheet or structure.CSV_NAME
    raw = _raw(path, meta["ext"], sheet, meta["sha256"])

    n = len(tables)
    with st.container(border=True, key=f"tblpick_{key}"):
        st.markdown(f"**I looked at the layout of this sheet: {n} table{'s' if n > 1 else ''} found**")
        if n > 1:
            msg = (f"This sheet holds {n} separate tables. I'll analyse one at a time — pick the one you want. "
                   "You can start another analysis for the others later.")
        else:
            msg = "This sheet isn't a plain table, so I rebuilt it as one. Check the preview below."
        if info["notes"]:
            msg += " I also found some notes around the data: <i>" + html.escape("; ".join(info["notes"][:2])[:200]) + "</i>"
        buddy.inline("present", msg)

        def label(i: int) -> str:
            t = tables[i]
            title = t["title"].split(" — ")[0]
            if len(title) > 42:
                title = title[:42].rsplit(" ", 1)[0] + "…"
            name = f" “{title}”" if title else ""
            return f"Table {i + 1}{name} · cells {t['range']} · ~{t['n_records']:,} rows × {t['n_fields']} columns"
        options = list(range(n)) + [WHOLE_SHEET]
        pick = st.radio("Which table should I use?", options, key=f"tblsel_{key}",
                        format_func=lambda i: "Use the whole sheet as it is (old behaviour)" if i == WHOLE_SHEET else label(i))
        if pick == WHOLE_SHEET:
            return sheet
        t = tables[pick]
        conf_text, style = confidence_label(t["confidence"])
        st.markdown(f'<span class="chip {style}">{conf_text} ({t["confidence"]:.0%})</span>', unsafe_allow_html=True)
        for name, (c, reason) in t["checks"].items():
            if reason:
                st.markdown(f"- {'✅' if c >= 0.85 else '🟡' if c >= 0.65 else '❓'} {html.escape(reason)}")

        # Corrections double as answers to the clarification questions below.
        spec = _spec(t)
        k = f"{key}_{pick}"
        low = {name for name, (c, _) in t["checks"].items() if c < 0.7}
        if low:
            st.markdown("**I'm not sure about this table — could you check?**")
        c1, c2, c3 = st.columns(3)
        flip_q = "Do the records go across the columns? (field names down column A)"
        spec["transpose"] = c1.toggle("Records go across" if "orientation" not in low else flip_q,
                                      value=t["transpose"], key=f"flip_{k}",
                                      help="Turn on when each column is one record and the field names are in the first column.")
        spec["header_rows"] = c2.number_input("Header rows", 0, 3, value=t["header_rows"], key=f"hdr_{k}",
                                              help="How many rows at the top hold the column names (merged headings count).")
        can_unpivot = bool(t["unpivot"]) or t["period_kind"] is not None
        if can_unpivot:
            on = c3.toggle("Months/periods as rows", value=bool(t["unpivot"]), key=f"unp_{k}",
                           help="One row per label and period instead of one column per period. Best for trend charts.")
            spec["unpivot"] = (t["unpivot"] or 1) if on else None
        if spec["header_rows"] != 1:
            spec["unpivot"] = None
        prev = _preview(raw, spec)
        if prev is None:
            st.warning("With these settings the table is empty. Try other settings.")
            return None
        st.caption("Preview of the rebuilt table, values as they are in your file. Dates, money amounts and "
                   "spellings are tidied in the next steps.")
        st.dataframe(prev, hide_index=True, width="stretch", height=min(320, 38 + 35 * len(prev)))
        return structure.make_ref(real, spec)
