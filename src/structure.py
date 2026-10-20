"""Find the tables inside a messy worksheet and rebuild each one as a clean, one-row-per-record table.

Real spreadsheets are often reports, not tables: a title and notes above the data, two or more tables on one sheet,
headers spread over two rows (merged cells), records laid out sideways (field names down the first column), or a
"wide" layout with one column per month. This module

1. **detects** rectangular blocks of cells separated by blank rows/columns (recursive XY-cut, then re-joins blocks that
   were only split by a spacer row or column),
2. **analyses** each block: title rows, how many header rows, whether records run down or across, and whether period
   columns (months, quarters, years) should be turned into rows — each with a confidence score and plain reasons,
3. **reconstructs** the chosen block into a grid whose first row is the header, which the normal loader then cleans.

A reconstruction is identified by a *sheet reference*: the worksheet name plus a compact spec, e.g.
``"Report§r5c1-r40c6h2TU1"`` (rows/cols are 0-based, inclusive; h = header rows; T = flip; U1 = unpivot with 1 label
column). Everything downstream (analyses, dashboards, caches) stores this string like an ordinary sheet name, so a
dashboard built from a reconstructed table rebuilds it the same way every time it is opened.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from datetime import date, datetime

import numpy as np
import pandas as pd

SEP = "§"
CSV_NAME = "CSV data"  # same as data_loader.CSV_SHEET_NAME
MAX_SCAN_ROWS = 200_000  # very large sheets are treated as one plain table (detection would be slow)
SAMPLE = 250  # rows used to judge cell types inside a block

_SPEC_RE = re.compile(r"^r(\d+)c(\d+)-r(\d+)c(\d+)h(\d+)(T?)(?:U(\d+))?$")
_MONTHS = {m: i for i, names in enumerate([
    ("jan", "january", "januari"), ("feb", "february", "februari"), ("mar", "march", "mac"), ("apr", "april"),
    ("may", "mei"), ("jun", "june"), ("jul", "july", "julai"), ("aug", "august", "ogos"), ("sep", "sept", "september"),
    ("oct", "october", "okt", "oktober"), ("nov", "november"), ("dec", "december", "dis", "disember")], start=1)
    for m in names}
_QUARTER_RE = re.compile(r"^q([1-4])(?:\s*[-'/ ]?\s*((?:19|20)?\d{2}))?$", re.I)
_YEAR_RE = re.compile(r"^(?:fy\s*)?((?:19|20)\d{2})$", re.I)
_YM_RE = re.compile(r"^((?:19|20)\d{2})[-/. ](0?[1-9]|1[0-2])$")
_MONTH_YEAR_RE = re.compile(r"^([a-z]+)[\s\-'/.]*((?:19|20)?\d{2})?$", re.I)
_WEEK_RE = re.compile(r"^(?:w|wk|week)\s*\d{1,2}$", re.I)
_TOTAL_COL_RE = re.compile(r"^(grand\s+)?(total|sum|ytd|year\s*to\s*date|jumlah|overall)\b", re.I)
_MEASURE_RE = re.compile(r"\b(sales|revenue|profit|units?|quantity|qty|cost|costs|amount|orders|visits|income|expenses?|"
                         r"spend|budget|target|volume|count|jualan|hasil)\b", re.I)
_UNIT_RE = re.compile(r"\((rm|myr|usd|\$|%|units?|kg|tonnes?)\)", re.I)


# --------------------------------------------------------------------------- sheet references

def split_ref(sheet: str | None) -> tuple[str | None, dict | None]:
    """'Report§r5c1-r40c6h2' → ('Report', spec). A plain sheet name returns (name, None)."""
    if not sheet or SEP not in sheet:
        return sheet, None
    name, _, raw = sheet.rpartition(SEP)
    m = _SPEC_RE.match(raw)
    if not m:
        return sheet, None
    top, left, bottom, right, hdr, flip, unpivot = m.groups()
    return name, {"top": int(top), "left": int(left), "bottom": int(bottom), "right": int(right), "header_rows": int(hdr),
                  "transpose": bool(flip), "unpivot": int(unpivot) if unpivot else None}


def make_ref(sheet: str, spec: dict) -> str:
    s = f"r{spec['top']}c{spec['left']}-r{spec['bottom']}c{spec['right']}h{spec['header_rows']}"
    s += "T" if spec.get("transpose") else ""
    s += f"U{spec['unpivot']}" if spec.get("unpivot") else ""
    return f"{sheet}{SEP}{s}"


def col_letter(i: int) -> str:
    s, k = "", i + 1
    while k:
        k, r = divmod(k - 1, 26)
        s = chr(65 + r) + s
    return s


def cell_range(spec: dict) -> str:
    return f"{col_letter(spec['left'])}{spec['top'] + 1}:{col_letter(spec['right'])}{spec['bottom'] + 1}"


def display_sheet(sheet: str | None) -> str:
    """Human-readable name for a sheet reference."""
    name, spec = split_ref(sheet)
    if not spec:
        return name or ""
    extra = [x for x, on in (("flipped", spec["transpose"]), ("months as rows" if spec["unpivot"] else "", spec["unpivot"]))
             if on and x]
    return f"{name} · table {cell_range(spec)}" + (f" ({', '.join(extra)})" if extra else "")


# --------------------------------------------------------------------------- cell classification

def _blank(v) -> bool:
    if v is None:
        return True
    if isinstance(v, str):
        return not v.strip()
    try:
        return bool(pd.isna(v))
    except (TypeError, ValueError):
        return False


_NUM_RE = re.compile(r"^[(\-+]?\s*(?:rm|\$|€|£|myr)?\s*[\d,]*\.?\d+\s*\)?\s*%?$", re.I)
_DATE_RE = re.compile(r"^\d{1,4}[/.\-]\d{1,2}[/.\-]\d{1,4}(?:[ T]\d{1,2}:\d{2}(?::\d{2})?)?$")


def kind(v) -> str:
    """'b' blank, 'n' number, 'd' date, 't' text."""
    if _blank(v):
        return "b"
    if isinstance(v, (datetime, date, pd.Timestamp, np.datetime64)):
        return "d"
    if isinstance(v, bool):
        return "t"
    if isinstance(v, (int, float, np.number)):
        return "n"
    s = str(v).strip()
    if _NUM_RE.match(s) and any(ch.isdigit() for ch in s):
        return "n"
    if _DATE_RE.match(s):
        return "d"
    return "t"


def period_label(v) -> tuple[str, object] | None:
    """Recognise column labels that name a time period. Returns (kind, sort key) or None."""
    if isinstance(v, (datetime, date, pd.Timestamp)):
        return "date", pd.Timestamp(v)
    if isinstance(v, (int, np.integer)) or (isinstance(v, float) and float(v).is_integer()):
        y = int(v)
        return ("year", y) if 1900 <= y <= 2100 else None
    if not isinstance(v, str):
        return None
    s = re.sub(r"\s+", " ", v.strip())
    if not s:
        return None
    if m := _YEAR_RE.match(s):
        return "year", int(m.group(1))
    if m := _QUARTER_RE.match(s):
        return "quarter", (m.group(2) or "", int(m.group(1)))
    if m := _YM_RE.match(s):
        return "month", (int(m.group(1)), int(m.group(2)))
    if _WEEK_RE.match(s):
        return "week", int(re.sub(r"\D", "", s))
    if m := _MONTH_YEAR_RE.match(s):
        word = m.group(1).lower().rstrip(".")
        if word in _MONTHS:
            return "month", (m.group(2) or "", _MONTHS[word])
    if _DATE_RE.match(s):
        return "date", s
    return None


# --------------------------------------------------------------------------- results

@dataclass
class Table:
    top: int
    left: int
    bottom: int
    right: int
    header_rows: int = 1
    transpose: bool = False
    unpivot: int | None = None  # number of label columns kept when period columns become rows
    title: str = ""
    n_records: int = 0
    n_fields: int = 0
    confidence: float = 1.0
    checks: dict = field(default_factory=dict)  # name → (confidence, reason)
    columns: list[str] = field(default_factory=list)
    period_kind: str | None = None

    @property
    def spec(self) -> dict:
        return {"top": self.top, "left": self.left, "bottom": self.bottom, "right": self.right,
                "header_rows": self.header_rows, "transpose": self.transpose, "unpivot": self.unpivot}

    @property
    def range(self) -> str:
        return cell_range(self.spec)

    def ref(self, sheet: str) -> str:
        return make_ref(sheet, self.spec)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["range"] = self.range
        return d


@dataclass
class SheetStructure:
    tables: list[Table]
    notes: list[str]
    simple: bool  # one ordinary table that the normal loader handles fine
    shape: tuple[int, int]


# --------------------------------------------------------------------------- block detection (XY-cut)

def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Contiguous runs of True as (start, end) inclusive."""
    out, start = [], None
    for i, v in enumerate(mask):
        if v and start is None:
            start = i
        elif not v and start is not None:
            out.append((start, i - 1))
            start = None
    if start is not None:
        out.append((start, len(mask) - 1))
    return out


def _cut(occ: np.ndarray, r0: int, r1: int, c0: int, c1: int, out: list, depth: int = 0) -> None:
    sub = occ[r0:r1 + 1, c0:c1 + 1]
    rows, cols = _runs(sub.any(axis=1)), _runs(sub.any(axis=0))
    if not rows or not cols:
        return
    # trim to the bounding box
    r0, r1, c0, c1 = r0 + rows[0][0], r0 + rows[-1][1], c0 + cols[0][0], c0 + cols[-1][1]
    if depth > 12:
        out.append((r0, c0, r1, c1))
        return
    if len(rows) > 1:
        base = r0 - rows[0][0]
        for a, b in rows:
            _cut(occ, base + a, base + b, c0, c1, out, depth + 1)
        return
    if len(cols) > 1:
        base = c0 - cols[0][0]
        for a, b in cols:
            _cut(occ, r0, r1, base + a, base + b, out, depth + 1)
        return
    out.append((r0, c0, r1, c1))


def _header_like_row(values: list) -> bool:
    vals = [v for v in values if not _blank(v)]
    if len(vals) < 2:
        return False
    texty = sum(kind(v) == "t" or period_label(v) is not None for v in vals)
    return texty / len(vals) >= 0.7


def _merge_blocks(raw_vals: np.ndarray, blocks: list[tuple]) -> list[tuple]:
    """Re-join pieces of one table that a single spacer row or column split apart."""
    changed = True
    blocks = sorted(blocks)
    while changed:
        changed = False
        for i in range(len(blocks)):
            for j in range(len(blocks)):
                if i == j:
                    continue
                a, b = blocks[i], blocks[j]
                # vertical: same columns, one blank row between, and the lower piece doesn't start with a header
                if (a[1], a[3]) == (b[1], b[3]) and b[0] - a[2] == 2 and (a[2] - a[0]) >= 1 \
                        and not _header_like_row(list(raw_vals[b[0], b[1]:b[3] + 1])):
                    blocks[i] = (a[0], a[1], b[2], b[3])
                    del blocks[j]
                    changed = True
                    break
                # horizontal: same rows, one blank column between (a spacer column inside one table)
                if (a[0], a[2]) == (b[0], b[2]) and b[1] - a[3] == 2 and (a[2] - a[0]) >= 2:
                    blocks[i] = (a[0], a[1], b[2], b[3])
                    del blocks[j]
                    changed = True
                    break
            if changed:
                break
    return blocks


# --------------------------------------------------------------------------- block analysis

def _kinds(block: np.ndarray) -> np.ndarray:
    if len(block) > SAMPLE:
        block = np.concatenate([block[:SAMPLE - 50], block[-50:]])
    return np.vectorize(kind, otypes=[object])(block) if block.size else np.empty(block.shape, dtype=object)


def _dominant_share(arr) -> float:
    vals = [k for k in arr if k != "b"]
    if not vals:
        return 1.0
    counts = pd.Series(vals).value_counts()
    return float(counts.iloc[0]) / len(vals)


def _orientation(k: np.ndarray, vals: np.ndarray) -> tuple[bool, float, str]:
    """Do records run across (field names down the first column)? Returns (flip, confidence, reason)."""
    if k.shape[0] < 3 or k.shape[1] < 3:
        return False, 0.9, "The table is small, so I read it the usual way (one record per row)."
    start = max(1, _header_count(k, vals)[0])  # header rows would make every column look mixed
    if k.shape[0] - start < 2:
        return False, 0.85, "There are only a few rows, so I read one record per row."
    first_col = [x for x in k[start:, 0]]
    labels_text = sum(x == "t" for x in first_col) / max(1, len(first_col))
    body = k[start:, 1:]
    col_mixed = float(np.mean([1 - _dominant_share(body[:, j]) for j in range(body.shape[1])]))
    row_mixed = float(np.mean([1 - _dominant_share(body[i, :]) for i in range(body.shape[0])]))
    kinds_rows = {next((x for x in body[i] if x != "b"), "b") for i in range(body.shape[0])}
    if labels_text >= 0.9 and col_mixed >= 0.25 and row_mixed <= 0.1 and len(kinds_rows - {"b"}) >= 2:
        conf = min(0.95, 0.6 + col_mixed)
        return True, conf, ("Each row holds one kind of value (for example all numbers or all dates) while each column "
                            "mixes kinds, so the records run across: the field names are in the first column.")
    if col_mixed <= 0.1:
        return False, 0.95, "Each column holds one kind of value, so there is one record per row."
    return False, max(0.55, 0.9 - col_mixed), "Most columns hold one kind of value, so I read one record per row."


def _header_count(k: np.ndarray, vals: np.ndarray) -> tuple[int, float, str]:
    """How many of the leading rows are header rows."""
    n = k.shape[0]
    if n < 2:
        return 0, 0.5, "The block has a single row."
    lower = k[min(n - 1, 3):]
    numeric_cols = [j for j in range(k.shape[1]) if sum(x in "nd" for x in lower[:, j]) >= 0.6 * max(1, sum(x != "b" for x in lower[:, j]))
                    and any(x in "nd" for x in lower[:, j])]
    def header_row(i: int) -> bool:
        row_k, row_v = k[i], vals[i]
        nonblank = [j for j in range(len(row_k)) if row_k[j] != "b"]
        if not nonblank:
            return False
        for j in numeric_cols:
            if row_k[j] in "nd" and period_label(row_v[j]) is None:
                return False
        texty = sum(row_k[j] == "t" or period_label(row_v[j]) is not None for j in nonblank)
        return texty / len(nonblank) >= 0.6
    if not numeric_cols:  # all-text table: every row "looks" like a header, so only the first row can be one
        first = [str(v).strip().lower() for v in vals[0] if not _blank(v)]
        if first and len(set(first)) == len(first):
            return 1, 0.6, "All columns are text, so I used the first row as the column names (please check)."
        return 0, 0.5, "All columns are text and the first row repeats values, so I couldn't find column names."
    count = 0
    while count < min(3, n - 1) and header_row(count):
        count += 1
    if count == 0:
        return 0, 0.7, "The first row already looks like data, so the table seems to have no header row."
    if count >= 2:
        return count, 0.8, (f"The column names are spread over {count} rows (for example a merged heading above "
                            "sub-headings); I combined them into one name per column.")
    return 1, 0.95 if numeric_cols else 0.75, "The first row holds the column names."


def _title_rows(vals: np.ndarray, k: np.ndarray) -> int:
    """Leading rows that hold only a title: one filled cell, at the table's left edge, above a fuller row.

    A lone cell further right is a merged group heading (e.g. "Amount" over "Sales" and "Units"), not a title."""
    t = 0
    while (t < min(3, k.shape[0] - 2) and k.shape[1] >= 2 and sum(x != "b" for x in k[t]) == 1 and k[t][0] != "b"
           and sum(x != "b" for x in k[t + 1]) >= 2):
        t += 1
    return t


def _title_above(raw_vals: np.ndarray, top: int, left: int, right: int) -> str:
    """Text of the title lines above a table: lone text cells in the 4 rows above it (blank rows allowed)."""
    lines = []
    for r in range(top - 1, max(-1, top - 5), -1):
        row = [v for v in raw_vals[r, max(0, left - 1): right + 1] if not _blank(v)]
        if not row:
            continue
        if len(row) == 1 and isinstance(row[0], str):
            lines.insert(0, row[0].strip())
            continue
        break
    return " — ".join(lines)[:160]


def _flatten_header(rows: list[list]) -> list[str]:
    """Combine 1–3 header rows into one name per column; merged (blank) cells repeat the value to their left."""
    filled = []
    for depth, row in enumerate(rows):
        out, last = [], None
        for v in row:
            if not _blank(v):
                last = v
                out.append(v)
            else:
                out.append(last if depth < len(rows) - 1 else None)
        filled.append(out)
    parts_by_col = []
    for j in range(len(rows[0])):
        parts = []
        for row in filled:
            v = row[j]
            if _blank(v):
                continue
            s = _label(v)
            if s and s not in parts:
                parts.append(s)
        parts_by_col.append(parts)
    # Keep a group heading only where it's needed to tell columns apart ("Q1 Sales" vs "Q2 Sales");
    # a unique bottom label ("Units" under "Amount") stands on its own.
    bottoms = [p[-1].casefold() if p else "" for p in parts_by_col]
    return [(p[-1] if p and bottoms.count(p[-1].casefold()) == 1 else " ".join(p)) for p in parts_by_col]


def _label(v) -> str:
    if isinstance(v, (datetime, date, pd.Timestamp)):
        ts = pd.Timestamp(v)
        return ts.strftime("%Y-%m-%d") if (ts.hour, ts.minute) == (0, 0) else ts.isoformat()
    if isinstance(v, float) and v.is_integer():
        return str(int(v))
    return re.sub(r"\s+", " ", str(v)).strip()


def _wide(header: list, k_body: np.ndarray) -> tuple[int | None, str | None, float, str]:
    """Are there ≥3 period columns (months, quarters, years…) holding numbers? Returns (label cols, kind, conf, reason)."""
    periods = [period_label(v) if not _blank(v) else None for v in header]
    idx = [j for j, p in enumerate(periods) if p]
    if len(idx) < 3:
        return None, None, 0.9, ""
    kinds_ = {p[0] for p in periods if p}
    first = idx[0]
    numeric = all(sum(x == "n" for x in k_body[:, j]) >= 0.6 * max(1, sum(x != "b" for x in k_body[:, j])) for j in idx)
    if first == 0 or not numeric:
        return None, None, 0.9, ""
    labels = first
    main = max(kinds_, key=lambda kk: sum(1 for p in periods if p and p[0] == kk))
    share = len(idx) / max(1, len(header) - labels)
    conf = 0.9 if share >= 0.7 else 0.7
    return labels, main, conf, (f"{len(idx)} columns are {main}s holding numbers. A layout with one column per {main} is "
                                f"easier to chart as rows: one row per label and {main}.")


def analyse_block(raw_vals: np.ndarray, top: int, left: int, bottom: int, right: int) -> Table | None:
    block = raw_vals[top:bottom + 1, left:right + 1]
    k = _kinds(block)
    t = _title_rows(block, k)
    title = " ".join(_label(v) for v in block[:t].ravel() if not _blank(v))
    top += t
    block, k = block[t:], k[t:]
    if block.shape[0] < 2 or block.shape[1] < 2 or sum(x != "b" for x in k.ravel()) < 4:
        return None
    title = title or _title_above(raw_vals, top, left, right)
    flip, o_conf, o_reason = _orientation(k, block)
    vals = block.T if flip else block
    kk = k.T if flip else k
    hdr, h_conf, h_reason = _header_count(kk, vals)
    header = _flatten_header([list(r) for r in vals[:hdr]]) if hdr else [f"Column {j + 1}" for j in range(vals.shape[1])]
    labels, pkind, w_conf, w_reason = (None, None, 0.9, "")
    if hdr == 1:
        labels, pkind, w_conf, w_reason = _wide(list(vals[0]), kk[1:])
    checks = {"header": (h_conf, h_reason), "orientation": (o_conf, o_reason)}
    if labels:
        checks["layout"] = (w_conf, w_reason)
    tbl = Table(top=top, left=left, bottom=bottom, right=right, header_rows=hdr, transpose=flip, unpivot=labels,
                title=title, n_records=vals.shape[0] - hdr, n_fields=vals.shape[1], checks=checks,
                columns=header, period_kind=pkind)
    tbl.confidence = round(min(c for c, _ in checks.values()), 2)
    return tbl


def detect(raw: pd.DataFrame) -> SheetStructure:
    """Find and analyse every table on a raw worksheet (no header interpretation)."""
    if raw.empty:
        return SheetStructure([], [], True, (0, 0))
    vals = raw.to_numpy(dtype=object)
    if len(vals) > MAX_SCAN_ROWS:
        return SheetStructure([], ["The sheet is very large, so it was read as one table."], True, raw.shape)
    occ = ~pd.DataFrame(vals).map(_blank).to_numpy()
    blocks: list[tuple] = []
    _cut(occ, 0, occ.shape[0] - 1, 0, occ.shape[1] - 1, blocks)
    blocks = _merge_blocks(vals, blocks)
    tables, notes = [], []
    for (r0, c0, r1, c1) in blocks:
        tbl = analyse_block(vals, r0, c0, r1, c1)
        if tbl is None:
            text = " ".join(_label(v) for v in vals[r0:r1 + 1, c0:c1 + 1].ravel() if not _blank(v))
            if text:
                notes.append(text[:200])
            continue
        tables.append(tbl)
    tables.sort(key=lambda t: (t.top, t.left))
    simple = (len(tables) == 1 and not tables[0].transpose and not tables[0].unpivot and tables[0].header_rows <= 1)
    return SheetStructure(tables, notes, simple, raw.shape)


# --------------------------------------------------------------------------- reconstruction

def _period_values(labels: list, kind_: str | None, title: str) -> tuple[list, str]:
    """Turn period labels into dates where the year is known, else readable text. Returns (values, column name)."""
    name = {"month": "Month", "quarter": "Quarter", "year": "Year", "week": "Week", "date": "Date"}.get(kind_ or "", "Period")
    year_in_title = re.search(r"\b((?:19|20)\d{2})\b", title or "")
    out = []
    for v in labels:
        p = period_label(v)
        if p and p[0] == "month":
            y, m = p[1]
            y = (int(y) + (2000 if int(y) < 100 else 0)) if y else (int(year_in_title.group(1)) if year_in_title else None)
            out.append(pd.Timestamp(year=y, month=m, day=1) if y else _label(v))
        elif p and p[0] == "date":
            try:
                out.append(pd.Timestamp(p[1]))
            except (ValueError, TypeError):
                out.append(_label(v))
        elif p and p[0] == "year":
            out.append(int(p[1]))
        else:
            out.append(_label(v))
    return out, name


def _value_name(title: str) -> str:
    m = _MEASURE_RE.search(title or "")
    if not m:
        return "Value"
    name = m.group(1).capitalize()
    unit = _UNIT_RE.search(title)
    return f"{name} ({unit.group(1).upper()})" if unit else name


def reconstruct(raw: pd.DataFrame, spec: dict) -> tuple[pd.DataFrame, list[str]]:
    """Cut out, orient, flatten and (optionally) unpivot one table. Returns (grid with header in row 0, notes)."""
    vals = raw.to_numpy(dtype=object)
    top, left, bottom, right = spec["top"], spec["left"], spec["bottom"], spec["right"]
    if bottom >= vals.shape[0] or right >= vals.shape[1] or top > bottom or left > right:
        from .data_loader import DataLoadError
        raise DataLoadError("The selected table no longer fits this worksheet. Pick the table again.")
    block = vals[top:bottom + 1, left:right + 1]
    notes = [f"Used the table at {cell_range(spec)} of the worksheet."]
    if spec["transpose"]:
        block = block.T
        notes.append("The table ran sideways (field names down the first column), so it was flipped to one record per row.")
    hdr = spec["header_rows"]
    if hdr:
        header = _flatten_header([list(r) for r in block[:hdr]])
        if hdr > 1:
            notes.append(f"Combined {hdr} header rows into one column name per column.")
    else:
        header = [f"Column {j + 1}" for j in range(block.shape[1])]
    if spec["transpose"] and (not header[0] or header[0].startswith("Column")):
        header[0] = "Name"
    body = block[hdr:]
    if spec.get("unpivot"):
        labels_n = spec["unpivot"]
        title = _title_above(vals, top, left, right)
        raw_header = list(block[0]) if hdr == 1 else header
        period_idx = [j for j in range(labels_n, block.shape[1]) if period_label(raw_header[j]) is not None]
        dropped = [header[j] for j in range(labels_n, block.shape[1]) if j not in period_idx]
        kinds_ = [period_label(raw_header[j])[0] for j in period_idx]
        main = max(set(kinds_), key=kinds_.count) if kinds_ else None
        periods, pname = _period_values([raw_header[j] for j in period_idx], main, title)
        value_name = _value_name(title)
        id_names = [h or f"Label {j + 1}" for j, h in enumerate(header[:labels_n])]
        rows = []
        for r in body:
            if all(_blank(x) for x in r):
                continue
            ids = list(r[:labels_n])
            for j, pv in zip(period_idx, periods):
                rows.append(ids + [pv, r[j]])
        grid = [id_names + [pname, value_name]] + rows
        notes.append(f"Turned {len(period_idx)} {main or 'period'} columns into rows ({pname} and {value_name}), so each row "
                     "is one label and period.")
        if dropped:
            totals = [d for d in dropped if _TOTAL_COL_RE.match(d)]
            others = [d for d in dropped if d not in totals]
            if totals:
                notes.append(f"Left out {', '.join(totals[:3])}: it repeats the sum of the period columns.")
            if others:
                notes.append(f"Left out column(s) {', '.join(others[:4])} because they aren't periods.")
        return pd.DataFrame(grid, dtype=object), notes
    grid = [header] + [list(r) for r in body]
    return pd.DataFrame(grid, dtype=object), notes


__all__ = ["SEP", "SheetStructure", "Table", "cell_range", "col_letter", "detect", "display_sheet", "kind", "make_ref",
           "period_label", "reconstruct", "split_ref"]
