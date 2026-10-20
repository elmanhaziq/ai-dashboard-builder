"""Reading uploaded spreadsheets into clean pandas DataFrames.

The loader is deliberately conservative: it detects the header row, normalises
column names, converts obviously-typed columns, and reports every assumption it
makes as a human-readable warning rather than silently discarding data.
"""
from __future__ import annotations

import csv
import io
import re
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd

from . import config

ALLOWED_EXTENSIONS = {".xlsx", ".xls", ".csv"}
CSV_SHEET_NAME = "CSV data"

Source = Union[bytes, str, Path]

_TOTAL_RE = re.compile(r"^\s*(grand\s+|sub\s*-?\s*)?totals?\b\s*:?", re.IGNORECASE)
_DATE_LIKE_RE = re.compile(
    r"^\s*(\d{1,4}[-/.]\d{1,2}[-/.]\d{1,4}(?:[ T]\d{1,2}:\d{2}(?::\d{2})?(?:\.\d+)?)?"
    r"|\d{1,2}[ -](jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*[ -,]*\d{2,4}"
    r"|(jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\.?[ -]\d{1,2},?[ -]\d{2,4}"
    r"|(jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*[ -]\d{4}"
    r"|\d{4}-\d{2})\s*$",
    re.IGNORECASE,
)
_DMY_RE = re.compile(r"^\s*(\d{1,2})[/.-](\d{1,2})[/.-](\d{2,4})")
_CURRENCY_RE = re.compile(r"[$€£¥₹]|\b(?:USD|EUR|GBP|MYR|RM|INR|JPY|AUD|CAD|SGD)\b", re.IGNORECASE)


class DataLoadError(Exception):
    """Raised with a user-facing message when a file cannot be read."""


@dataclass
class LoadResult:
    df: pd.DataFrame
    sheet: str
    header_row: int  # 0-based row in the raw sheet used as header; -1 when generated
    warnings: list[str] = field(default_factory=list)
    raw_shape: tuple[int, int] = (0, 0)
    conversions: dict[str, str] = field(default_factory=dict)
    percent_columns: list[str] = field(default_factory=list)
    currency_columns: list[str] = field(default_factory=list)
    removed_rows: pd.DataFrame = field(default_factory=pd.DataFrame)


# --------------------------------------------------------------------------- validation

def validate_upload(filename: str, size: int, max_bytes: int | None = None) -> str:
    """Validate name and size of an upload. Returns the lowercase extension."""
    max_bytes = max_bytes or config.MAX_UPLOAD_BYTES
    if not filename:
        raise DataLoadError("The uploaded file has no name.")
    ext = Path(filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise DataLoadError(
            f"Unsupported file type '{ext or 'none'}'. Please upload an .xlsx, .xls or .csv file."
        )
    if size <= 0:
        raise DataLoadError("The uploaded file is empty (0 bytes).")
    if size > max_bytes:
        raise DataLoadError(
            f"The file is {size / 1_048_576:.1f} MB, which exceeds the {max_bytes / 1_048_576:.0f} MB limit. "
            "Try removing unused sheets or columns, or export the relevant sheet as CSV."
        )
    return ext


def _as_bytes(source: Source) -> bytes:
    if isinstance(source, (bytes, bytearray)):
        return bytes(source)
    try:
        return Path(source).read_bytes()
    except OSError as exc:
        raise DataLoadError(f"Could not read the stored file: {exc.strerror or exc}.") from exc


def _detect_format(data: bytes, ext: str) -> str:
    """Determine the real format from magic bytes (an '.xls' can actually be xlsx, etc.)."""
    if data[:4] == b"PK\x03\x04":
        return ".xlsx"
    if data[:8] == b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1":
        return ".xls"
    if ext in (".xlsx", ".xls"):
        head = data[:2048].lstrip().lower()
        if head.startswith(b"<") :
            raise DataLoadError(
                "This file has an Excel extension but contains HTML/XML, not a workbook. "
                "Open it in Excel and use 'Save As' → Excel Workbook (.xlsx)."
            )
        if b"\x00" not in data[:4096]:
            return ".csv"  # text file mislabeled as Excel; treat as delimited text
        raise DataLoadError(
            "The workbook appears to be corrupted or is not a real Excel file. "
            "Try re-saving it from Excel as .xlsx."
        )
    if b"\x00" in data[:4096]:
        raise DataLoadError("This CSV file contains binary content and cannot be read as text.")
    return ".csv"


# --------------------------------------------------------------------------- sheets

def list_sheets(source: Source, ext: str) -> list[str]:
    data = _as_bytes(source)
    fmt = _detect_format(data, ext.lower())
    if fmt == ".csv":
        return [CSV_SHEET_NAME]
    try:
        engine = "openpyxl" if fmt == ".xlsx" else "xlrd"
        with pd.ExcelFile(io.BytesIO(data), engine=engine) as xl:
            sheets = [str(s) for s in xl.sheet_names]
    except Exception as exc:  # noqa: BLE001 - library raises many types for corrupt files
        raise DataLoadError(_excel_error_message(exc)) from exc
    if not sheets:
        raise DataLoadError("The workbook does not contain any worksheets.")
    return sheets


def sheet_dimensions(source: Source, ext: str) -> dict[str, tuple[int, int]]:
    """Approximate (rows, columns) per sheet without loading cell data. Best effort."""
    data = _as_bytes(source)
    fmt = _detect_format(data, ext.lower())
    out: dict[str, tuple[int, int]] = {}
    try:
        if fmt == ".xlsx":
            import openpyxl
            wb = openpyxl.load_workbook(io.BytesIO(data), read_only=True, data_only=True)
            try:
                for ws in wb.worksheets:
                    out[str(ws.title)] = (int(ws.max_row or 0), int(ws.max_column or 0))
            finally:
                wb.close()
        elif fmt == ".xls":
            import xlrd
            book = xlrd.open_workbook(file_contents=data, on_demand=True)
            for i in range(book.nsheets):
                sh = book.sheet_by_index(i)
                out[str(sh.name)] = (sh.nrows, sh.ncols)
    except Exception:  # noqa: BLE001 - dimensions are only a hint
        return {}
    return out


def choose_default_sheet(sheet_sizes: dict[str, tuple[int, int]]) -> str | None:
    """Pick the sheet with the most non-empty cells (rows × columns)."""
    if not sheet_sizes:
        return None
    return max(sheet_sizes, key=lambda s: (sheet_sizes[s][0] * max(sheet_sizes[s][1], 1), -list(sheet_sizes).index(s)))


def _excel_error_message(exc: Exception) -> str:
    text = str(exc).lower()
    if "encrypt" in text or "password" in text:
        return "The workbook is password-protected. Remove the password in Excel and upload it again."
    if "zip" in text or "corrupt" in text or "not a" in text or "unsupported format" in text:
        return "The workbook appears to be corrupted or is not a valid Excel file. Try re-saving it from Excel."
    return f"The workbook could not be opened ({type(exc).__name__}). Try re-saving it as .xlsx or CSV."


# --------------------------------------------------------------------------- raw reading

def _decode_text(data: bytes) -> str:
    for enc in ("utf-8-sig", "cp1252", "latin-1"):
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            continue
    raise DataLoadError("The CSV file's text encoding could not be detected. Save it as UTF-8 and retry.")


def _read_csv_raw(data: bytes, warnings: list[str]) -> pd.DataFrame:
    text = _decode_text(data)
    if not text.strip():
        raise DataLoadError("The CSV file contains no data.")
    sample = text[:50_000]
    try:
        delimiter = csv.Sniffer().sniff(sample, delimiters=",;\t|").delimiter
    except csv.Error:
        delimiter = ","
    try:
        return pd.read_csv(
            io.StringIO(text), sep=delimiter, header=None, dtype=str,
            skip_blank_lines=False, keep_default_na=True,
        )
    except pd.errors.ParserError:
        rows = list(csv.reader(io.StringIO(text), delimiter=delimiter))
        width = max((len(r) for r in rows), default=0)
        ragged = sum(1 for r in rows if 0 < len(r) < width)
        if ragged:
            warnings.append(
                f"{ragged:,} rows had fewer fields than others; missing fields were left blank."
            )
        return pd.DataFrame([r + [None] * (width - len(r)) for r in rows], dtype=object)
    except pd.errors.EmptyDataError as exc:
        raise DataLoadError("The CSV file contains no data.") from exc


def read_raw(source: Source, ext: str, sheet: str | None = None) -> tuple[pd.DataFrame, str, list[str]]:
    """Read a sheet with no header interpretation. Returns (raw_df, sheet_name, warnings).

    ``sheet`` may be a table reference made by ``structure.make_ref``; the whole worksheet is returned either way."""
    from .structure import split_ref

    sheet = split_ref(sheet)[0]
    data = _as_bytes(source)
    fmt = _detect_format(data, ext.lower())
    warnings: list[str] = []
    if fmt != ext.lower():
        warnings.append(f"The file extension is '{ext}', but the content is {fmt[1:].upper()}; it was read as {fmt[1:].upper()}.")
    if fmt == ".csv":
        return _read_csv_raw(data, warnings), CSV_SHEET_NAME, warnings
    engine = "openpyxl" if fmt == ".xlsx" else "xlrd"
    try:
        with pd.ExcelFile(io.BytesIO(data), engine=engine) as xl:
            names = [str(s) for s in xl.sheet_names]
            if not names:
                raise DataLoadError("The workbook does not contain any worksheets.")
            target = sheet if sheet is not None else names[0]
            if target not in names:
                raise DataLoadError(f"Worksheet '{target}' was not found. Available sheets: {', '.join(names)}.")
            raw = xl.parse(target, header=None, dtype=object)
    except DataLoadError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise DataLoadError(_excel_error_message(exc)) from exc
    return raw, target, warnings


# --------------------------------------------------------------------------- cleaning helpers

def _is_blank(v) -> bool:
    if v is None:
        return True
    if isinstance(v, float) and np.isnan(v):
        return True
    if v is pd.NaT:
        return True
    if isinstance(v, str) and not v.strip():
        return True
    try:
        return bool(pd.isna(v))
    except (TypeError, ValueError):
        return False


def _looks_numeric(v) -> bool:
    if isinstance(v, (int, float, np.number)) and not isinstance(v, bool):
        return True
    if isinstance(v, str):
        try:
            float(v.replace(",", "").strip().rstrip("%").lstrip("$€£"))
            return True
        except ValueError:
            return False
    return False


def detect_header_row(raw: pd.DataFrame, scan_rows: int = 30) -> int:
    """Return the 0-based index of the most likely header row, or -1 if none is found."""
    if raw.empty:
        return -1
    head = raw.head(scan_rows)
    counts = [sum(not _is_blank(v) for v in row) for row in head.itertuples(index=False)]
    max_count = max(counts) if counts else 0
    if max_count == 0:
        return -1
    for i, row in enumerate(head.itertuples(index=False)):
        values = [v for v in row if not _is_blank(v)]
        if len(values) < max(1, 0.6 * max_count) or (max_count >= 2 and len(values) < 2):
            continue
        texty = [v for v in values if isinstance(v, str) and not _looks_numeric(v)]
        if len(texty) / len(values) >= 0.6 and len({str(v).strip().lower() for v in values}) >= 0.8 * len(values):
            return i
        # First substantial row is numeric/data-like: no header present.
        return -1
    return -1


def _header_label(v, idx: int) -> str:
    if _is_blank(v):
        return f"Column {idx + 1}"
    if isinstance(v, (datetime, date, pd.Timestamp)):
        return pd.Timestamp(v).strftime("%Y-%m-%d")
    if isinstance(v, float) and v.is_integer():
        return str(int(v))
    return re.sub(r"\s+", " ", str(v)).strip() or f"Column {idx + 1}"


def make_unique_names(names: list[str]) -> tuple[list[str], list[str]]:
    """De-duplicate names case-insensitively. Returns (names, list of renamed originals)."""
    seen: dict[str, int] = {}
    out, renamed = [], []
    used = set()
    for name in names:
        key = name.lower()
        if key in used:
            n = seen.get(key, 1)
            candidate = f"{name} ({n + 1})"
            while candidate.lower() in used:
                n += 1
                candidate = f"{name} ({n + 1})"
            seen[key] = n + 1
            renamed.append(name)
            out.append(candidate)
            used.add(candidate.lower())
        else:
            seen[key] = 1
            out.append(name)
            used.add(key)
    return out, renamed


def _clean_numeric_strings(s: pd.Series) -> tuple[pd.Series, bool, bool]:
    """Strip currency/thousand separators. Returns (cleaned, had_percent, had_currency)."""
    text = s.astype(str).str.strip()
    had_percent = bool(text.str.endswith("%").mean() > 0.5)
    had_currency = bool(text.str.contains(_CURRENCY_RE).mean() > 0.5)
    neg = text.str.match(r"^\(.*\)$")
    cleaned = (
        text.str.replace(_CURRENCY_RE, "", regex=True)
        .str.replace(r"[,\s%]", "", regex=True)
        .str.replace(r"^\((.*)\)$", r"\1", regex=True)
    )
    cleaned = cleaned.where(~neg, "-" + cleaned)
    return cleaned, had_percent, had_currency


def _infer_dayfirst(values: pd.Series) -> bool | None:
    """True/False when the d/m vs m/d order can be decided, None when ambiguous."""
    m = values.astype(str).str.extract(_DMY_RE)
    if m.empty or m[0].isna().all():
        return False
    a = pd.to_numeric(m[0], errors="coerce")
    b = pd.to_numeric(m[1], errors="coerce")
    if (a > 12).any():
        return True
    if (b > 12).any():
        return False
    return None


def _coerce_column(s: pd.Series, name: str, result: LoadResult) -> pd.Series:
    nonnull = s[~s.map(_is_blank)]
    if nonnull.empty:
        return s.where(~s.map(_is_blank), np.nan)
    s = s.where(~s.map(_is_blank), np.nan)
    n = len(nonnull)

    is_dt = nonnull.map(lambda v: isinstance(v, (datetime, date, pd.Timestamp)) and not isinstance(v, bool))
    is_num = nonnull.map(lambda v: isinstance(v, (int, float, np.number)) and not isinstance(v, bool))
    is_bool = nonnull.map(lambda v: isinstance(v, (bool, np.bool_)))

    if is_bool.all():
        result.conversions[name] = "boolean"
        return s.map(lambda v: v if _is_blank(v) else ("Yes" if bool(v) else "No"))

    if is_dt.mean() >= 0.9:
        out = pd.to_datetime(s.where(s.map(lambda v: isinstance(v, (datetime, date, pd.Timestamp))), None), errors="coerce")
        bad = int(n - out.notna().sum())
        if bad:
            result.warnings.append(f"'{name}': {bad:,} non-date value(s) were treated as missing.")
        result.conversions[name] = "datetime"
        return out

    strings = nonnull.astype(str).str.strip()
    # Preserve codes with leading zeros (zip codes, account numbers) as text.
    leading_zero = strings.str.match(r"^0\d+$").mean() > 0.1

    if not leading_zero:
        cleaned, had_pct, had_cur = _clean_numeric_strings(nonnull[~is_num])
        parsed = pd.Series(np.nan, index=nonnull.index, dtype=float)
        parsed[is_num] = pd.to_numeric(nonnull[is_num], errors="coerce")
        if (~is_num).any():
            parsed[~is_num] = pd.to_numeric(cleaned, errors="coerce")
        ok = parsed.notna().mean()
        if ok >= 0.9:
            out = pd.Series(np.nan, index=s.index, dtype=float)
            out[parsed.index] = parsed
            bad = int(parsed.isna().sum())
            if bad:
                examples = ", ".join(repr(v) for v in nonnull[parsed.isna()].astype(str).unique()[:3])
                result.warnings.append(
                    f"'{name}': {bad:,} value(s) could not be read as numbers (e.g. {examples}) and are treated as missing."
                )
            if had_pct:
                result.percent_columns.append(name)
            if had_cur:
                result.currency_columns.append(name)
            if (~is_num).any() and (is_num.any() or had_pct or had_cur):
                result.conversions[name] = "numeric (cleaned text)"
            else:
                result.conversions[name] = "numeric"
            if out.notna().all() and np.allclose(out, out.round()) and out.abs().max() < 2**53:
                return out.astype("int64")
            return out

    # Date strings
    if not is_num.any() and strings.str.match(_DATE_LIKE_RE).mean() >= 0.8:
        dayfirst = _infer_dayfirst(strings)
        # ISO dates (2024-05-06) are always year-month-day; only the other styles follow the day/month inference.
        # Parsing them together with dayfirst=True would silently swap day and month in the ISO values.
        iso = strings.str.match(r"^\s*\d{4}[-/.]\d{1,2}[-/.]\d{1,2}")
        parsed = pd.Series(pd.NaT, index=strings.index, dtype="datetime64[ns]")
        for mask, kw in ((iso, {"yearfirst": True, "dayfirst": False}), (~iso, {"dayfirst": bool(dayfirst)})):
            if mask.any():
                try:
                    part = pd.to_datetime(strings[mask], errors="coerce", format="mixed", **kw)
                except (ValueError, TypeError):
                    part = pd.to_datetime(strings[mask], errors="coerce", **kw)
                parsed[mask] = part
        if parsed.notna().mean() >= 0.9:
            out = pd.Series(pd.NaT, index=s.index, dtype="datetime64[ns]")
            out[parsed.index] = parsed
            bad = int(parsed.isna().sum())
            if bad:
                result.warnings.append(f"'{name}': {bad:,} value(s) could not be read as dates and are treated as missing.")
            if dayfirst is None:
                result.warnings.append(
                    f"'{name}': dates such as 03/04/2024 are ambiguous; they were read as month/day/year."
                )
            elif dayfirst:
                result.warnings.append(f"'{name}': dates were read as day/month/year.")
            result.conversions[name] = "datetime (parsed text)"
            return out

    # Mixed text: normalise to strings so grouping behaves consistently.
    if is_num.any() and not is_num.all():
        result.warnings.append(
            f"'{name}' mixes numbers and text ({int(is_num.sum()):,} numeric, {int((~is_num).sum()):,} text); kept as text."
        )
    return s.map(lambda v: v if _is_blank(v) else (_header_label(v, 0) if isinstance(v, float) else str(v).strip()))


# --------------------------------------------------------------------------- main entry point

def load_dataset(source: Source, ext: str, sheet: str | None = None, header_row: int | None = None) -> LoadResult:
    """Load one sheet into a cleaned DataFrame with warnings describing every assumption.

    When ``sheet`` is a table reference (see ``structure``), that table is cut out, flipped, given a one-row header and
    unpivoted as recorded in the reference first; the cleaning below then works on it exactly as on a plain sheet."""
    from .structure import reconstruct, split_ref

    raw, sheet_name, warnings = read_raw(source, ext, sheet)
    result = LoadResult(df=pd.DataFrame(), sheet=sheet_name, header_row=-1, warnings=warnings, raw_shape=raw.shape)

    if raw.empty or raw.map(_is_blank).all().all():
        raise DataLoadError(f"The worksheet '{sheet_name}' is empty. Choose another sheet or check the file.")

    _, spec = split_ref(sheet)
    if spec:
        raw, notes = reconstruct(raw, spec)
        result.warnings.extend(notes)
        result.sheet = sheet
        header_row = 0 if spec["header_rows"] else -1

    h = detect_header_row(raw) if header_row is None else header_row
    if h >= len(raw):
        raise DataLoadError("The selected header row is beyond the end of the sheet.")
    result.header_row = h

    if h >= 0:
        header_values = list(raw.iloc[h])
        body = raw.iloc[h + 1:].copy()
        skipped = sum(1 for i in range(h) if not raw.iloc[i].map(_is_blank).all())
        if skipped:
            result.warnings.append(f"Skipped {skipped} row(s) above the header row (titles or notes).")
    else:
        header_values = [None] * raw.shape[1]
        body = raw.copy()
        result.warnings.append("No header row was detected; columns were named 'Column 1', 'Column 2', …")

    names = [_header_label(v, i) for i, v in enumerate(header_values)]
    blank_header_cols = [i for i, v in enumerate(header_values) if _is_blank(v)]

    blank_mask = body.map(_is_blank)
    # Drop columns that are completely empty (no header, no data).
    empty_cols = [i for i in range(body.shape[1]) if blank_mask.iloc[:, i].all() and i in blank_header_cols]
    keep = [i for i in range(body.shape[1]) if i not in empty_cols]
    if empty_cols:
        body = body.iloc[:, keep]
        blank_mask = blank_mask.iloc[:, keep]
        result.warnings.append(f"Removed {len(empty_cols)} completely empty column(s).")
    filled_no_header = [names[i] for i in keep if i in blank_header_cols and h >= 0]
    names = [names[i] for i in keep]
    if filled_no_header:
        result.warnings.append(f"{len(filled_no_header)} column(s) had no header and were named automatically: {', '.join(filled_no_header[:5])}.")

    names, renamed = make_unique_names(names)
    if renamed:
        result.warnings.append(f"Duplicate column names were made unique: {', '.join(sorted(set(renamed))[:5])}.")
    body.columns = names

    empty_rows = blank_mask.all(axis=1)
    if empty_rows.any():
        result.warnings.append(f"Removed {int(empty_rows.sum()):,} empty row(s).")
        body = body[~empty_rows.values]

    # Repeated header rows (common when sheets are pasted together).
    if h >= 0 and len(body):
        header_norm = [str(v).strip().lower() for v in header_values if not _is_blank(v)]
        if header_norm:
            def _is_header(row) -> bool:
                vals = [str(v).strip().lower() for v in row if not _is_blank(v)]
                return len(vals) >= max(2, len(header_norm) // 2) and set(vals) <= set(header_norm)
            rep = body.apply(_is_header, axis=1)
            if rep.any():
                result.warnings.append(f"Removed {int(rep.sum())} repeated header row(s) found inside the data.")
                body = body[~rep.values]

    # Total / subtotal rows would double-count aggregates.
    if len(body):
        # A row is a total row when a cell is a "Total"/"Subtotal" label and either that label is in one of the
        # first columns, or most of the row's other cells (IDs, dates, names) are blank — typical of report layouts.
        label_cols = list(body.columns[: min(3, body.shape[1])])
        blank_share = body.map(_is_blank).mean(axis=1)
        is_total = pd.Series(False, index=body.index)
        for c in body.columns:
            is_label = body[c].map(lambda v: isinstance(v, str) and bool(_TOTAL_RE.match(v)) and len(v.strip()) <= 30)
            is_total |= is_label & ((c in label_cols) | (blank_share >= 0.4))
        if is_total.any() and is_total.sum() <= max(50, 0.05 * len(body)):
            result.removed_rows = body[is_total].copy()
            result.warnings.append(
                f"Excluded {int(is_total.sum())} row(s) labelled 'Total'/'Subtotal' so they are not double-counted. "
                "They are listed in the Data Explorer."
            )
            body = body[~is_total.values]

    if body.empty:
        raise DataLoadError(f"The worksheet '{sheet_name}' has a header but no data rows.")

    if len(body) > config.MAX_ROWS:
        result.warnings.append(
            f"The sheet has {len(body):,} rows; only the first {config.MAX_ROWS:,} were loaded "
            "(raise DASHBOARD_MAX_ROWS in .env to change this)."
        )
        body = body.iloc[: config.MAX_ROWS]

    body = body.reset_index(drop=True)
    converted = {c: _coerce_column(body[c], c, result) for c in body.columns}
    result.df = pd.DataFrame(converted, columns=body.columns)
    return result
