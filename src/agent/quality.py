"""Detect data-quality issues and turn them into plain-language cleaning proposals.

Every proposal explains what was found, why it matters, the recommended treatment, and alternatives.
Consequential changes require the user's approval; informational findings never change data.
Rules: missing numbers are never filled with zero, and records are never removed on anomaly evidence alone.
"""
from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd

from ..data_profiler import DATE, DIMENSION, IDENTIFIER, METRIC, DatasetProfile
from .cleaning import describe


def _issue_id(*parts) -> str:
    return hashlib.sha1("|".join(map(str, parts)).encode("utf-8")).hexdigest()[:10]


def _op(name: str, column: str | None = None, **params) -> dict:
    return {"op": name, "column": column, "params": params}


def _choice(op: dict) -> dict:
    return {"op": op, "label": describe(op)}


def _skewed(s: pd.Series) -> bool:
    v = pd.to_numeric(s, errors="coerce").dropna()
    if len(v) < 10 or v.std() == 0:
        return False
    return abs(float(v.skew())) > 1.0


def detect_issues(df: pd.DataFrame, profile: DatasetProfile, load_warnings: list[str], removed_rows: int = 0,
                  interpretations: dict | None = None) -> list[dict]:
    interpretations = interpretations or {}
    issues: list[dict] = []
    n = len(df)

    def add(kind, column, severity, title, detail, why, recommended, alternatives, requires_approval=True, extra=None):
        issues.append({
            "id": _issue_id(kind, column, detail), "kind": kind, "column": column, "severity": severity,
            "title": title, "detail": detail, "why": why,
            "recommended": _choice(recommended), "alternatives": [_choice(a) for a in alternatives],
            "requires_approval": requires_approval, "status": "pending" if requires_approval else "info",
            "decision": None, **(extra or {}),
        })

    # Exact duplicate rows.
    dup = int(df.duplicated().sum()) if n else 0
    if dup:
        add("duplicates", None, "high", f"{dup:,} duplicate rows",
            f"I found {dup:,} row(s) that are exact copies of another row.",
            "Exact copies are counted twice in totals, which can inflate sales or counts.",
            _op("remove_duplicates"), [_op("flag_duplicates"), _op("keep")])

    # Possible duplicate transactions: an identifier repeated across different rows.
    for col in profile.identifier_columns:
        cp = profile.columns[col]
        if cp.unique and cp.unique < (n - cp.missing) and cp.unique_pct > 80:
            repeats = int(df[col].duplicated(keep=False).sum() - df.duplicated(keep=False).sum())
            if repeats > 0:
                add("possible_duplicates", col, "medium", f"Repeated {col} values",
                    f"{repeats:,} row(s) share a '{col}' with another row even though other details differ.",
                    "These may be the same transaction entered twice, or legitimate multi-line records. I can't tell automatically.",
                    _op("flag_duplicates", columns=[col]), [_op("keep")])
            break

    for col, cp in profile.columns.items():
        if col not in df.columns:
            continue
        role = cp.role
        it = interpretations.get(col, {})
        if it.get("business_role") == "ignore":
            continue
        # Missing values.
        if cp.missing and cp.missing < n:
            pct = cp.missing / n * 100
            sev = "high" if pct >= 20 else "medium" if pct >= 5 else "low"
            if role == METRIC:
                rec = _op("keep", col)
                alts = [_op("fill_median", col) if _skewed(df[col]) else _op("fill_mean", col),
                        _op("exclude_missing", col)]
                why = ("Missing numbers are left out of totals and averages. Filling them with an estimate changes the "
                       "results, so I recommend keeping them as missing. I will never replace them with zero.")
            elif role == DIMENSION:
                rec, alts = _op("fill_unknown", col), [_op("keep", col), _op("exclude_missing", col)]
                why = "Blank categories show up as '(blank)' in charts. Labelling them 'Unknown' makes charts easier to read."
            elif role == DATE:
                rec, alts = _op("keep", col), [_op("exclude_missing", col)]
                why = "Rows without a date can't appear on trend charts but still count in totals."
            else:
                continue
            add("missing", col, sev, f"Missing values in {col}",
                f"I found {cp.missing:,} missing value(s) in '{col}' ({pct:.1f}% of rows).", why, rec, alts)

        # Inconsistent category labels (case / spacing variants).
        if role == DIMENSION and 1 < cp.unique <= 500:
            s = df[col].dropna().astype(str)
            keys = s.map(lambda v: " ".join(v.split()).casefold())
            groups = s.groupby(keys).agg(lambda v: v.value_counts().to_dict())
            mapping, examples = {}, []
            for _, counts in groups.items():
                if len(counts) > 1:
                    preferred = max(counts.items(), key=lambda kv: (kv[1], kv[0] != kv[0].upper()))[0].strip()
                    preferred = " ".join(preferred.split())
                    for variant in counts:
                        mapping[" ".join(variant.split())] = preferred
                        mapping[variant] = preferred
                    examples.append(", ".join(sorted(counts)))
            if examples:
                shown = "; ".join(examples[:3])
                add("inconsistent_labels", col, "high", f"Inconsistent labels in {col}",
                    f"I found the same value written in different ways in '{col}': {shown}.",
                    "Charts would show these as separate groups and split their totals.",
                    _op("normalize_text", col, mapping=mapping), [_op("keep", col)])

        # Outliers: informational only — never removed automatically.
        if role == METRIC and cp.stats.get("outliers"):
            add("outliers", col, "low", f"Unusual values in {col}",
                f"{cp.stats['outliers']:,} value(s) in '{col}' are unusually high or low compared with the rest.",
                "They may be genuine (a big order) or errors. I keep them; you can check them in the Data Explorer.",
                _op("keep", col), [], requires_approval=False)

    # Findings from loading the file (already handled, explained for transparency).
    for w in load_warnings:
        low = w.lower()
        if "total" in low and "excluded" in low:
            add("subtotals", None, "medium", "Total and subtotal rows",
                w, "Keeping them would count those amounts twice.", _op("keep"), [], requires_approval=False)
        elif "day/month/year" in low or "ambiguous" in low or "could not be read as dates" in low:
            add("dates", None, "medium", "Date formats", w,
                "Dates written in different styles were converted to one standard format so trends line up correctly.",
                _op("keep"), [], requires_approval=False)
        elif "could not be read as numbers" in low:
            add("non_numeric", None, "medium", "Text inside number columns", w,
                "Those cells are treated as missing so they don't break totals.", _op("keep"), [], requires_approval=False)
        elif "repeated header" in low or "empty row" in low or "empty column" in low or "skipped" in low:
            add("layout", None, "low", "Spreadsheet layout", w, "These rows weren't data, so they were left out.",
                _op("keep"), [], requires_approval=False)
    if any(cp.is_currency for cp in profile.columns.values()):
        cols = [c for c, cp in profile.columns.items() if cp.is_currency][:4]
        add("currency", None, "low", "Currency values", f"Currency symbols and thousand separators were removed from {', '.join(cols)} so they can be added up.",
            "Money amounts stored as text can't be totalled.", _op("keep"), [], requires_approval=False)
    order = {"high": 0, "medium": 1, "low": 2}
    issues.sort(key=lambda i: (not i["requires_approval"], order[i["severity"]]))
    return issues


def issue_cells(df: pd.DataFrame, issues: list[dict]) -> list[dict]:
    """Where each issue is in the data, for highlighting: rows (bool Series) and columns (None = whole row).

    Only issues that point at specific cells are returned (not file-level notes such as 'Total rows')."""
    out = []
    for issue in issues:
        kind, col = issue["kind"], issue.get("column")
        if col is not None and col not in df.columns:
            continue
        rows, cols = None, [col] if col else None
        if kind == "missing":
            rows = df[col].isna()
        elif kind == "duplicates":
            rows = df.duplicated(keep="first")
        elif kind == "possible_duplicates":
            rows = df[col].duplicated(keep=False) & df[col].notna() & ~df.duplicated(keep=False)
        elif kind == "inconsistent_labels":
            mapping = (issue.get("recommended") or {}).get("op", {}).get("params", {}).get("mapping") or {}
            s = df[col]
            rows = s.notna() & s.map(lambda v: isinstance(v, str) and v in mapping and mapping[v] != v)
            rows = rows.astype(bool)
        elif kind == "outliers":
            v = pd.to_numeric(df[col], errors="coerce")
            q1, q3 = v.quantile(.25), v.quantile(.75)
            iqr = q3 - q1
            rows = ((v < q1 - 1.5 * iqr) | (v > q3 + 1.5 * iqr)) if iqr > 0 else pd.Series(False, index=df.index)
        if rows is None or not bool(rows.any()):
            continue
        out.append({"id": issue["id"], "kind": kind, "title": issue["title"], "columns": cols, "rows": rows.fillna(False),
                    "count": int(rows.sum()), "status": issue.get("status")})
    return out


def chosen_ops(issues: list[dict]) -> list[dict]:
    """Operations the user approved, in a safe order (labels → fills → exclusions → duplicates)."""
    rank = {"normalize_text": 0, "fill_unknown": 1, "fill_median": 1, "fill_mean": 1, "exclude_missing": 2,
            "flag_duplicates": 3, "remove_duplicates": 4}
    ops = []
    for issue in issues:
        if issue.get("status") == "approved" and issue.get("decision") and issue["decision"]["op"] != "keep":
            ops.append(issue["decision"])
    return sorted(ops, key=lambda o: rank.get(o["op"], 9))


__all__ = ["detect_issues", "chosen_ops", "issue_cells", "np"]
