"""Whitelisted, reversible cleaning operations.

Operations never modify the uploaded file: they are replayed on a copy every time the data is loaded.
Each operation is a small JSON object ``{"op": ..., "column": ..., "params": {...}}`` validated here, so
model-generated input can never execute code or touch anything outside these rules.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

DUPLICATE_FLAG_COLUMN = "Possible duplicate"

OPERATIONS = {
    "keep": "Leave as is",
    "fill_unknown": "Replace missing values with “Unknown”",
    "fill_median": "Fill missing numbers with the median (middle value)",
    "fill_mean": "Fill missing numbers with the average",
    "exclude_missing": "Leave rows with missing values out of the dashboard",
    "normalize_text": "Make spelling and capitalisation consistent",
    "remove_duplicates": "Remove exact duplicate rows",
    "flag_duplicates": "Mark possible duplicates (keep them, add a flag column)",
}
NEEDS_COLUMN = {"fill_unknown", "fill_median", "fill_mean", "exclude_missing", "normalize_text"}
NUMERIC_ONLY = {"fill_median", "fill_mean"}


class CleaningError(ValueError):
    pass


def validate_op(op: Any, columns: list[str], numeric: set[str]) -> dict:
    if not isinstance(op, dict) or op.get("op") not in OPERATIONS:
        raise CleaningError(f"Unsupported cleaning operation: {op!r}.")
    name = op["op"]
    col = op.get("column") or None
    params = op.get("params") if isinstance(op.get("params"), dict) else {}
    if name in NEEDS_COLUMN:
        if col not in columns:
            raise CleaningError(f"'{col}' is not a column in this dataset.")
        if name in NUMERIC_ONLY and col not in numeric:
            raise CleaningError(f"'{col}' isn't numeric, so it can't be filled with an average or median.")
    clean_params: dict = {}
    if name == "fill_unknown":
        clean_params["value"] = str(params.get("value") or "Unknown")[:40]
    if name == "normalize_text":
        mapping = params.get("mapping") or {}
        if not isinstance(mapping, dict):
            raise CleaningError("Spelling fixes must be a mapping of variant → preferred spelling.")
        clean_params["mapping"] = {str(k)[:200]: str(v)[:200] for k, v in list(mapping.items())[:500]}
    if name in ("remove_duplicates", "flag_duplicates"):
        subset = params.get("columns") or []
        if not isinstance(subset, list) or any(c not in columns for c in subset):
            raise CleaningError("Duplicate check uses unknown columns.")
        clean_params["columns"] = subset
    return {"op": name, "column": col, "params": clean_params}


def describe(op: dict) -> str:
    name, col = op["op"], op.get("column")
    if name == "keep":
        return OPERATIONS["keep"]
    base = OPERATIONS[name]
    if name == "fill_unknown" and op["params"].get("value") not in (None, "Unknown"):
        base = f"Replace missing values with “{op['params']['value']}”"
    if name in ("remove_duplicates", "flag_duplicates") and op["params"].get("columns"):
        base += f" (matching on {', '.join(op['params']['columns'])})"
    return f"{base} in '{col}'" if col else base


def _norm_key(v: str) -> str:
    return " ".join(str(v).split()).casefold()


def apply_op(df: pd.DataFrame, op: dict) -> tuple[pd.DataFrame, int]:
    """Apply one validated operation to a copy. Returns (new frame, rows affected)."""
    name, col, params = op["op"], op.get("column"), op.get("params") or {}
    if name == "keep":
        return df, 0
    out = df.copy()
    if name == "fill_unknown":
        mask = out[col].isna() | out[col].astype(str).str.strip().eq("")
        out[col] = out[col].astype(object).where(~mask, params.get("value", "Unknown"))
        return out, int(mask.sum())
    if name in ("fill_median", "fill_mean"):
        s = pd.to_numeric(out[col], errors="coerce")
        mask = s.isna()
        fill = s.median() if name == "fill_median" else s.mean()
        if pd.isna(fill):
            return out, 0
        out[col] = s.fillna(fill)
        return out, int(mask.sum())
    if name == "exclude_missing":
        mask = out[col].isna()
        return out[~mask].reset_index(drop=True), int(mask.sum())
    if name == "normalize_text":
        mapping = params.get("mapping") or {}
        s = out[col]
        def fix(v):
            if pd.isna(v):
                return v
            text = " ".join(str(v).split())
            return mapping.get(text, mapping.get(str(v), text))
        new = s.map(fix)
        changed = int(((new.astype(str) != s.astype(str)) & s.notna()).sum())
        out[col] = new
        return out, changed
    subset = params.get("columns") or None
    dup = out.duplicated(subset=subset, keep="first")
    if name == "remove_duplicates":
        return out[~dup].reset_index(drop=True), int(dup.sum())
    if name == "flag_duplicates":
        allmarks = out.duplicated(subset=subset, keep=False)
        out[DUPLICATE_FLAG_COLUMN] = np.where(allmarks, "Yes", "No")
        return out, int(allmarks.sum())
    raise CleaningError(f"Unsupported cleaning operation: {name}.")


def apply_ops(df: pd.DataFrame, ops: list[dict]) -> tuple[pd.DataFrame, list[dict]]:
    """Replay operations in order on a copy of ``df``. Returns (cleaned frame, per-operation log)."""
    out, log = df, []
    for op in ops:
        before = len(out)
        out, affected = apply_op(out, op)
        log.append({"op": op["op"], "column": op.get("column"), "rows_affected": affected,
                    "rows_before": before, "rows_after": len(out), "description": describe(op)})
    return out, log


def preview(df: pd.DataFrame, op: dict, n: int = 3) -> dict:
    """What an operation would change, without changing anything."""
    new, affected = apply_op(df, op)
    sample: list[dict] = []
    col = op.get("column")
    if col and col in df.columns and op["op"] in ("fill_unknown", "fill_median", "fill_mean", "normalize_text"):
        changed = df[col].astype(str) != new[col].astype(str) if len(new) == len(df) else pd.Series(False, index=df.index)
        for i in list(df.index[changed.values])[:n]:
            sample.append({"before": None if pd.isna(df.at[i, col]) else str(df.at[i, col]), "after": str(new.at[i, col])})
    return {"description": describe(op), "rows_affected": affected, "rows_before": len(df), "rows_after": len(new),
            "examples": sample}
