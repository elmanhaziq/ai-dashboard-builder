"""Build the working dataset: original data → approved cleaning → friendly names → interpreted profile.

The original upload is never changed; this pipeline is replayed whenever the data is loaded, so a dashboard
built by the analyst reopens with exactly the same cleaning and interpretations after a restart.
"""
from __future__ import annotations

import pandas as pd

from ..data_profiler import DatasetProfile, apply_overrides, profile_dataset
from .cleaning import apply_ops

ROLE_MAP = {"measure": "metric", "category": "dimension", "date": "date", "identifier": "identifier",
            "text": "text", "ignore": "hidden"}


def safe_renames(columns: list[str], renames: dict[str, str]) -> dict[str, str]:
    """Drop renames that would collide with another column name."""
    out, taken = {}, {c.casefold() for c in columns}
    for old, new in renames.items():
        if old not in columns or not new or new == old:
            continue
        key = new.casefold()
        if key in taken and key != old.casefold():
            continue
        out[old] = new
        taken.add(key)
    return out


def interpretation_overrides(interpretations: dict, renames: dict[str, str]) -> dict:
    overrides = {}
    for col, it in interpretations.items():
        name = renames.get(col, col)
        ov = {}
        role = ROLE_MAP.get(it.get("business_role") or "")
        if role:
            ov["role"] = role
        if it.get("aggregation") in ("sum", "mean"):
            ov["aggregation"] = it["aggregation"]
        if ov:
            overrides[name] = ov
    return overrides


def column_notes(interpretations: dict, renames: dict[str, str]) -> dict[str, str]:
    notes = {}
    for col, it in interpretations.items():
        meaning = (it.get("meaning") or "").strip()
        unit = (it.get("unit") or "").strip()
        if meaning or unit:
            text = meaning or col
            if unit:
                text += f" (unit: {unit})"
            if renames.get(col):
                text += f" — original column '{col}'"
            notes[renames.get(col, col)] = text[:300]
    return notes


def transform_frame(df: pd.DataFrame, ops: list[dict], renames: dict[str, str]) -> tuple[pd.DataFrame, list[dict], dict[str, str]]:
    cleaned, log = apply_ops(df, ops)
    renames = safe_renames(list(cleaned.columns), renames)
    if renames:
        cleaned = cleaned.rename(columns=renames)
    return cleaned, log, renames


def build_working(df: pd.DataFrame, ops: list[dict], interpretations: dict, renames: dict[str, str],
                  percent_columns: list[str] | None = None, currency_columns: list[str] | None = None
                  ) -> tuple[pd.DataFrame, DatasetProfile, list[dict], dict[str, str]]:
    cleaned, log, renames = transform_frame(df, ops, renames)
    pct = [renames.get(c, c) for c in (percent_columns or [])]
    cur = [renames.get(c, c) for c in (currency_columns or [])]
    profile = profile_dataset(cleaned, pct, cur)
    profile, _ = apply_overrides(profile, interpretation_overrides(interpretations, renames))
    for name, note in column_notes(interpretations, renames).items():
        if name in profile.columns:
            profile.columns[name].notes.insert(0, f"Means: {note}")
    return cleaned, profile, log, renames


def dashboard_transform(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Replay a dashboard's stored cleaning + renames (dashboards built by the analyst)."""
    ops = cfg.get("cleaning_ops") or []
    renames = cfg.get("renames") or {}
    if not ops and not renames:
        return df
    out, _, _ = transform_frame(df, ops, renames)
    return out
