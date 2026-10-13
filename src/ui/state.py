"""Session state, caching and dataset/dashboard lifecycle helpers for the UI."""
from __future__ import annotations

import json
import time
import uuid

import pandas as pd
import streamlit as st

from .. import config
from ..chart_generator import ChartConfigError
from ..dashboard_service import build_default_dashboard, validate_dashboard_config
from ..data_loader import DataLoadError, LoadResult, load_dataset, read_raw
from ..data_profiler import DatasetProfile, apply_overrides, profile_dataset
from ..storage import Storage, StorageError, session_storage, sweep_sessions


@st.cache_resource
def _local_storage() -> Storage:
    return Storage()


_last_sweep = {"t": 0.0}


def get_storage() -> Storage:
    """Local mode: the shared project folders. Online mode: a private temporary folder per browser session."""
    if not config.ONLINE:
        return _local_storage()
    sid = st.session_state.get("_session_id")
    if not sid:
        sid = st.session_state["_session_id"] = uuid.uuid4().hex
    if time.time() - _last_sweep["t"] > 600:  # remove other visitors' expired sessions, at most every 10 minutes
        _last_sweep["t"] = time.time()
        sweep_sessions(config.SESSION_ROOT, config.SESSION_TTL_MINUTES)
    return session_storage(config.SESSION_ROOT, sid)


# Cached loads are keyed by the file's full path (which includes the private session folder online) and its hash,
# so a cached result can never be served to a different visitor.
@st.cache_data(show_spinner=False, max_entries=6)
def _load_file(path: str, ext: str, sheet: str | None, header_row: int | None, sha: str) -> tuple[LoadResult, DatasetProfile]:
    result = load_dataset(path, ext, sheet, header_row)
    profile = profile_dataset(result.df, result.percent_columns, result.currency_columns)
    return result, profile


def load_cached(dataset_id: str, sheet: str | None, header_row: int | None, sha: str) -> tuple[LoadResult, DatasetProfile]:
    """Load + profile a stored dataset."""
    storage = get_storage()
    meta = storage.get_dataset(dataset_id)
    return _load_file(str(storage.dataset_file(dataset_id)), meta["ext"], sheet, header_row, sha)


def raw_preview(dataset_id: str, sheet: str | None, sha: str, rows: int = 200) -> pd.DataFrame:
    storage = get_storage()
    meta = storage.get_dataset(dataset_id)
    return _raw_preview(str(storage.dataset_file(dataset_id)), meta["ext"], sheet, sha, rows)


@st.cache_data(show_spinner=False, max_entries=4)
def _raw_preview(path: str, ext: str, sheet: str | None, sha: str, rows: int) -> pd.DataFrame:
    raw, _, _ = read_raw(path, ext, sheet)
    preview = raw.head(rows).astype(object)
    # Show cells exactly as text so mixed-type columns display faithfully.
    preview = preview.map(lambda v: "" if v is None or (isinstance(v, float) and v != v) else str(v))
    preview.columns = [f"Col {i + 1}" for i in range(preview.shape[1])]
    preview.index = range(1, len(preview) + 1)
    return preview


def load_for(dataset_id: str, sheet: str | None, header_row: int | None) -> tuple[dict, LoadResult, DatasetProfile]:
    meta = get_storage().get_dataset(dataset_id)
    result, profile = load_cached(dataset_id, sheet, header_row, meta["sha256"])
    return meta, result, profile


def load_dashboard_data(cfg: dict) -> tuple[dict, LoadResult, DatasetProfile, list[str]]:
    """Dataset for a dashboard, with the user's column-role corrections applied to the profile."""
    meta, result, profile = load_for(cfg["dataset_id"], cfg.get("sheet"), cfg.get("header_row"))
    profile, warnings = apply_overrides(profile, cfg.get("column_overrides"))
    return meta, result, profile, warnings


# --------------------------------------------------------------------------- dashboard session

def snapshot(cfg: dict) -> str:
    return json.dumps(cfg, sort_keys=True, default=str)


def set_dashboard(cfg: dict, saved: bool) -> None:
    st.session_state["dashboard"] = cfg
    st.session_state["saved_snapshot"] = snapshot(cfg) if saved else ""
    st.session_state.pop("staging", None)
    st.session_state["qa_history"] = []
    st.session_state["filters_initialised_for"] = None


def current_dashboard() -> dict | None:
    return st.session_state.get("dashboard")


def is_dirty() -> bool:
    cfg = current_dashboard()
    return bool(cfg) and snapshot(cfg) != st.session_state.get("saved_snapshot", "")


def open_dashboard(dashboard_id: str) -> list[str]:
    """Load a saved dashboard and its dataset into the session. Returns warnings."""
    storage = get_storage()
    cfg = storage.load_dashboard(dashboard_id)
    _, _, profile, override_warnings = load_dashboard_data(cfg)
    cfg, warnings = validate_dashboard_config(cfg, profile)
    warnings = override_warnings + warnings
    set_dashboard(cfg, saved=not warnings)
    st.session_state["open_warnings"] = warnings
    return warnings


def generate_dashboard(dataset_id: str, sheet: str | None, header_row: int | None) -> dict:
    storage = get_storage()
    meta, result, profile = load_for(dataset_id, sheet, header_row)
    cfg = build_default_dashboard(profile, result.df, dataset_id, meta["original_name"], result.sheet, header_row)
    cfg = storage.save_dashboard(cfg)
    set_dashboard(cfg, saved=True)
    st.session_state["open_warnings"] = []
    return cfg


def save_current() -> None:
    cfg = current_dashboard()
    if not cfg:
        return
    saved = get_storage().save_dashboard(cfg)
    cfg["updated_at"] = saved["updated_at"]
    st.session_state["saved_snapshot"] = snapshot(cfg)


def close_dashboard() -> None:
    for k in ("dashboard", "saved_snapshot", "staging", "qa_history", "open_warnings", "filters_initialised_for"):
        st.session_state.pop(k, None)


__all__ = [
    "ChartConfigError", "DataLoadError", "StorageError", "close_dashboard", "current_dashboard", "generate_dashboard",
    "get_storage", "is_dirty", "load_for", "open_dashboard", "raw_preview", "save_current", "set_dashboard",
]
