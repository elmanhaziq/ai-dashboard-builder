"""Online-mode building blocks: per-session storage, session expiry, and portable dashboard bundles."""
import io
import json
import os
import time
import zipfile

import pytest

from src.dashboard_service import build_default_dashboard
from src.storage import BUNDLE_CONFIG, BUNDLE_META, StorageError, session_storage, sweep_sessions
from tests import synthetic as syn
from tests.conftest import csv_bytes, load_and_profile


def _make_dashboard(storage):
    meta = storage.save_upload("sales.csv", csv_bytes(syn.sales()), ".csv")
    res, p = load_and_profile(syn.sales())
    cfg = build_default_dashboard(p, res.df, meta["id"], meta["original_name"], "CSV data")
    return storage.save_dashboard(cfg)


def test_sessions_are_isolated(tmp_path):
    a = session_storage(tmp_path, "a" * 32)
    b = session_storage(tmp_path, "b" * 32)
    cfg = _make_dashboard(a)
    assert [d["id"] for d in a.list_dashboards()] == [cfg["id"]]
    assert b.list_dashboards() == [] and b.list_datasets() == []
    with pytest.raises(StorageError):
        b.load_dashboard(cfg["id"])  # other visitors can't open it even with the id
    with pytest.raises(StorageError):
        session_storage(tmp_path, "../" + "c" * 29)


def test_inactive_sessions_are_swept(tmp_path):
    old = session_storage(tmp_path, "a" * 32)
    _make_dashboard(old)
    fresh = session_storage(tmp_path, "b" * 32)
    two_hours_ago = time.time() - 3 * 3600
    os.utime(tmp_path / ("a" * 32), (two_hours_ago, two_hours_ago))
    (tmp_path / "not-a-session").mkdir()
    assert sweep_sessions(tmp_path, ttl_minutes=120) == 1
    assert not (tmp_path / ("a" * 32)).exists()
    assert (tmp_path / ("b" * 32)).exists() and (tmp_path / "not-a-session").exists()
    assert fresh.list_dashboards() == []


def test_bundle_round_trip(tmp_path):
    src = session_storage(tmp_path, "a" * 32)
    cfg = _make_dashboard(src)
    cfg["charts"][0]["title"] = "My custom chart"
    cfg["column_overrides"] = {"Customer ID": {"role": "dimension"}}
    src.save_dashboard(cfg)
    fname, blob = src.export_bundle(cfg["id"])
    assert fname.endswith(".dashboard.zip")

    dst = session_storage(tmp_path, "b" * 32)
    restored = dst.import_bundle(blob)
    assert restored["id"] != cfg["id"] and restored["dataset_id"] != cfg["dataset_id"]
    assert restored["charts"][0]["title"] == "My custom chart"
    assert restored["column_overrides"] == {"Customer ID": {"role": "dimension"}}
    assert dst.dataset_file(restored["dataset_id"]).read_bytes() == src.dataset_file(cfg["dataset_id"]).read_bytes()


def _zip(entries: dict) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name, data in entries.items():
            zf.writestr(name, data)
    return buf.getvalue()


@pytest.mark.parametrize("blob", [
    b"not a zip at all",
    _zip({"dashboard.json": "{}"}),
    _zip({BUNDLE_CONFIG: "{not json", BUNDLE_META: "{}", "data/original.csv": "a,b\n1,2\n"}),
    _zip({BUNDLE_CONFIG: json.dumps({"id": "../../x"}), BUNDLE_META: "{}", "data/original.csv": "a,b\n1,2\n"}),
    _zip({BUNDLE_CONFIG: json.dumps({"id": "a" * 16, "dataset_id": "b" * 16}), BUNDLE_META: "{}",
          "data/original.xlsx": "this is not a workbook"}),
])
def test_bad_bundles_are_rejected(tmp_path, blob):
    st = session_storage(tmp_path, "a" * 32)
    with pytest.raises(StorageError):
        st.import_bundle(blob)
    assert st.list_datasets() == [] and st.list_dashboards() == []


def test_bundle_size_limit_checked_before_decompressing(tmp_path):
    st = session_storage(tmp_path, "a" * 32)
    bomb = _zip({BUNDLE_CONFIG: json.dumps({"id": "a" * 16, "dataset_id": "b" * 16}), BUNDLE_META: "{}",
                 "data/original.csv": "0" * 5_000_000})
    assert len(bomb) < 100_000  # compresses very well
    with pytest.raises(StorageError, match="larger than the upload limit"):
        st.import_bundle(bomb, max_bytes=1_000_000)


def test_mode_detection(monkeypatch):
    import importlib

    import src.config as cfg
    monkeypatch.setenv("DASHBOARD_MODE", "online")
    assert importlib.reload(cfg).ONLINE is True
    assert cfg.MAX_UPLOAD_MB == 20
    monkeypatch.setenv("DASHBOARD_MODE", "local")
    assert importlib.reload(cfg).ONLINE is False
    monkeypatch.delenv("DASHBOARD_MODE")
    importlib.reload(cfg)
