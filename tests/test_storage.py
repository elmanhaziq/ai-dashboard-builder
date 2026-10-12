import json

import pytest

from src.dashboard_service import build_default_dashboard
from src.storage import StorageError, safe_display_name
from tests import synthetic as syn
from tests.conftest import csv_bytes, load_and_profile


def _dashboard(storage, name="sales.csv"):
    data = csv_bytes(syn.sales())
    meta = storage.save_upload(name, data, ".csv")
    res, p = load_and_profile(syn.sales())
    cfg = build_default_dashboard(p, res.df, meta["id"], meta["original_name"], "CSV data")
    return meta, storage.save_dashboard(cfg)


def test_save_and_reopen_dashboard(storage):
    meta, cfg = _dashboard(storage)
    loaded = storage.load_dashboard(cfg["id"])
    assert loaded["charts"] == cfg["charts"] and loaded["kpis"] == cfg["kpis"]
    assert loaded["dataset_id"] == meta["id"]
    # Data is not embedded in the dashboard file
    text = (storage.dashboard_dir / f"{cfg['id']}.json").read_text(encoding="utf-8")
    assert len(text) < 20_000 and "Product 12" not in text
    # No temp files left behind by atomic writes
    assert not list(storage.dashboard_dir.glob(".tmp-*"))


def test_list_rename_duplicate_delete(storage):
    _, cfg = _dashboard(storage)
    storage.rename_dashboard(cfg["id"], "  Q1 review  ")
    assert storage.load_dashboard(cfg["id"])["name"] == "Q1 review"
    copy = storage.duplicate_dashboard(cfg["id"])
    assert copy["id"] != cfg["id"] and copy["name"].endswith("(copy)")
    assert {d["id"] for d in storage.list_dashboards()} == {cfg["id"], copy["id"]}
    storage.delete_dashboard(copy["id"])
    assert [d["id"] for d in storage.list_dashboards()] == [cfg["id"]]
    with pytest.raises(StorageError):
        storage.rename_dashboard(cfg["id"], "   ")


def test_duplicate_filenames_do_not_overwrite(storage):
    a = storage.save_upload("report.csv", b"a,b\n1,2\n", ".csv")
    b = storage.save_upload("report.csv", b"a,b\n3,4\n", ".csv")
    assert a["id"] != b["id"]
    assert storage.dataset_file(a["id"]).read_bytes() == b"a,b\n1,2\n"
    assert storage.dataset_file(b["id"]).read_bytes() == b"a,b\n3,4\n"
    # identical re-upload is de-duplicated
    assert storage.save_upload("report.csv", b"a,b\n1,2\n", ".csv")["id"] == a["id"]


def test_delete_dataset_removes_files_and_dashboards(storage):
    meta, cfg = _dashboard(storage)
    removed = storage.delete_dataset(meta["id"])
    assert removed == 1
    assert not (storage.upload_dir / meta["id"]).exists()
    assert storage.list_dashboards() == []


@pytest.mark.parametrize("bad_id", ["../evil", "..\\evil", "ABC", "a" * 40, "", "abcd/1234", "12345678.json"])
def test_path_traversal_ids_rejected(storage, bad_id):
    with pytest.raises(StorageError):
        storage.load_dashboard(bad_id)
    with pytest.raises(StorageError):
        storage.get_dataset(bad_id)


def test_corrupted_dashboard_file_is_reported(storage):
    bad = storage.dashboard_dir / "0123456789abcdef.json"
    bad.write_text("{not json", encoding="utf-8")
    with pytest.raises(StorageError, match="corrupted"):
        storage.load_dashboard("0123456789abcdef")
    listing = storage.list_dashboards()
    assert listing[0]["error"]


def test_dashboard_with_wrong_embedded_id_rejected(storage):
    _, cfg = _dashboard(storage)
    path = storage.dashboard_dir / f"{cfg['id']}.json"
    raw = json.loads(path.read_text(encoding="utf-8"))
    raw["id"] = "ffffffffffffffff"
    path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(StorageError):
        storage.load_dashboard(cfg["id"])


def test_safe_display_name():
    assert safe_display_name("..\\..\\Windows\\evil.csv") == "evil.csv"
    assert safe_display_name("/etc/passwd") == "passwd"
    assert safe_display_name('a<b>:"c".xlsx') == "a_b___c_.xlsx"


def test_clear_all(storage):
    _dashboard(storage)
    n_ds, n_db = storage.clear_all()
    assert (n_ds, n_db) == (1, 1)
    assert storage.list_datasets() == [] and storage.list_dashboards() == []
