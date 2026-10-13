"""Local filesystem persistence for uploaded datasets and dashboard configurations.

Layout::

    uploads/<dataset_id>/original<.ext>   the file exactly as uploaded
    uploads/<dataset_id>/meta.json        original filename, size, hash, sheets
    saved_dashboards/<dashboard_id>.json  chart/KPI/filter configuration (no data)

Files are stored unencrypted on the local disk. Deletion uses normal file removal
(it is not a secure wipe).
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import re
import shutil
import tempfile
import time
import uuid
import zipfile
from pathlib import Path
from typing import Any

from . import config
from .dashboard_service import new_dashboard_id, now_iso, validate_dashboard_config
from .data_loader import ALLOWED_EXTENSIONS

_ID_RE = re.compile(r"^[a-f0-9]{8,32}$")


class StorageError(Exception):
    """User-facing storage problem."""


def _check_id(value: str, kind: str = "id") -> str:
    if not isinstance(value, str) or not _ID_RE.match(value):
        raise StorageError(f"Invalid {kind}.")
    return value


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".tmp-", suffix=path.suffix)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".tmp-")
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def safe_display_name(filename: str) -> str:
    """Strip any path components and control characters from a user-supplied name."""
    name = Path(str(filename).replace("\\", "/")).name
    name = re.sub(r"[\x00-\x1f<>:\"|?*]", "_", name).strip(" .")
    return name[:200] or "upload"


class Storage:
    def __init__(self, upload_dir: Path | None = None, dashboard_dir: Path | None = None):
        self.upload_dir = Path(upload_dir or config.UPLOAD_DIR)
        self.dashboard_dir = Path(dashboard_dir or config.DASHBOARD_DIR)
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.dashboard_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ datasets
    def _dataset_dir(self, dataset_id: str) -> Path:
        path = (self.upload_dir / _check_id(dataset_id, "dataset id")).resolve()
        if path.parent != self.upload_dir.resolve():
            raise StorageError("Invalid dataset path.")
        return path

    def save_upload(self, filename: str, data: bytes, ext: str) -> dict:
        ext = ext.lower()
        if ext not in ALLOWED_EXTENSIONS:
            raise StorageError("Unsupported file type.")
        digest = hashlib.sha256(data).hexdigest()
        for meta in self.list_datasets():
            if meta.get("sha256") == digest and meta.get("original_name") == safe_display_name(filename):
                return meta  # identical file re-uploaded: reuse it rather than storing a copy
        dataset_id = uuid.uuid4().hex[:16]
        folder = self._dataset_dir(dataset_id)
        meta = {
            "id": dataset_id,
            "original_name": safe_display_name(filename),
            "stored_name": f"original{ext}",
            "ext": ext,
            "size": len(data),
            "sha256": digest,
            "uploaded_at": now_iso(),
        }
        try:
            atomic_write_bytes(folder / meta["stored_name"], data)
            atomic_write_text(folder / "meta.json", json.dumps(meta, indent=2))
        except OSError as exc:
            shutil.rmtree(folder, ignore_errors=True)
            raise StorageError(f"Could not save the upload to disk ({exc.strerror or exc}).") from exc
        return meta

    def update_dataset_meta(self, dataset_id: str, **fields: Any) -> dict:
        meta = self.get_dataset(dataset_id)
        meta.update({k: v for k, v in fields.items() if k not in ("id", "stored_name", "ext", "sha256")})
        atomic_write_text(self._dataset_dir(dataset_id) / "meta.json", json.dumps(meta, indent=2, default=str))
        return meta

    def get_dataset(self, dataset_id: str) -> dict:
        folder = self._dataset_dir(dataset_id)
        try:
            meta = json.loads((folder / "meta.json").read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise StorageError("The dataset for this dashboard no longer exists on disk.") from exc
        except (OSError, json.JSONDecodeError) as exc:
            raise StorageError("The dataset's metadata file is unreadable.") from exc
        if not isinstance(meta, dict) or meta.get("id") != dataset_id or meta.get("ext") not in ALLOWED_EXTENSIONS:
            raise StorageError("The dataset's metadata file is invalid.")
        return meta

    def dataset_file(self, dataset_id: str) -> Path:
        meta = self.get_dataset(dataset_id)
        path = self._dataset_dir(dataset_id) / f"original{meta['ext']}"
        if not path.is_file():
            raise StorageError("The uploaded file for this dataset is missing.")
        return path

    def list_datasets(self) -> list[dict]:
        out = []
        for folder in sorted(self.upload_dir.iterdir()) if self.upload_dir.exists() else []:
            if folder.is_dir() and _ID_RE.match(folder.name):
                try:
                    out.append(self.get_dataset(folder.name))
                except StorageError:
                    continue
        return sorted(out, key=lambda m: m.get("uploaded_at", ""), reverse=True)

    def delete_dataset(self, dataset_id: str, delete_dashboards: bool = True) -> int:
        """Delete a dataset folder and (by default) dashboards built on it. Returns dashboards removed."""
        folder = self._dataset_dir(dataset_id)
        removed = 0
        if delete_dashboards:
            for d in self.list_dashboards():
                if d.get("dataset_id") == dataset_id:
                    self.delete_dashboard(d["id"])
                    removed += 1
        shutil.rmtree(folder, ignore_errors=False) if folder.exists() else None
        return removed

    # ------------------------------------------------------------------ dashboards
    def _dashboard_path(self, dashboard_id: str) -> Path:
        path = (self.dashboard_dir / f"{_check_id(dashboard_id, 'dashboard id')}.json").resolve()
        if path.parent != self.dashboard_dir.resolve():
            raise StorageError("Invalid dashboard path.")
        return path

    def save_dashboard(self, cfg: dict) -> dict:
        clean, _ = validate_dashboard_config(cfg)
        clean["updated_at"] = now_iso()
        try:
            atomic_write_text(self._dashboard_path(clean["id"]), json.dumps(clean, indent=2, default=str))
        except OSError as exc:
            raise StorageError(f"Could not save the dashboard ({exc.strerror or exc}).") from exc
        return clean

    def load_dashboard(self, dashboard_id: str) -> dict:
        path = self._dashboard_path(dashboard_id)
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise StorageError("That dashboard no longer exists.") from exc
        except (OSError, json.JSONDecodeError) as exc:
            raise StorageError("The dashboard file is corrupted and cannot be opened.") from exc
        try:
            cfg, _ = validate_dashboard_config(raw)
        except Exception as exc:  # noqa: BLE001
            raise StorageError(f"The dashboard file is invalid: {exc}") from exc
        if cfg["id"] != dashboard_id:
            raise StorageError("The dashboard file's id does not match its name.")
        return cfg

    def list_dashboards(self) -> list[dict]:
        out = []
        for path in self.dashboard_dir.glob("*.json"):
            if not _ID_RE.match(path.stem):
                continue
            try:
                cfg = self.load_dashboard(path.stem)
            except StorageError as exc:
                out.append({"id": path.stem, "name": path.stem, "error": str(exc), "updated_at": ""})
                continue
            out.append({k: cfg.get(k) for k in ("id", "name", "dataset_id", "dataset_name", "sheet", "created_at", "updated_at")}
                       | {"n_charts": len(cfg["charts"]), "n_kpis": len(cfg["kpis"])})
        return sorted(out, key=lambda d: d.get("updated_at") or "", reverse=True)

    def rename_dashboard(self, dashboard_id: str, new_name: str) -> dict:
        name = str(new_name).strip()[:120]
        if not name:
            raise StorageError("Dashboard name cannot be empty.")
        cfg = self.load_dashboard(dashboard_id)
        cfg["name"] = name
        return self.save_dashboard(cfg)

    def duplicate_dashboard(self, dashboard_id: str) -> dict:
        cfg = self.load_dashboard(dashboard_id)
        cfg = dict(cfg, id=new_dashboard_id(), name=f"{cfg['name']} (copy)"[:120], created_at=now_iso())
        return self.save_dashboard(cfg)

    def delete_dashboard(self, dashboard_id: str) -> None:
        path = self._dashboard_path(dashboard_id)
        try:
            path.unlink()
        except FileNotFoundError:
            pass
        except OSError as exc:
            raise StorageError(f"Could not delete the dashboard ({exc.strerror or exc}).") from exc

    def clear_all(self) -> tuple[int, int]:
        datasets = self.list_datasets()
        dashboards = list(self.dashboard_dir.glob("*.json"))
        for p in dashboards:
            p.unlink(missing_ok=True)
        for meta in datasets:
            shutil.rmtree(self._dataset_dir(meta["id"]), ignore_errors=True)
        return len(datasets), len(dashboards)

    def disk_usage(self) -> int:
        return sum(p.stat().st_size for p in self.upload_dir.rglob("*") if p.is_file())

    # ------------------------------------------------------------------ portable bundles
    def export_bundle(self, dashboard_id: str) -> tuple[str, bytes]:
        """A .zip holding the dashboard settings and its original data file, for download."""
        cfg = self.load_dashboard(dashboard_id)
        meta = self.get_dataset(cfg["dataset_id"])
        data = self.dataset_file(cfg["dataset_id"]).read_bytes()
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.writestr(BUNDLE_CONFIG, json.dumps(cfg, indent=2, default=str))
            zf.writestr(BUNDLE_META, json.dumps({"original_name": meta["original_name"], "format": BUNDLE_FORMAT}, indent=2))
            zf.writestr(f"data/original{meta['ext']}", data)
        stem = re.sub(r"[^A-Za-z0-9._-]+", "_", cfg["name"]).strip("_")[:60] or "dashboard"
        return f"{stem}.dashboard.zip", buf.getvalue()

    def import_bundle(self, blob: bytes, max_bytes: int | None = None) -> dict:
        """Restore a dashboard exported with :meth:`export_bundle`. Returns the saved config.

        The archive is treated as untrusted: only the three expected entries are read, sizes are capped
        before decompression, and the configuration is validated like any other saved dashboard.
        """
        from .data_loader import DataLoadError, _detect_format, list_sheets

        limit = max_bytes or config.MAX_UPLOAD_BYTES
        if len(blob) > limit:
            raise StorageError("The dashboard file is larger than the upload limit.")
        try:
            zf = zipfile.ZipFile(io.BytesIO(blob))
        except zipfile.BadZipFile as exc:
            raise StorageError("This isn't a dashboard file (.dashboard.zip) exported from this app.") from exc
        with zf:
            infos = {i.filename: i for i in zf.infolist()}
            data_names = [n for n in infos if re.fullmatch(r"data/original\.(xlsx|xls|csv)", n)]
            if BUNDLE_CONFIG not in infos or BUNDLE_META not in infos or len(data_names) != 1:
                raise StorageError("This isn't a dashboard file (.dashboard.zip) exported from this app.")
            for name in (BUNDLE_CONFIG, BUNDLE_META, data_names[0]):
                if infos[name].file_size > limit:
                    raise StorageError("The data inside this dashboard file is larger than the upload limit.")
            try:
                raw_cfg = json.loads(zf.read(BUNDLE_CONFIG).decode("utf-8"))
                meta = json.loads(zf.read(BUNDLE_META).decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise StorageError("The dashboard file is damaged.") from exc
            data = zf.read(data_names[0])
        ext = "." + data_names[0].rsplit(".", 1)[1]
        try:
            cfg, _ = validate_dashboard_config(raw_cfg)
        except Exception as exc:  # noqa: BLE001
            raise StorageError(f"The dashboard settings are invalid: {exc}") from exc
        try:
            if _detect_format(data, ext) != ext:  # bundles we export always match; anything else was tampered with
                raise DataLoadError("its contents don't match its file type.")
            list_sheets(data, ext)
        except DataLoadError as exc:
            raise StorageError(f"The data inside the dashboard file can't be read: {exc}") from exc
        name = meta.get("original_name") if isinstance(meta, dict) else None
        dataset = self.save_upload(str(name or f"data{ext}"), data, ext)
        cfg.update(id=new_dashboard_id(), dataset_id=dataset["id"], dataset_name=dataset["original_name"], created_at=now_iso())
        return self.save_dashboard(cfg)


BUNDLE_CONFIG = "dashboard.json"
BUNDLE_META = "data/meta.json"
BUNDLE_FORMAT = "ai-dashboard-builder/1"

_SESSION_RE = re.compile(r"^[a-f0-9]{32}$")


def session_storage(root: Path, session_id: str) -> Storage:
    """Private storage for one browser session (online mode)."""
    if not _SESSION_RE.match(session_id):
        raise StorageError("Invalid session.")
    base = Path(root) / session_id
    base.mkdir(parents=True, exist_ok=True)
    os.utime(base)  # mark activity for the inactivity sweep
    return Storage(base / "uploads", base / "saved_dashboards")


def sweep_sessions(root: Path, ttl_minutes: int, now: float | None = None) -> int:
    """Delete session folders inactive for longer than ``ttl_minutes``. Returns the number removed."""
    root = Path(root)
    if not root.is_dir():
        return 0
    cutoff = (now if now is not None else time.time()) - ttl_minutes * 60
    removed = 0
    for child in root.iterdir():
        try:
            if child.is_dir() and _SESSION_RE.match(child.name) and child.stat().st_mtime < cutoff:
                shutil.rmtree(child, ignore_errors=True)
                removed += 1
        except OSError:
            continue
    return removed
