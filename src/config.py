"""Application configuration, read from environment variables with safe defaults."""
from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _load_dotenv(path: Path) -> None:
    """Minimal .env reader (KEY=VALUE lines). Existing environment variables win."""
    if not path.is_file():
        return
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))
    except OSError:
        pass


_load_dotenv(PROJECT_ROOT / ".env")


def _int_env(name: str, default: int) -> int:
    try:
        return max(1, int(os.environ.get(name, default)))
    except (TypeError, ValueError):
        return default


MAX_UPLOAD_MB: int = _int_env("DASHBOARD_MAX_UPLOAD_MB", 50)
MAX_UPLOAD_BYTES: int = MAX_UPLOAD_MB * 1024 * 1024
MAX_ROWS: int = _int_env("DASHBOARD_MAX_ROWS", 1_000_000)

DATA_DIR: Path = Path(os.environ.get("DASHBOARD_DATA_DIR", str(PROJECT_ROOT))).resolve()
UPLOAD_DIR: Path = DATA_DIR / "uploads"
DASHBOARD_DIR: Path = DATA_DIR / "saved_dashboards"

# Categorical chart palette (fixed order, never cycled past the end; extra series fold into "Other").
SERIES_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
ACCENT = "#1f5fbf"
