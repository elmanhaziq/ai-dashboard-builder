"""Application configuration, read from environment variables with safe defaults."""
from __future__ import annotations

import os
import tempfile
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


def _detect_mode() -> str:
    """'local' (single user, persistent folders) or 'online' (shared server, private per-visitor sessions).

    Set DASHBOARD_MODE explicitly; otherwise Streamlit Community Cloud is detected from its app path.
    """
    mode = os.environ.get("DASHBOARD_MODE", "").strip().lower()
    if mode in ("local", "online"):
        return mode
    return "online" if str(PROJECT_ROOT).replace("\\", "/").startswith("/mount/src/") else "local"


MODE: str = _detect_mode()
ONLINE: bool = MODE == "online"

# Online servers have little memory and are shared, so the defaults are smaller there.
MAX_UPLOAD_MB: int = _int_env("DASHBOARD_MAX_UPLOAD_MB", 20 if ONLINE else 50)
MAX_UPLOAD_BYTES: int = MAX_UPLOAD_MB * 1024 * 1024
MAX_ROWS: int = _int_env("DASHBOARD_MAX_ROWS", 300_000 if ONLINE else 1_000_000)

DATA_DIR: Path = Path(os.environ.get("DASHBOARD_DATA_DIR", str(PROJECT_ROOT))).resolve()
UPLOAD_DIR: Path = DATA_DIR / "uploads"
DASHBOARD_DIR: Path = DATA_DIR / "saved_dashboards"

# Online mode: each browser session gets its own temporary folder, removed after this much inactivity.
SESSION_ROOT: Path = Path(os.environ.get("DASHBOARD_SESSION_DIR", str(Path(tempfile.gettempdir()) / "dashboard_sessions")))
SESSION_TTL_MINUTES: int = _int_env("DASHBOARD_SESSION_TTL_MINUTES", 120)

# Categorical chart palette (fixed order, never cycled past the end; extra series fold into "Other").
SERIES_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
ACCENT = "#1f5fbf"
