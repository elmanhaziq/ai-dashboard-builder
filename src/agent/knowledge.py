"""What the assistant has learned, so it adapts to each kind of data over time.

* **Dataset notes**: facts about one data layout ("Amount is in RM", "Region 'KL' means Kuala Lumpur"), stored under a
  signature of the column names. A file with the same columns (e.g. next month's report) reuses them, and a file with
  mostly the same columns gets them too, marked as coming from a similar file.
* **Word meanings** (aliases): "when I say *department* I mean *Product Category*". These also help the offline
  question engine, not only the AI.
* **General preferences**: how the user likes answers and charts ("prefers horizontal bars", "answer in Malay").

Notes come from the conversation with the user (the assistant saves them with its ``remember`` tool) and the user can
see and delete every one of them. They are stored next to the saved dashboards, so in online mode they disappear with
the visitor's session like everything else.
"""
from __future__ import annotations

import hashlib
import json
import re
import threading
from datetime import datetime, timezone
from pathlib import Path

FILE_NAME = "assistant_knowledge.json"
MAX_NOTES = 40
MAX_GENERAL = 30
MAX_ALIASES = 60
MAX_TEXT = 300
SIMILAR_JACCARD = 0.6

_LOCK = threading.Lock()
_CTRL = re.compile(r"[\x00-\x1f\x7f]+")


def _clean(text, n: int = MAX_TEXT) -> str:
    return _CTRL.sub(" ", str(text or "")).strip()[:n]


def signature(columns) -> str:
    names = sorted({str(c).strip().casefold() for c in columns})
    return hashlib.sha1("\x1f".join(names).encode("utf-8")).hexdigest()[:16]


def _path(base_dir: Path) -> Path:
    return Path(base_dir) / FILE_NAME


def _load(base_dir: Path) -> dict:
    try:
        data = json.loads(_path(base_dir).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        data = {}
    if not isinstance(data, dict):
        data = {}
    data.setdefault("general", [])
    data.setdefault("datasets", {})
    return data


def _save(base_dir: Path, data: dict) -> None:
    from ..storage import atomic_write_text

    Path(base_dir).mkdir(parents=True, exist_ok=True)
    atomic_write_text(_path(base_dir), json.dumps(data, indent=1, ensure_ascii=False))


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _entry(data: dict, columns, name: str = "") -> dict:
    sig = signature(columns)
    e = data["datasets"].setdefault(sig, {"columns": [], "notes": [], "aliases": {}, "seen": 0, "name": ""})
    e["columns"] = [str(c)[:80] for c in list(columns)[:80]]
    if name:
        e["name"] = _clean(name, 120)
    return e


def _similar(data: dict, columns) -> tuple[str, dict] | None:
    cols = {str(c).strip().casefold() for c in columns}
    best, score = None, 0.0
    for sig, e in data["datasets"].items():
        other = {str(c).strip().casefold() for c in e.get("columns", [])}
        if not other or not (e.get("notes") or e.get("aliases")):
            continue
        j = len(cols & other) / len(cols | other)
        if j > score:
            best, score = (sig, e), j
    return best if best and score >= SIMILAR_JACCARD else None


def for_dataset(base_dir: Path, columns) -> dict:
    """Everything learned that applies to a dataset with these columns."""
    data = _load(base_dir)
    exact = data["datasets"].get(signature(columns))
    similar = None
    if not exact or not (exact.get("notes") or exact.get("aliases")):
        hit = _similar(data, columns)
        if hit and hit[0] != signature(columns):
            similar = hit[1]
    src = exact if exact and (exact.get("notes") or exact.get("aliases")) else similar or {}
    present = {str(c) for c in columns}
    return {
        "notes": [n["text"] for n in src.get("notes", [])],
        "aliases": {t: c for t, c in (src.get("aliases") or {}).items() if c in present},
        "general": [n["text"] for n in data["general"]],
        "from_similar_file": bool(similar) and src is similar,
        "similar_name": similar.get("name", "") if similar else "",
        "seen": (exact or {}).get("seen", 0),
    }


def remember(base_dir: Path, columns, text: str, scope: str = "dataset", term: str = "", column: str = "",
             dataset_name: str = "") -> str:
    """Save one learned fact. Returns a short confirmation (or why nothing was saved)."""
    text = _clean(text)
    term, column = _clean(term, 60), _clean(column, 120)
    with _LOCK:
        data = _load(base_dir)
        if term and column:
            if column not in {str(c) for c in columns}:
                return f"Not saved: there's no column called '{column}'."
            e = _entry(data, columns, dataset_name)
            e["aliases"][term.casefold()] = column
            e["aliases"] = dict(list(e["aliases"].items())[-MAX_ALIASES:])
            _save(base_dir, data)
            return f"Remembered: “{term}” means {column}."
        if len(text) < 4:
            return "Not saved: the note is empty."
        target = data["general"] if scope == "general" else _entry(data, columns, dataset_name)["notes"]
        if any(n["text"].casefold() == text.casefold() for n in target):
            return "Already remembered."
        target.append({"text": text, "at": _now()})
        limit = MAX_GENERAL if scope == "general" else MAX_NOTES
        del target[:-limit]
        _save(base_dir, data)
    return "Remembered."


def forget(base_dir: Path, columns, kind: str, key) -> None:
    """Delete one note ('notes' / 'general' by index) or one alias ('aliases' by term)."""
    with _LOCK:
        data = _load(base_dir)
        if kind == "general":
            if 0 <= int(key) < len(data["general"]):
                del data["general"][int(key)]
        else:
            e = data["datasets"].get(signature(columns))
            if not e:
                return
            if kind == "notes" and 0 <= int(key) < len(e["notes"]):
                del e["notes"][int(key)]
            elif kind == "aliases":
                e["aliases"].pop(str(key), None)
        _save(base_dir, data)


def adopt_similar(base_dir: Path, columns, dataset_name: str = "") -> None:
    """Copy notes learned on a similar file to this layout (called the first time a similar file is used)."""
    with _LOCK:
        data = _load(base_dir)
        e = data["datasets"].get(signature(columns))
        if e and (e.get("notes") or e.get("aliases")):
            return
        hit = _similar(data, columns)
        if not hit:
            return
        present = {str(c) for c in columns}
        new = _entry(data, columns, dataset_name)
        new["notes"] = [dict(n) for n in hit[1].get("notes", [])]
        new["aliases"] = {t: c for t, c in hit[1].get("aliases", {}).items() if c in present}
        _save(base_dir, data)


def touch(base_dir: Path, columns, dataset_name: str = "") -> int:
    """Count how many times this data layout has been opened with the assistant. Returns the count."""
    with _LOCK:
        data = _load(base_dir)
        e = _entry(data, columns, dataset_name)
        e["seen"] = int(e.get("seen", 0)) + 1
        e["last_seen"] = _now()
        _save(base_dir, data)
        return e["seen"]


def engine_aliases(knowledge: dict) -> dict[str, list[str]]:
    """Learned word meanings in the format the question engine expects ({column: [other names]})."""
    out: dict[str, list[str]] = {}
    for term, col in (knowledge.get("aliases") or {}).items():
        out.setdefault(col, []).append(term)
    return out
