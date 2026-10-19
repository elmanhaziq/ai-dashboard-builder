"""Persistent analysis memory for one dataset (sheet): what the analyst learned, asked, and decided.

Stored as JSON next to the uploaded file (``uploads/<dataset_id>/analysis-<sheet>.json``) using the
existing atomic-write storage, so it survives restarts and travels with the dataset folder.
"""
from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any

from .workflow import Stage

MEMORY_VERSION = 1
MAX_EVENTS = 300
MAX_CONVERSATION = 400


def now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def memory_key(sheet: str | None) -> str:
    return hashlib.sha1((sheet or "").encode("utf-8")).hexdigest()[:10]


@dataclass
class AnalysisMemory:
    dataset_id: str
    sheet: str | None
    header_row: int | None
    dataset_name: str
    stage: str = Stage.PROFILE.value
    profiled: bool = False
    objective: str = ""
    objective_category: str = ""
    interpretations: dict[str, dict] = field(default_factory=dict)
    ambiguous: list[dict] = field(default_factory=list)
    questions: list[dict] = field(default_factory=list)
    pending_question: str | None = None
    cleaning_issues: list[dict] = field(default_factory=list)
    cleaning_reviewed: bool = False
    cleaning_applied: bool = False
    applied_ops: list[dict] = field(default_factory=list)
    transformations: list[dict] = field(default_factory=list)
    plan: dict | None = None
    plan_valid: bool = False
    plan_errors: list[str] = field(default_factory=list)
    dashboard_id: str | None = None
    conversation: list[dict] = field(default_factory=list)
    llm: dict = field(default_factory=lambda: {"transcript": [], "episode": 1, "summary": ""})
    consent: bool = False
    mode: str = "guided"
    llm_choice: dict = field(default_factory=dict)  # {"provider": ..., "model": ...} chosen by the user
    events: list[dict] = field(default_factory=list)
    created_at: str = field(default_factory=now)
    updated_at: str = field(default_factory=now)
    version: int = MEMORY_VERSION

    # ------------------------------------------------------------------ serialisation
    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: Any) -> "AnalysisMemory":
        if not isinstance(raw, dict) or not isinstance(raw.get("dataset_id"), str):
            raise ValueError("Analysis memory is malformed.")
        known = {f for f in cls.__dataclass_fields__}
        data = {k: v for k, v in raw.items() if k in known}
        mem = cls(**data)
        try:
            Stage(mem.stage)
        except ValueError:
            mem.stage = Stage.PROFILE.value
        return mem

    # ------------------------------------------------------------------ conversation
    def add_message(self, role: str, text: str, kind: str = "text", data: dict | None = None) -> None:
        self.conversation.append({"role": role, "text": text, "kind": kind, "data": data or {}, "at": now()})
        if len(self.conversation) > MAX_CONVERSATION:
            self.conversation = self.conversation[-MAX_CONVERSATION:]

    def log_event(self, name: str, detail: dict | None = None, ok: bool = True) -> None:
        self.events.append({"name": name, "detail": detail or {}, "ok": ok, "at": now()})
        if len(self.events) > MAX_EVENTS:
            self.events = self.events[-MAX_EVENTS:]

    # ------------------------------------------------------------------ questions
    def find_question(self, column: str | None, kind: str) -> dict | None:
        for q in self.questions:
            if q["column"] == column and q["kind"] == kind:
                return q
        return None

    def ask(self, question: str, column: str | None, kind: str, suggestions: list[str]) -> tuple[dict, bool]:
        """Register a question. Returns (question, is_new). An already-answered question is never re-asked."""
        existing = self.find_question(column, kind)
        if existing and existing["status"] == "answered":
            return existing, False
        if existing:
            q = existing
            q.update(question=question, suggestions=suggestions)
        else:
            q = {"id": uuid.uuid4().hex[:10], "column": column, "kind": kind, "question": question,
                 "suggestions": suggestions[:6], "status": "open", "answer": None, "asked_at": now(), "answered_at": None}
            self.questions.append(q)
        self.pending_question = q["id"]
        return q, True

    def pending(self) -> dict | None:
        return next((q for q in self.questions if q["id"] == self.pending_question), None)

    def answer_pending(self, text: str) -> dict | None:
        q = self.pending()
        if not q:
            return None
        q.update(status="answered", answer=text.strip()[:1000], answered_at=now())
        self.pending_question = None
        return q

    def answers_for(self, column: str) -> list[dict]:
        return [q for q in self.questions if q["column"] == column and q["status"] == "answered"]

    # ------------------------------------------------------------------ interpretations
    def unresolved_columns(self) -> list[str]:
        return [a["column"] for a in self.ambiguous
                if a.get("blocking") and a["column"] not in self.interpretations]

    def renames(self) -> dict[str, str]:
        """Original column name → friendly display name (only real changes)."""
        out = {}
        for col, it in self.interpretations.items():
            name = (it.get("display_name") or "").strip()
            if name and name != col and it.get("business_role") != "ignore":
                out[col] = name[:80]
        return out

    def approved_ops(self) -> list[dict]:
        return list(self.applied_ops)


# ---------------------------------------------------------------------- persistence helpers

def load_memory(storage, dataset_id: str, sheet: str | None) -> AnalysisMemory | None:
    raw = storage.load_analysis(dataset_id, memory_key(sheet))
    return AnalysisMemory.from_dict(raw) if raw else None


def save_memory(storage, memory: AnalysisMemory) -> None:
    memory.updated_at = now()
    storage.save_analysis(memory.dataset_id, memory_key(memory.sheet), json.loads(json.dumps(memory.to_dict(), default=str)))
