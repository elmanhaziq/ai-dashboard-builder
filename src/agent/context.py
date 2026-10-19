"""Everything the agent tools need for one dataset, independent of Streamlit (so it is fully testable)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import pandas as pd

from ..data_loader import LoadResult, load_dataset
from ..data_profiler import DatasetProfile, profile_dataset
from ..storage import Storage
from .ambiguity import find_ambiguous
from .memory import AnalysisMemory, load_memory, save_memory
from .pipeline import build_working
from .workflow import Stage

Loader = Callable[[str, str | None, int | None], tuple[LoadResult, DatasetProfile]]


def direct_loader(storage: Storage) -> Loader:
    def load(dataset_id: str, sheet: str | None, header_row: int | None):
        meta = storage.get_dataset(dataset_id)
        result = load_dataset(storage.dataset_file(dataset_id), meta["ext"], sheet, header_row)
        return result, profile_dataset(result.df, result.percent_columns, result.currency_columns)
    return load


@dataclass
class AgentContext:
    storage: Storage
    memory: AnalysisMemory
    loader: Loader
    sample_rows_allowed: bool = True
    _cache: dict = field(default_factory=dict)

    # ------------------------------------------------------------------ construction
    @classmethod
    def open(cls, storage: Storage, dataset_id: str, sheet: str | None, header_row: int | None = None,
             loader: Loader | None = None) -> "AgentContext":
        """Restore the saved analysis for this dataset/sheet, or start a new one."""
        memory = load_memory(storage, dataset_id, sheet)
        if memory is None:
            meta = storage.get_dataset(dataset_id)
            memory = AnalysisMemory(dataset_id=dataset_id, sheet=sheet, header_row=header_row, dataset_name=meta["original_name"])
        ctx = cls(storage=storage, memory=memory, loader=loader or direct_loader(storage))
        if not memory.profiled:
            ctx.profile_dataset()
        return ctx

    def save(self) -> None:
        save_memory(self.storage, self.memory)

    # ------------------------------------------------------------------ data access
    @property
    def original(self) -> LoadResult:
        if "orig" not in self._cache:
            self._cache["orig"], self._cache["orig_profile"] = self.loader(
                self.memory.dataset_id, self.memory.sheet, self.memory.header_row)
        return self._cache["orig"]

    @property
    def original_profile(self) -> DatasetProfile:
        self.original  # noqa: B018 - populate cache
        return self._cache["orig_profile"]

    def working(self) -> tuple[pd.DataFrame, DatasetProfile]:
        """Cleaned + interpreted data (recomputed when decisions change)."""
        m = self.memory
        key = repr((m.applied_ops, sorted(m.interpretations.items(), key=lambda kv: kv[0]) if m.interpretations else []))
        if self._cache.get("work_key") != key:
            r = self.original
            df, profile, _, renames = build_working(r.df, m.applied_ops, m.interpretations, m.renames(),
                                                    r.percent_columns, r.currency_columns)
            self._cache.update(work_key=key, work_df=df, work_profile=profile, renames=renames)
        return self._cache["work_df"], self._cache["work_profile"]

    def column_name(self, original: str) -> str:
        """Current (possibly renamed) name of an original column."""
        self.working()
        return self._cache.get("renames", {}).get(original, original)

    def original_name(self, name: str) -> str:
        self.working()
        inv = {v: k for k, v in self._cache.get("renames", {}).items()}
        return inv.get(name, name)

    def invalidate(self) -> None:
        self._cache.pop("work_key", None)

    # ------------------------------------------------------------------ stage A: discovery
    def profile_dataset(self) -> None:
        r, p = self.original, self.original_profile
        m = self.memory
        m.ambiguous = find_ambiguous(p, r.currency_columns)
        m.profiled = True
        if Stage(m.stage) in (Stage.UPLOAD, Stage.PROFILE):
            m.stage = Stage.OBJECTIVE.value
        m.log_event("profile_dataset", {"rows": p.n_rows, "columns": p.n_cols, "ambiguous": [a["column"] for a in m.ambiguous]})
        self.save()
