"""Run the analyst's discovery stage on a spreadsheet and print what it would ask and propose.

Prints column-level findings only (no data rows). Uses the offline guided rules; no AI service is called.

    .venv\\Scripts\\python.exe scripts\\benchmark_agent.py path\\to\\file.xlsx [sheet]
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.agent.context import AgentContext  # noqa: E402
from src.agent.tools import run_tool  # noqa: E402
from src.data_loader import list_sheets  # noqa: E402
from src.storage import Storage  # noqa: E402


def main(path: str, sheet: str | None = None) -> None:
    p = Path(path)
    data = p.read_bytes()
    tmp = Path(tempfile.mkdtemp())
    storage = Storage(tmp / "uploads", tmp / "dash")  # scratch copy: the benchmark never touches your library
    meta = storage.save_upload(p.name, data, p.suffix.lower())
    sheet = sheet or list_sheets(data, p.suffix.lower())[0]
    ctx = AgentContext.open(storage, meta["id"], sheet)
    r, prof = ctx.original, ctx.original_profile
    print(f"File: {p.name} | sheet: {sheet} | rows: {prof.n_rows:,} | columns: {prof.n_cols}")
    print("\nColumn roles:")
    for c, cp in prof.columns.items():
        print(f"  - {c}: {cp.role}" + (f" ({cp.aggregation})" if cp.role == 'metric' else ""))
    print("\nLoading notes:", *(f"  - {w}" for w in r.warnings) or ["  (none)"], sep="\n")
    print("\nColumns the analyst would ask about (in order):")
    for a in ctx.memory.ambiguous:
        print(f"  - {a['column']} [{'must ask' if a['blocking'] else 'optional'}] {a['reasons'][0]}  suggestions: {a['suggestions'][:2]}")
    if not ctx.memory.ambiguous:
        print("  (none — all names are clear)")
    ctx.memory.objective, ctx.memory.objective_category = "General exploration", "exploratory"
    for a in ctx.memory.ambiguous:  # pretend the user skipped every question so we can reach the quality review
        ctx.memory.interpretations[a["column"]] = {"business_role": "ignore", "meaning": "skipped", "aggregation": "none"}
    from src.agent.workflow import sync
    sync(ctx.memory)
    res, _ = run_tool(ctx, "propose_cleaning_operations", {})
    print("\nData-quality findings:")
    for i in res.get("issues", []):
        print(f"  - [{'needs approval' if i['needs_approval'] else 'info'}] {i['title']}: {i['what']} → {i['recommended']}")


if __name__ == "__main__":
    main(*sys.argv[1:3])
