"""End to end on the extremely messy workbook: pick a table → analyse → clean → dashboard → reopen."""
import io

import pandas as pd
import pytest

from src import structure
from src.agent import guided
from src.agent.actions import all_decided, record_cleaning_decision
from src.agent.context import AgentContext
from src.agent.pipeline import dashboard_transform
from src.agent.quality import quality_score
from src.agent.workflow import Stage
from src.chart_renderer import prepare_chart_data
from src.dashboard_service import validate_dashboard_config
from src.data_loader import load_dataset
from src.data_profiler import profile_dataset
from src.query_engine import answer_question
from src.storage import Storage
from tests.messy_workbook import build


def _analyse(storage: Storage, dataset_id: str, ref: str, goal: str) -> AgentContext:
    ctx = AgentContext.open(storage, dataset_id, ref, None)
    guided.start(ctx)
    for _ in range(25):
        m = ctx.memory
        if m.dashboard_id:
            break
        if Stage(m.stage) in (Stage.CLEANING_REVIEW, Stage.CLEANING_APPROVAL) and m.cleaning_reviewed:
            for issue in m.cleaning_issues:
                if issue.get("requires_approval") and issue["status"] == "pending":
                    record_cleaning_decision(ctx, issue["id"], True, None)  # accept every recommendation
            assert all_decided(ctx)
            guided.handle(ctx, None, "cleaning_decided")
            continue
        q = m.pending()
        guided.handle(ctx, (q.get("suggestions") or [goal])[0] if q else goal, None)
        ctx.save()
    return ctx


@pytest.fixture
def storage(tmp_path):
    return Storage(tmp_path / "u", tmp_path / "d")


@pytest.mark.parametrize("sheet,table,goal", [("Sales Report", 0, "Sales performance by region"),
                                              ("Monthly", 0, "Monthly sales trend by region")])
def test_messy_table_to_dashboard_and_back(storage, sheet, table, goal):
    book = build()
    meta = storage.save_upload("messy.xlsx", book, ".xlsx")
    raw = pd.read_excel(io.BytesIO(book), sheet_name=sheet, header=None, dtype=object)
    ref = structure.detect(raw).tables[table].ref(sheet)

    ctx = _analyse(storage, meta["id"], ref, goal)
    m = ctx.memory
    assert m.dashboard_id, f"analysis stopped at {m.stage}"
    work, wprofile = ctx.working()
    before = quality_score(ctx.original.df, ctx.original_profile)["score"]
    after = quality_score(work, wprofile)["score"]
    assert after >= before

    cfg = storage.load_dashboard(m.dashboard_id)
    assert cfg["sheet"] == ref and cfg["charts"] and cfg["kpis"]

    # Reopen exactly like the app does: rebuild the table from the reference, replay the approved clean-up.
    reloaded = load_dataset(storage.dataset_file(meta["id"]), ".xlsx", cfg["sheet"], cfg.get("header_row"))
    df = dashboard_transform(reloaded.df, cfg)
    profile = profile_dataset(df, reloaded.percent_columns, reloaded.currency_columns)
    clean, warnings = validate_dashboard_config(cfg, profile)
    assert len(clean["charts"]) == len(cfg["charts"]), warnings
    for spec in clean["charts"]:
        if spec["type"] != "table":
            data, _ = prepare_chart_data(df, spec, profile)
            assert len(data)
    if sheet == "Sales Report":
        # inconsistent spellings were merged by the approved clean-up
        assert {"central", "CENTRAL"}.isdisjoint(set(df["Region"].dropna().astype(str).str.strip()))
        assert "Total" not in set(df["Region"].dropna())
    else:
        assert str(df["Month"].dtype).startswith("datetime64") and len(df) == 30

    # natural-language analysis on the validated data
    measure = "Sales (RM)"
    out = answer_question(f"What share of {measure} does each Region contribute?", df, profile)
    assert out.ok and abs(out.table["Share %"].sum() - 100) < 0.5
