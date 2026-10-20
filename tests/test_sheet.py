"""Which cells the spreadsheet view highlights for each clean-up suggestion."""
import numpy as np
import pandas as pd

from src.agent.quality import detect_issues, issue_cells
from src.data_profiler import profile_dataset
from src.ui.sheet_view import _letters


def _messy() -> pd.DataFrame:
    rng = np.random.default_rng(3)
    n = 60
    df = pd.DataFrame({
        "Region": rng.choice(["North", "South", "East"], n),
        "Sales": rng.normal(100, 10, n).round(2),
        "Order ID": np.arange(1000, 1000 + n),
    })
    df.loc[[3, 7], "Sales"] = np.nan
    df.loc[[5, 9], "Region"] = ["north", "NORTH "]
    df.loc[11, "Sales"] = 5000.0
    return pd.concat([df, df.iloc[[0]]], ignore_index=True)  # one exact duplicate row


def test_issue_cells_point_at_the_right_cells():
    df = _messy()
    profile = profile_dataset(df)
    issues = detect_issues(df, profile, [])
    cells = {c["kind"]: c for c in issue_cells(df, issues)}
    assert cells["missing"]["columns"] == ["Sales"] and set(df.index[cells["missing"]["rows"]]) == {3, 7}
    assert set(df.index[cells["inconsistent_labels"]["rows"]]) == {5, 9}
    assert cells["duplicates"]["columns"] is None and list(df.index[cells["duplicates"]["rows"]]) == [60]
    assert 11 in set(df.index[cells["outliers"]["rows"]])
    assert all(c["count"] == int(c["rows"].sum()) for c in cells.values())


def test_issue_cells_skips_file_level_notes_and_unknown_columns():
    df = _messy()
    issues = [{"id": "x", "kind": "subtotals", "column": None, "title": "Totals"},
              {"id": "y", "kind": "missing", "column": "Gone", "title": "Missing in Gone"}]
    assert issue_cells(df, issues) == []


def test_excel_letters():
    assert _letters(28)[:3] == ["A", "B", "C"] and _letters(28)[25:] == ["Z", "AA", "AB"]


def test_grid_colours_problem_rows_only():
    from src.ui.sheet_view import COLORS, _grid

    df = _messy()
    profile = profile_dataset(df)
    cells = issue_cells(df, detect_issues(df, profile, []))
    styler, shown, total = _grid(df, profile, cells, True)
    html = styler.to_html()
    assert shown == total < len(df)
    assert COLORS["missing"] in html and COLORS["duplicates"] in html
    assert "A · Region" in styler.data.columns[0] and "⚠" in styler.data.columns[1]
    assert list(styler.data.index[:2]) == [4, 6]  # row numbers count from 1
    assert "(empty)" in html and "None" not in html.replace("NoneType", "")
