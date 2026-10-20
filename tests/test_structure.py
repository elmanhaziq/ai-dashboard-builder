"""Spreadsheet structure detection and table reconstruction on messy layouts."""
import io

import pandas as pd
import pytest

from src import structure
from src.data_loader import DataLoadError, load_dataset
from tests.messy_workbook import build


@pytest.fixture(scope="module")
def book() -> bytes:
    return build()


def _raw(book, sheet):
    return pd.read_excel(io.BytesIO(book), sheet_name=sheet, header=None, dtype=object)


def test_two_tables_title_notes_and_merged_header(book):
    s = structure.detect(_raw(book, "Sales Report"))
    assert not s.simple and len(s.tables) == 2
    sales, targets = s.tables
    assert sales.range == "A5:E38" and sales.header_rows == 2
    assert sales.columns == ["Order ID", "Order Date", "Region", "Sales (RM)", "Units"]
    assert "Sales Report 2024" in sales.title
    assert targets.range == "H5:I9" and targets.title == "Targets" and targets.columns == ["Region", "Target (RM)"]
    assert any("SST" in n for n in s.notes)
    assert 0 < sales.confidence <= 1 and sales.checks["header"][1]


def test_reconstructed_sales_table_loads_and_cleans(book):
    t = structure.detect(_raw(book, "Sales Report")).tables[0]
    r = load_dataset(book, ".xlsx", t.ref("Sales Report"))
    df = r.df
    assert df.shape == (30, 5)  # blank row and Total row removed
    assert str(df["Order Date"].dtype).startswith("datetime64") and df["Order Date"].notna().all()
    assert pd.api.types.is_numeric_dtype(df["Sales (RM)"]) and df["Sales (RM)"].isna().sum() == 1
    assert any("Combined 2 header rows" in w for w in r.warnings)
    assert r.sheet == t.ref("Sales Report")


def test_wide_months_become_rows_with_dates(book):
    t = structure.detect(_raw(book, "Monthly")).tables[0]
    assert t.unpivot == 1 and t.period_kind == "month"
    df = load_dataset(book, ".xlsx", t.ref("Monthly")).df
    assert list(df.columns) == ["Region", "Month", "Sales (RM)"]
    assert len(df) == 30 and df["Month"].min() == pd.Timestamp("2024-01-01")
    assert df.groupby("Region")["Sales (RM)"].sum()["Central"] == sum(1000 + 50 * m for m in range(6))


def test_sideways_table_is_flipped(book):
    t = structure.detect(_raw(book, "Branch Profile")).tables[0]
    assert t.transpose
    df = load_dataset(book, ".xlsx", t.ref("Branch Profile")).df
    assert list(df["Name"]) == ["KL Central", "Penang", "Johor Bahru", "Kuching"]
    assert str(df["Opened"].dtype).startswith("datetime64") and df["Staff"].sum() == 110


def test_plain_table_is_simple_and_unchanged(book):
    s = structure.detect(_raw(book, "Clean"))
    assert s.simple
    plain = load_dataset(book, ".xlsx", "Clean").df
    via_ref = load_dataset(book, ".xlsx", s.tables[0].ref("Clean")).df
    pd.testing.assert_frame_equal(plain, via_ref)


def test_stacked_tables_in_csv_and_repeated_sub_headers():
    csv = ("Quarterly report,,,,\n,,,,\n,Q1,,Q2,\nStore,Sales,Units,Sales,Units\nA,10,1,12,2\nB,20,2,22,3\n,,,,\n"
           "Staff list,,,,\nName,Role,Store,,\nAli,Manager,A,,\nSiti,Cashier,B,,\n")
    raw = pd.read_csv(io.StringIO(csv), header=None, dtype=object)
    s = structure.detect(raw)
    assert len(s.tables) == 2
    sales, staff = s.tables
    assert sales.header_rows == 2 and sales.columns == ["Store", "Q1 Sales", "Q1 Units", "Q2 Sales", "Q2 Units"]
    assert staff.title == "Staff list" and staff.columns == ["Name", "Role", "Store"]
    df = load_dataset(csv.encode(), ".csv", staff.ref("CSV data")).df
    assert list(df["Name"]) == ["Ali", "Siti"]


def test_year_and_quarter_columns_unpivot():
    raw = pd.DataFrame([["Product", 2022, 2023, 2024], ["Tea", 5, 6, 7], ["Coffee", 8, 9, 10]], dtype=object)
    t = structure.detect(raw).tables[0]
    assert t.unpivot == 1 and t.period_kind == "year"
    grid, _ = structure.reconstruct(raw, t.spec)
    assert list(grid.iloc[0]) == ["Product", "Year", "Value"] and len(grid) == 7
    raw_q = pd.DataFrame([["Region", "Q1", "Q2", "Q3", "Q4"], ["N", 1, 2, 3, 4], ["S", 5, 6, 7, 8]], dtype=object)
    assert structure.detect(raw_q).tables[0].period_kind == "quarter"


def test_sheet_reference_roundtrip_and_bad_reference(book):
    spec = {"top": 4, "left": 0, "bottom": 37, "right": 4, "header_rows": 2, "transpose": False, "unpivot": None}
    ref = structure.make_ref("Sales Report", spec)
    assert structure.split_ref(ref) == ("Sales Report", spec)
    assert structure.split_ref("Plain § name") == ("Plain § name", None)
    assert structure.display_sheet(ref) == "Sales Report · table A5:E38"
    with pytest.raises(DataLoadError):
        load_dataset(book, ".xlsx", structure.make_ref("Clean", dict(spec, bottom=500)))
