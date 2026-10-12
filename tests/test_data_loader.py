import io

import numpy as np
import pandas as pd
import pytest

from src.data_loader import (
    CSV_SHEET_NAME, DataLoadError, choose_default_sheet, detect_header_row, list_sheets, load_dataset,
    sheet_dimensions, validate_upload,
)
from tests import synthetic as syn
from tests.conftest import csv_bytes, xlsx_bytes


# ------------------------------------------------------------------ validation

@pytest.mark.parametrize("name", ["report.pdf", "data.txt", "macro.xlsm", "noext", "evil.exe"])
def test_rejects_unsupported_extensions(name):
    with pytest.raises(DataLoadError, match="Unsupported file type"):
        validate_upload(name, 100)


def test_rejects_empty_and_oversized_files():
    with pytest.raises(DataLoadError, match="empty"):
        validate_upload("a.csv", 0)
    with pytest.raises(DataLoadError, match="exceeds"):
        validate_upload("a.xlsx", 60 * 1024 * 1024, max_bytes=50 * 1024 * 1024)
    assert validate_upload("Sales.XLSX", 1000) == ".xlsx"


def test_corrupted_workbook_gives_friendly_error():
    with pytest.raises(DataLoadError, match="corrupted|not a valid"):
        load_dataset(b"PK\x03\x04garbage-not-a-zip" + b"\x00" * 100, ".xlsx")
    with pytest.raises(DataLoadError):
        load_dataset(b"\x00\x01\x02binary", ".xlsx")


def test_html_disguised_as_excel_is_rejected():
    with pytest.raises(DataLoadError, match="HTML"):
        load_dataset(b"<html><table><tr><td>1</td></tr></table></html>", ".xls")


def test_csv_mislabelled_as_xlsx_is_read_as_text():
    res = load_dataset(b"a,b\n1,2\n3,4\n", ".xlsx")
    assert list(res.df.columns) == ["a", "b"]
    assert any("content is CSV" in w for w in res.warnings)


# ------------------------------------------------------------------ CSV

def test_load_csv_basic_types():
    res = load_dataset(csv_bytes(syn.sales()), ".csv")
    df = res.df
    assert res.sheet == CSV_SHEET_NAME
    assert df.shape == (500, 8)
    assert pd.api.types.is_datetime64_any_dtype(df["Order Date"])
    assert pd.api.types.is_numeric_dtype(df["Sales"])
    assert df["Sales"].sum() == pytest.approx(syn.sales()["Sales"].sum())


def test_semicolon_delimited_csv():
    res = load_dataset(b"Name;Amount\nA;1\nB;2\nC;3\n", ".csv")
    assert list(res.df.columns) == ["Name", "Amount"]
    assert res.df["Amount"].tolist() == [1, 2, 3]


def test_ragged_csv_rows_are_padded_not_dropped():
    res = load_dataset(b"a,b,c\n1,2,3\n4,5\n6,7,8,9\n", ".csv")
    assert len(res.df) == 3


def test_latin1_csv_is_decoded():
    res = load_dataset("City,Value\nMünchen,1\nSão Paulo,2\n".encode("cp1252"), ".csv")
    assert "München" in res.df["City"].tolist()


def test_header_only_csv_errors():
    with pytest.raises(DataLoadError, match="no data rows"):
        load_dataset(b"a,b,c\n", ".csv")


def test_leading_zero_codes_are_kept_as_text():
    res = load_dataset(b"Zip,Value\n01234,1\n00501,2\n12345,3\n", ".csv")
    assert res.df["Zip"].tolist() == ["01234", "00501", "12345"]


# ------------------------------------------------------------------ Excel & worksheets

def test_multi_sheet_workbook_listing_and_selection():
    data = xlsx_bytes({"Notes": pd.DataFrame({"x": ["hello"]}), "Sales": syn.sales(), "Spend": syn.marketing()})
    assert list_sheets(data, ".xlsx") == ["Notes", "Sales", "Spend"]
    dims = sheet_dimensions(data, ".xlsx")
    assert choose_default_sheet(dims) == "Sales"
    res = load_dataset(data, ".xlsx", sheet="Spend")
    assert res.sheet == "Spend" and len(res.df) == 240
    with pytest.raises(DataLoadError, match="not found"):
        load_dataset(data, ".xlsx", sheet="Missing")


def test_empty_worksheet_errors():
    data = xlsx_bytes({"Empty": pd.DataFrame(), "Data": syn.expenses()})
    with pytest.raises(DataLoadError, match="empty"):
        load_dataset(data, ".xlsx", sheet="Empty")


def test_legacy_xls_file():
    xlwt = pytest.importorskip("xlwt")
    book = xlwt.Workbook()
    sh = book.add_sheet("Data")
    rows = [["Dept", "Amount"], ["HR", 10.5], ["IT", 20.0], ["HR", 5.0]]
    for i, r in enumerate(rows):
        for j, v in enumerate(r):
            sh.write(i, j, v)
    buf = io.BytesIO()
    book.save(buf)
    assert list_sheets(buf.getvalue(), ".xls") == ["Data"]
    res = load_dataset(buf.getvalue(), ".xls")
    assert res.df["Amount"].sum() == pytest.approx(35.5)


def test_messy_sheet_cleaning():
    data = xlsx_bytes({"Messy": pd.DataFrame(syn.messy_rows())}, header=False)
    res = load_dataset(data, ".xlsx")
    df = res.df
    text = " ".join(res.warnings)
    assert res.header_row == 2  # title + blank row skipped
    assert list(df.columns) == ["Date", "Region", "Amount ($)", "Amount ($) (2)", "Notes", "Discount %"]
    assert "Duplicate column names" in text
    assert "Total" in text and len(res.removed_rows) == 1  # TOTAL row excluded, not double counted
    assert "empty row" in text
    assert len(df) == 41  # 40 data rows + 1 duplicate; blank and total rows removed
    assert pd.api.types.is_datetime64_any_dtype(df["Date"])
    assert "day/month/year" in text
    assert pd.api.types.is_numeric_dtype(df["Amount ($)"])
    assert df["Amount ($)"].iloc[1] == pytest.approx(1037.5)  # "$1,037.50"
    assert "Discount %" in res.percent_columns
    assert df["Amount ($) (2)"].sum() == sum(10 + i for i in range(40)) + 12  # duplicate of row i=2 counted, TOTAL not


def test_detect_header_row_with_numeric_first_row():
    raw = pd.DataFrame([[1, 2], [3, 4]])
    assert detect_header_row(raw) == -1
    res = load_dataset(b"1,2\n3,4\n", ".csv")
    assert list(res.df.columns) == ["Column 1", "Column 2"]


def test_manual_header_row_override():
    data = xlsx_bytes({"S": pd.DataFrame(syn.messy_rows())}, header=False)
    res = load_dataset(data, ".xlsx", header_row=2)
    assert res.header_row == 2 and "Region" in res.df.columns


def test_non_numeric_values_are_reported_not_silently_dropped():
    vals = [str(i) for i in range(30)] + ["unknown"]
    res = load_dataset(csv_bytes(pd.DataFrame({"Qty": vals})), ".csv")
    assert pd.api.types.is_numeric_dtype(res.df["Qty"])
    assert res.df["Qty"].isna().sum() == 1
    assert any("could not be read as numbers" in w for w in res.warnings)


def test_mixed_column_kept_as_text_with_warning():
    df = pd.DataFrame({"Code": ["A1", "B2", 3, 4, "C5", "D6"] * 3})
    data = xlsx_bytes({"S": df})
    res = load_dataset(data, ".xlsx")
    assert not pd.api.types.is_numeric_dtype(res.df["Code"])
    assert any("mixes numbers and text" in w for w in res.warnings)


def test_original_data_not_lost_on_duplicates():
    df = pd.DataFrame({"a": [1, 1, 2], "b": ["x", "x", "y"]})
    res = load_dataset(csv_bytes(df), ".csv")
    assert len(res.df) == 3  # duplicates retained (reported by profiler)
    assert np.array_equal(res.df["a"].to_numpy(), [1, 1, 2])
