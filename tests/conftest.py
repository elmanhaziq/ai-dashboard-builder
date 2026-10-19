import io
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_loader import load_dataset  # noqa: E402
from src.data_profiler import profile_dataset  # noqa: E402
from tests import synthetic as syn  # noqa: E402


def xlsx_bytes(sheets: dict[str, pd.DataFrame], header: bool = True) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as xw:
        for name, df in sheets.items():
            df.to_excel(xw, sheet_name=name, index=False, header=header)
    return buf.getvalue()


def csv_bytes(df: pd.DataFrame, **kw) -> bytes:
    return df.to_csv(index=False, **kw).encode("utf-8")


def load_and_profile(df: pd.DataFrame):
    res = load_dataset(csv_bytes(df), ".csv")
    return res, profile_dataset(res.df, res.percent_columns, res.currency_columns)


@pytest.fixture
def sales_loaded():
    return load_and_profile(syn.sales())


@pytest.fixture
def marketing_loaded():
    return load_and_profile(syn.marketing())


@pytest.fixture
def storage(tmp_path):
    from src.storage import Storage
    return Storage(tmp_path / "uploads", tmp_path / "saved_dashboards")


@pytest.fixture(autouse=True)
def _isolated_ai_usage(tmp_path, monkeypatch):
    """Tests must never write to the real ai_usage.json."""
    from src.agent import usage
    monkeypatch.setattr(usage, "USAGE_FILE", tmp_path / "ai_usage_test.json")
