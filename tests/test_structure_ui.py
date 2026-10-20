"""The table picker as a user sees it (Streamlit AppTest)."""
from pathlib import Path

from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parent.parent

SCRIPT = """
import sys
from pathlib import Path
sys.path.insert(0, {root!r})
import streamlit as st
from src.storage import Storage
from src.ui import state
from src.ui.structure_view import table_picker
from tests.messy_workbook import build

storage = Storage(Path({tmp!r}) / "u", Path({tmp!r}) / "d")
state.get_storage = lambda: storage
if "ds" not in st.session_state:
    st.session_state["ds"] = storage.save_upload("messy.xlsx", build(), ".xlsx")["id"]
st.session_state["ref"] = table_picker(st.session_state["ds"], {sheet!r}, "t")
"""


def _app(tmp_path, sheet: str) -> AppTest:
    return AppTest.from_string(SCRIPT.format(root=str(ROOT), tmp=str(tmp_path), sheet=sheet)).run(timeout=60)


def test_picker_offers_tables_and_corrections(tmp_path):
    at = _app(tmp_path, "Sales Report")
    assert not at.exception
    radio = at.radio[0]
    assert len(radio.options) == 3  # two tables + "whole sheet"
    assert at.session_state["ref"] == "Sales Report§r4c0-r37c4h2"
    radio.set_value(1).run(timeout=60)
    assert at.session_state["ref"] == "Sales Report§r4c7-r8c8h1"
    at.radio[0].set_value("__whole__").run(timeout=60)
    assert at.session_state["ref"] == "Sales Report"


def test_picker_flip_and_unpivot_toggles(tmp_path):
    at = _app(tmp_path, "Monthly")
    assert at.session_state["ref"].endswith("h1U1")
    at.toggle(key="unp_t_0").set_value(False).run(timeout=60)
    assert at.session_state["ref"].endswith("h1")
    at.toggle(key="flip_t_0").set_value(True).run(timeout=60)
    assert "T" in at.session_state["ref"].rpartition("§")[2]


def test_plain_sheet_needs_no_picker(tmp_path):
    at = _app(tmp_path, "Clean")
    assert at.session_state["ref"] == "Clean" and not at.radio


STAGING = """
import sys
from pathlib import Path
sys.path.insert(0, {root!r})
import streamlit as st
from src.storage import Storage
from src.ui import state
from src.ui.upload_view import render_staging
from tests.messy_workbook import build

storage = Storage(Path({tmp!r}) / "u", Path({tmp!r}) / "d")
state.get_storage = lambda: storage
if "staging" not in st.session_state:
    ds = storage.save_upload("messy.xlsx", build(), ".xlsx")["id"]
    st.session_state["staging"] = {{"dataset_id": ds, "sheet": "Monthly", "header_row": None}}
render_staging(st.session_state["staging"])
"""


def test_dashboard_upload_flow_uses_the_rebuilt_table(tmp_path):
    at = AppTest.from_string(STAGING.format(root=str(ROOT), tmp=str(tmp_path))).run(timeout=90)
    assert not at.exception
    staging = at.session_state["staging"]
    assert staging["sheet"].startswith("Monthly§") and staging["header_row"] is None
    assert not [s for s in at.selectbox if s.label == "Header row"]  # the rebuilt table knows its header
    assert any("Month" in str(m.value) for m in at.metric) or at.dataframe
