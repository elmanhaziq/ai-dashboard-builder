"""AI Dashboard Builder — Streamlit entry point.

Run with:  streamlit run app.py
"""
from __future__ import annotations

import streamlit as st

st.set_page_config(page_title="Dashboard Builder", page_icon=":material/space_dashboard:", layout="wide",
                   initial_sidebar_state="expanded")

try:  # Started with a Python that lacks the dependencies (e.g. a global `streamlit run`)?
    import openpyxl, plotly, xlrd  # noqa: E401, F401
except ImportError as _exc:
    import sys
    st.error(f"Missing package: **{_exc.name}**. The app was started with a Python environment that doesn't have "
             f"this project's dependencies:\n\n`{sys.executable}`")
    st.markdown("Close this window and start the app from the project folder with the launcher, which uses the "
                "project's own `.venv`:")
    st.code(".\\run.bat", language="powershell")
    st.markdown("Or call the project's Python directly:")
    st.code(".venv\\Scripts\\python.exe -m streamlit run app.py", language="powershell")
    st.stop()

from src.ui import pages  # noqa: E402  (set_page_config must run first)
from src.ui.theme import brand, inject_css  # noqa: E402

inject_css()
brand()

_pages = {
    "overview": st.Page(pages.overview_page, title="Overview", icon=":material/space_dashboard:", url_path="overview", default=True),
    "dashboards": st.Page(pages.dashboards_page, title="My Dashboards", icon=":material/folder_open:", url_path="dashboards"),
    "explorer": st.Page(pages.explorer_page, title="Data Explorer", icon=":material/table_view:", url_path="explorer"),
    "settings": st.Page(pages.settings_page, title="Settings", icon=":material/settings:", url_path="settings"),
}
st.session_state["_pages"] = _pages
nav = st.navigation(list(_pages.values()), position="sidebar")

with st.sidebar:
    st.markdown(
        '<div class="privacy">Local mode: files and dashboards are stored on this computer only. '
        "No data is sent to external services.</div>",
        unsafe_allow_html=True,
    )

nav.run()
