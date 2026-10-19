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
    "analyst": st.Page(pages.analyst_page, title="AI Analyst", icon=":material/insights:", url_path="analyst", default=True),
    "overview": st.Page(pages.overview_page, title="Dashboard", icon=":material/space_dashboard:", url_path="overview"),
    "dashboards": st.Page(pages.dashboards_page, title="My Dashboards", icon=":material/folder_open:", url_path="dashboards"),
    "explorer": st.Page(pages.explorer_page, title="Data Explorer", icon=":material/table_view:", url_path="explorer"),
    "settings": st.Page(pages.settings_page, title="Settings", icon=":material/settings:", url_path="settings"),
}
st.session_state["_pages"] = _pages
nav = st.navigation(list(_pages.values()), position="sidebar")

from src import config  # noqa: E402
from src.agent.llm import provider_status  # noqa: E402

_status = provider_status()
_ai, _ai_local = _status["enabled"], _status["local"]

with st.sidebar:
    if config.ONLINE:
        st.markdown(
            '<div class="privacy"><b>Online version.</b> Your uploads are processed on this server, visible only to your '
            f"browser session, and deleted after {config.SESSION_TTL_MINUTES // 60} hours of inactivity. "
            "Don't upload confidential data. Download dashboards to keep them."
            + (" The AI assistant only sees column summaries and a few example rows, and only if you allow it." if _ai else "")
            + "</div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="privacy">Local mode: files and dashboards are stored on this computer only. '
            + ("Nothing is sent to external services unless you allow an online AI assistant for an analysis; it then "
               "sees column summaries and a few example rows, never the whole file.</div>" if _ai and not _ai_local
               else "No data is sent to external services.</div>"),
            unsafe_allow_html=True,
        )

nav.run()
