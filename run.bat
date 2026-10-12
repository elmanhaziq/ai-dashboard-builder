@echo off
REM Launch the AI Dashboard Builder. Creates .venv and installs dependencies on first run.
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo Creating virtual environment in .venv ...
    py -3 -m venv .venv 2>nul || python -m venv .venv
    if errorlevel 1 (
        echo Could not create a virtual environment. Install Python 3.11+ from https://www.python.org/ and try again.
        pause
        exit /b 1
    )
)

".venv\Scripts\python.exe" -c "import streamlit, pandas, plotly, openpyxl, xlrd" 2>nul
if errorlevel 1 (
    echo Installing dependencies ...
    ".venv\Scripts\python.exe" -m pip install --upgrade pip
    ".venv\Scripts\python.exe" -m pip install -r requirements.txt
    if errorlevel 1 (
        echo Dependency installation failed. See the messages above.
        pause
        exit /b 1
    )
)

echo Starting Dashboard Builder at http://localhost:8501  (press Ctrl+C to stop)
".venv\Scripts\python.exe" -m streamlit run app.py --server.port 8501 --server.headless false
