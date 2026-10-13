# AI Dashboard Builder (local prototype)

Upload an Excel workbook or CSV file. The app profiles the data, recommends KPIs and charts, and builds an
interactive dashboard that you can filter, customise, save and reopen. You can also ask questions in plain
English and get answers calculated with pandas from your own data.

Everything runs on your computer. No API keys, databases, Docker or cloud services are needed, and no data
leaves the machine.

---

## Quick start (Windows)

Double-click **`run.bat`** in File Explorer, or run it from a terminal in the project folder.
In **PowerShell** the `.\` prefix is required, because PowerShell doesn't run scripts from the current folder by name alone:

```powershell
cd $HOME\Documents\ai-dashboard-builder
.\run.bat
```

In Command Prompt (`cmd`), `run.bat` on its own works.

On first run, it creates a `.venv` virtual environment and installs the dependencies. It then opens
http://localhost:8501. Press `Ctrl+C` in the terminal to stop the app.

### Manual setup

Use these steps if you prefer to set things up yourself. Python 3.11 or newer is required; 3.13 is tested.

```bat
cd path\to\ai-dashboard-builder
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

In PowerShell, activate with `.venv\Scripts\Activate.ps1`. If scripts are blocked, run
`Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once, or call `.venv\Scripts\python.exe -m streamlit run app.py` directly.

### Try it with sample data

The app never inserts demo figures into your dashboards. To generate clearly synthetic test files:

```bat
.venv\Scripts\python.exe scripts\make_sample_data.py
```

This writes `sample_data\synthetic_business_workbook.xlsx` (a workbook with several sheets: sales, marketing,
expenses, inventory, and a deliberately messy sheet), plus CSV and legacy `.xls` files. Upload any of them on
the **Overview** page.

## Running the tests

```bat
.venv\Scripts\python.exe -m pytest
```

The 119 tests use only the synthetic data in `tests/synthetic.py`. They cover file loading, worksheet selection,
header detection, cleaning, profiling, type inference, KPI calculations, chart recommendations, filtering and
aggregation, natural-language questions, saving and loading dashboards, and invalid input handling.

---

## What the app does

| Step | Behaviour |
|---|---|
| **Upload** | `.xlsx`, `.xls` and `.csv` files up to 50 MB, by drag-and-drop. The file type is checked from the file's contents, not just its extension, and corrupt, encrypted or HTML-disguised files are rejected with a clear message. |
| **Worksheet** | Shows every sheet with its approximate row count and pre-selects the largest. The header row is detected automatically and can be overridden. |
| **Cleaning** | Skips title rows above the header and removes empty rows and columns. Duplicate column names get suffixes and blank headers are named automatically. Repeated header rows are removed, and `Total`/`Subtotal` rows are excluded so totals aren't double-counted. Currency, thousands separators, `%` values, `(negative)` values and mixed date formats are parsed. **Every change is listed as a warning, and nothing is dropped silently.** |
| **Profiling** | Each column gets a role: date, measure, category, identifier, free text, coordinate or empty. The profile also covers missing values, duplicates, unique counts, summary statistics, outliers and date ranges. IDs, years and row numbers are never summed. Rates, percentages, prices and scores are averaged rather than summed. |
| **Dashboard** | KPI cards (with period-over-period change when there's a valid event date), plus line, bar, donut, scatter, box, histogram, summary-table and map charts chosen by deterministic rules. Titles name the real column, for example "Sum of Amount", so a measure is never relabelled as "Revenue". |
| **Filtering** | A filter bar for date ranges and categories, plus cross-filtering: click a bar to filter the whole dashboard. Every KPI, chart, insight and answer is recalculated on the filtered rows. The uploaded data itself is never modified. |
| **Visual builder** | **Add visual**, or **⋮ → Edit visual** on any chart, opens a builder with a live preview, in the style of Power BI. Pick a visual type from the gallery (line, area, column, bar, donut, matrix heatmap, scatter, histogram, box plot, table, map), then fill the field wells: axis, values (up to 4 measures, each with its own calculation), legend, rows for a matrix, and bubble size. **Filters on this visual** apply to that chart only. **Format** covers title, sort, top-N, side-by-side / stacked / 100% stacked series, data labels, time grouping and half or full width. |
| **Chart menu (⋮)** | Edit visual, duplicate, switch between half and full width, move earlier or later, remove, and **revert to recommendation** for any auto-generated chart you changed. |
| **Fields** | Correct the auto-detection. Set any column to Measure, Category, Identifier, Text or Hidden (only types that fit the data are allowed), and set a measure's default calculation to Sum or Average. Every visual, KPI, filter and answer follows the change. Visuals that no longer fit are flagged rather than deleted, so **Reset to detected** brings them back. |
| **KPIs** | Add, remove and rename KPI cards. |
| **Persistence** | Save, rename, duplicate and delete dashboards. They reopen after a restart. |
| **AI insights** | Rule-based findings: concentration, trend, outliers (1.5 × IQR), correlation and data quality. All of them are computed with pandas. |
| **Questions** | e.g. *"Which region has the highest total sales?"*, *"Average sales for each region"*, *"Which month had the highest sales?"*, *"Top 5 products by quantity"*, *"Which columns have the most missing values?"*, *"What trends or outliers are visible?"*, *"Summarize the key findings"*, *"Total sales in North"*. If a question can't be interpreted reliably, the app asks you to rephrase instead of guessing. |
| **Data Explorer** | Search, column selection, sortable table, CSV export of the current view, per-column statistics, a cleaning log, and the original sheet as stored. |

## Project structure

```
ai-dashboard-builder/
├── app.py                    Streamlit entry point and navigation
├── run.bat                   Windows launcher (creates .venv, installs deps, starts app)
├── requirements.txt
├── .env.example              Optional settings (upload limit, row limit, data folder)
├── .streamlit/config.toml    Theme and server upload limit
├── assets/logo.svg
├── scripts/make_sample_data.py
├── src/
│   ├── config.py             Settings from environment / .env
│   ├── data_loader.py        File validation, sheet listing, header detection, cleaning, type conversion
│   ├── data_profiler.py      Column roles, statistics, data-quality profile (DatasetProfile)
│   ├── chart_generator.py    Rule-based chart & KPI recommendations; chart/KPI spec validation
│   ├── chart_renderer.py     Aggregation for charts and Plotly figure styling
│   ├── dashboard_service.py  Dashboard config, filters, KPI calculation, config validation
│   ├── insights.py           Deterministic insights (trend, outliers, concentration, correlation)
│   ├── query_engine.py       Question → QueryPlan → pandas execution
│   ├── storage.py            Local file storage with atomic writes
│   ├── formatting.py
│   └── ui/                   Streamlit pages, dashboard view, chart editor, theme
├── tests/                    pytest suite + synthetic datasets
├── uploads/                  Uploaded files (git-ignored)
└── saved_dashboards/         Dashboard JSON configs (git-ignored)
```

## Local storage and privacy

- Uploaded files are saved unchanged in `uploads\<dataset-id>\original.<ext>`, alongside a small `meta.json`.
  Each upload gets a unique ID, so files with the same name never overwrite each other. Re-uploading an
  identical file with the same name reuses the stored copy.
- Dashboards are stored as `saved_dashboards\<dashboard-id>.json`. These files hold chart, KPI and filter
  **configuration only**: no data and no code. They're validated against the dataset each time they're opened,
  and anything that no longer fits is skipped with a warning.
- Writes are atomic: data goes to a temporary file first, which then replaces the target.
- **Files are stored unencrypted**, and deletion is ordinary file deletion, not a secure wipe.
- `uploads/`, `saved_dashboards/`, `sample_data/`, `.env` and `.venv/` are listed in `.gitignore`.

### Clearing local data

- In the app: **Settings → Uploaded datasets → Delete** removes one file and its dashboards.
  **Settings → Clear all local data** removes everything.
- Manually, while the app is stopped:

```bat
rmdir /s /q uploads saved_dashboards
mkdir uploads saved_dashboards
```

## Online version (Streamlit Community Cloud)

The same code runs as a shared web app. On Streamlit Community Cloud the app switches to **online mode**
automatically; anywhere else, set `DASHBOARD_MODE=online`. Online mode changes how data is handled:

| | Local mode (your computer) | Online mode (web) |
|---|---|---|
| Where files are processed | This computer | The app's server |
| Who can see uploads and dashboards | You | Only the visitor's own browser session |
| How long they're kept | Until you delete them | Deleted after 2 hours of inactivity, when the server restarts, or via *Delete my data now* |
| Keeping your work | Saved automatically to `saved_dashboards/` | **My Dashboards → Save** downloads a `.dashboard.zip` (settings plus the original file); **Import** restores it |
| Upload / row limits | 50 MB / 1,000,000 rows | 20 MB / 300,000 rows |

In online mode, server folder paths are never shown, and the app tells visitors not to upload confidential
data. Imported `.dashboard.zip` files are treated as untrusted: only the expected entries are read, sizes are
checked before decompressing, and the settings are validated like any saved dashboard.

**To deploy your own copy:** sign in at [share.streamlit.io](https://share.streamlit.io) with GitHub, choose
**Create app → Deploy a public app from GitHub**, then pick this repository, branch `main`, main file `app.py`,
and Python 3.13 under *Advanced settings*. Optionally add `DASHBOARD_MODE = "online"` to Secrets as well.

## Configuration

Copy `.env.example` to `.env` to change any of these settings:

| Variable | Default | Meaning |
|---|---|---|
| `DASHBOARD_MAX_UPLOAD_MB` | 50 | Upload size limit. Also update `maxUploadSize` in `.streamlit/config.toml`. |
| `DASHBOARD_MAX_ROWS` | 1,000,000 | Rows loaded per sheet. Any extra rows are skipped with a warning. |
| `DASHBOARD_DATA_DIR` | project folder | Where `uploads/` and `saved_dashboards/` live (local mode). |
| `DASHBOARD_MODE` | auto | `local` or `online`. Auto-detects Streamlit Community Cloud. |
| `DASHBOARD_SESSION_TTL_MINUTES` | 120 | Online mode: minutes of inactivity before a visitor's files are deleted. |

## Adding an LLM later

`query_engine.py` separates **interpreting** a question from **calculating** the answer:

```
question → Interpreter → QueryPlan (JSON, validated against the profile) → execute_plan() (pandas)
```

An LLM-backed interpreter only needs to return a `QueryPlan`. The plan is validated (known intent, existing
columns with the right roles, a supported aggregation) and executed by the same deterministic pandas code,
so the model never produces the numbers. Column interpretation and narrative summaries can be added the same
way, using `DatasetProfile.to_dict()` as the model's input. If a provider is added, configure it through
environment variables and show a clear notice before any data is sent off the machine. Keep local-only mode
as the default.

## Known limitations

- Several values in one visual always share **one** axis. There are no dual-axis charts, because they make
  unrelated scales look comparable. The builder warns you when values differ in scale by more than 20×.
- Charts can't be freely dragged or resized on a grid; use the Earlier/Later buttons and the half or full
  width setting instead. Calculated fields (such as `Revenue − Cost`) aren't supported yet.

- Questions follow common patterns (ranking, group totals and averages, periods, trends, outliers,
  correlation, missing values, counts, simple "in &lt;value&gt;" filters). Complex multi-condition questions
  aren't supported, and the app asks you to rephrase them.
- Maps need latitude and longitude columns; place names aren't geocoded. Map tiles load from the internet,
  but the points themselves are drawn locally.
- Ambiguous dates such as `03/04/2024` are read month-first unless a value in the column shows day-first
  order. A warning is shown either way, and you can check the result in the Data Explorer.
- European number formats such as `1.234,56` aren't parsed as numbers.
- Workbooks with merged multi-row headers or several tables on one sheet are read as a single table starting
  at the detected header row. Use the header-row selector, or tidy the sheet, if the result looks wrong.
- Formulas are read using their last calculated value as stored by Excel.
- The app is a single-user local tool: there are no user accounts or access control.
