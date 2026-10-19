# AI Dashboard Builder (local prototype)

Upload an Excel workbook or CSV file and work with an **AI data analyst**. It reads your data, asks what you
want to understand, asks about columns it can't interpret (such as `UAG_2`), explains data-quality problems in
plain language, waits for your approval before cleaning anything, then builds an interactive dashboard. You can
keep chatting with it afterwards. You can also build and customise dashboards yourself, Power BI style.

The analyst can use a **free** AI: open-source models through **Groq** or **OpenRouter**, **Google Gemini**
(AI Studio key), or **Ollama** running on your own computer. Paid **Claude** also works. You enable one with a
key in `.env` and allow it for each analysis; otherwise a built-in **guided assistant** runs the same workflow
offline. Either way, every number, cleaning step and chart is calculated by the app's own pandas engine. No
database, Docker or cloud service is needed.

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

The 154 tests use only the synthetic data in `tests/synthetic.py`. They cover file loading, worksheet selection,
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
| **Sharing** | **Manage → Download as web page (.html)** saves one file that opens in any browser, offline, with no installation: KPIs, interactive charts (hover, zoom) and insights, using the current filters. It's a snapshot, so filters can't be changed in it. **Manage → Download to edit later (.zip)** saves the settings plus the original file; import it under My Dashboards to keep editing. |
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
│   ├── report_export.py      Standalone HTML export
│   ├── agent/                AI Analyst: workflow, tools, LLM loop, guided assistant, memory, cleaning, planner
│   └── ui/                   Streamlit pages (AI Analyst, dashboard, builder, explorer, settings), theme
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
| `DASHBOARD_LLM_PROVIDER` | auto | `auto`, `groq`, `gemini`, `openrouter`, `ollama`, `openai_compatible`, `anthropic`, or `none` (always guided). `auto` uses the first key it finds, in the order Anthropic → Gemini → Groq → OpenRouter. Ollama is only used when chosen explicitly. |
| `GROQ_API_KEY` | (unset) | Free key from console.groq.com: fast hosted open-source models. Default model `llama-3.3-70b-versatile`. |
| `GEMINI_API_KEY` | (unset) | Free key from aistudio.google.com. Default model `gemini-flash-latest`. On the free tier Google may use what you send to improve its products. |
| `OPENROUTER_API_KEY` | (unset) | Key from openrouter.ai. Default model `meta-llama/llama-3.3-70b-instruct:free`. |
| `ANTHROPIC_API_KEY` | (unset) | Paid Claude. Default model `claude-opus-5-5`. |
| `DASHBOARD_LLM_MODEL` | per provider | Override the model, e.g. `qwen3:4b` for Ollama. Free model names change, so check the provider's list if you get a "doesn't offer the model" message. |
| `DASHBOARD_LLM_BASE_URL` | per provider | Endpoint for `openai_compatible` (LM Studio, vLLM…) or a non-default Ollama address. |
| `DASHBOARD_LLM_API_KEY` | (unset) | Key for `openai_compatible`. |
| `DASHBOARD_OLLAMA_NUM_CTX` | 16384 | Ollama context window. |
| `DASHBOARD_LLM_EFFORT` | `medium` | Claude only: reasoning effort. |

On Streamlit Community Cloud, put the key (e.g. `GROQ_API_KEY = "..."`) in the app's **Secrets** instead of `.env`.
Anyone visiting the public site who allows the AI will then use your key and its free-tier limits.

### Which free AI should I use?

| Option | Cost | Speed | Privacy | Notes |
|---|---|---|---|---|
| **Groq** (open-source Llama models) | Free tier | Fast | Data goes to Groq | Best balance of free, open source and speed |
| **Google Gemini** (AI Studio key) | Free tier | Fast | Data goes to Google; free-tier data may be used to improve Google's products | Not open source |
| **OpenRouter** `:free` models | Free, rate-limited | Varies | Data goes to OpenRouter and the model host | Free model list changes often |
| **Ollama** on your computer | Free | Needs a graphics card. On a laptop CPU (tested on an i5-1135G7 with `qwen3:4b`), one step took 2–8 minutes. | Nothing leaves the computer | Best for privacy; practical only on a strong machine |

The AI only plans and explains. Calculations always run locally in pandas, so a free model can't produce wrong
totals; it can only ask less polished questions. Small models are more likely to misuse tools, and the app
rejects and reports those calls.

## AI Analyst (agentic workflow)

The **AI Analyst** page is the starting screen. You upload a file, answer a few questions, approve clean-up
suggestions, and receive a dashboard. A progress bar shows where you are:

```
UPLOAD → PROFILE → OBJECTIVE → CLARIFICATION → CLEANING_REVIEW → CLEANING_APPROVAL
       → DASHBOARD_PLANNING → PLAN_VALIDATION → DASHBOARD_GENERATION → READY
```

**Division of labour.** The LLM decides what to do next, asks questions, and explains. Python profiles,
calculates, cleans, validates and draws:

| Layer | Responsibility |
|---|---|
| `src/agent/orchestrator.py` | Tool-use loop with Claude (`claude-opus-5-5` via the official `anthropic` SDK). It offers only the tools allowed at the current stage, allows at most 12 tool calls per message, stops whenever the user must answer, keeps an append-only transcript, and repairs an interrupted turn. If Claude is unavailable, the guided assistant takes over for that turn. |
| `src/agent/tools.py` | The 18 tools Claude can call: `inspect_dataset`, `get_column_profile`, `get_sample_records`, `identify_ambiguous_columns`, `ask_user_clarification`, `set_business_objective`, `save_column_interpretation`, `go_to_stage`, `propose_cleaning_operations`, `preview_cleaning_operations`, `request_cleaning_approval`, `apply_approved_transformations`, `suggest_dashboard_plan`, `create_dashboard_plan`, `validate_dashboard_plan`, `execute_dashboard_plan`, `answer_data_question`, `update_dashboard`. Each has a strict JSON schema; inputs are validated again in Python and the stage is checked before anything runs. No tool executes model-written code. |
| `src/agent/workflow.py` | The state machine. A stage can only be entered when its requirements are met (goal recorded, unclear columns resolved, every clean-up proposal decided and applied, plan validated…). Going back to an earlier stage is allowed. |
| `src/agent/actions.py` | Things **only the user** can do from the UI, such as approving or rejecting a clean-up step. They are deliberately not tools, so the model can never approve changes itself. |
| `src/agent/ambiguity.py`, `quality.py` | Deterministic detection of unclear column names and data-quality issues, described in plain language. |
| `src/agent/cleaning.py`, `pipeline.py` | Whitelisted clean-up operations: keep, fill with "Unknown", fill with median or mean, leave rows out, fix spelling and capitalisation, remove or flag duplicates. They run on a copy, never fill missing numbers with zero, and never delete unusual records automatically. |
| `src/agent/planner.py` | Builds a plan around your goal, then validates it, whether it came from Claude or the rules. Validation rejects sums of average-type measures, IDs used as measures, columns you excluded, and titles that state numbers. It then builds the dashboard with the existing chart engine. |
| `src/agent/guided.py` | The offline assistant: the same tools and checks, driven by fixed rules. |
| `src/agent/memory.py` | Per-dataset memory stored as JSON next to the upload: goal, column meanings, questions and answers, clean-up decisions, change log, conversation and stage. Reopening a dataset restores all of it, and answered questions are never asked again. |

**What Claude receives.** Only after you tick the consent box when starting an analysis: column names, summary
statistics, the most common values per column, at most 5 example rows (values cut to 40 characters), your
messages, and the results of the tools above. The full file is never sent, and Claude is told to quote only
numbers that came from tool results. Without consent, or without an API key, nothing is sent to any AI service.

**Dashboards built by the analyst** store the approved clean-up steps, friendly column names and column
meanings in their settings. Those are replayed on a copy of the original file whenever the dashboard opens, so
it looks the same after a restart. All the manual tools (Fields, visual builder, filters, exports) work on them
as usual, and **Back to AI analyst** returns to the conversation.

**Benchmark.** `scripts\benchmark_agent.py <file>` prints what the analyst would ask and propose for any
spreadsheet, without changing your library. `tests/synthetic.py: malaysian_sales_rows()` is a deliberately messy
test report with cryptic codes, RM text, mixed date formats, inconsistent labels, duplicates and subtotal rows.

## Known limitations

- **The Claude integration was tested with a scripted fake model, not the live API**: no API key was available
  while it was built. The request format follows Anthropic's SDK documentation and is covered by tests, but run
  a real analysis after adding your key and report anything odd.
- The guided assistant understands answers through keywords. It handles suggested answers and short
  descriptions well, but long, nuanced replies work better with Claude.
- Unclear-name detection is tuned for English column names and common business abbreviations.
- "Leave rows with missing values out" applies to the whole dashboard, not to a single chart.

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
