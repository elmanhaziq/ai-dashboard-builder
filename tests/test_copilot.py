"""Dashboard assistant (Dash), colour themes and learned knowledge."""
import json

import pytest

from src import themes
from src.agent import knowledge
from src.agent.copilot import Copilot, new_state
from src.agent.llm import LLMUnavailable
from src.chart_renderer import build_figure, prepare_chart_data
from src.dashboard_service import build_default_dashboard, validate_dashboard_config
from tests.test_agent import FakeLLM


@pytest.fixture
def dash(sales_loaded, tmp_path):
    res, profile = sales_loaded
    cfg = build_default_dashboard(profile, res.df, "a" * 12, "sales.csv", "CSV data")
    return cfg, res.df, profile, tmp_path / "kn"


def _copilot(dash, llm=None, state=None):
    cfg, df, profile, kdir = dash
    return Copilot(cfg, df, profile, state if state is not None else new_state(), kdir, llm=llm)


# ---------------------------------------------------------------- themes
def test_clean_theme_rejects_bad_colours_and_keeps_good_ones():
    t = themes.clean_theme({"preset": "Ocean", "accent": "red; background:url(x)", "palette": ["#AABBCC", "nope", "#112233"],
                            "evil": "<script>"})
    assert t["accent"] == themes.PRESETS["Ocean"]["accent"]
    assert t["palette"] == ["#aabbcc", "#112233"]
    assert "evil" not in t
    assert themes.clean_theme("Ocean") is None
    assert themes.clean_theme({"palette": ["#000000"]})["palette"] == themes.PRESETS["Classic"]["palette"]


def test_dark_theme_gets_readable_text():
    r = themes.resolve(themes.from_preset("Midnight"))
    assert r["dark"] and themes.luminance(r["ink"]) > 0.7
    assert not themes.resolve(None)["dark"]


def test_theme_flows_into_charts_and_saved_config(dash):
    cfg, df, profile, _ = dash
    cfg["theme"] = themes.from_preset("Sunset")
    spec = next(c for c in cfg["charts"] if c["type"] in ("bar", "hbar") and not c.get("color"))
    agg, _ = prepare_chart_data(df, spec, profile)
    fig = build_figure(agg, spec, profile, theme=cfg["theme"])
    assert fig.data[0].marker.color == themes.PRESETS["Sunset"]["palette"][0]
    fig = build_figure(agg, dict(spec, color_hex="#123456"), profile, theme=cfg["theme"])
    assert fig.data[0].marker.color == "#123456"
    dark = build_figure(agg, spec, profile, theme=themes.from_preset("Midnight"))
    assert dark.layout.paper_bgcolor == themes.PRESETS["Midnight"]["card_bg"]
    clean, _ = validate_dashboard_config(json.loads(json.dumps(cfg)), profile)
    assert clean["theme"]["preset"] == "Sunset"


# ---------------------------------------------------------------- knowledge
def test_knowledge_remembers_aliases_and_reuses_for_similar_files(tmp_path):
    cols = ["Order Date", "Region", "Product Category", "Net Sales (RM)", "Qty"]
    assert knowledge.remember(tmp_path, cols, "", term="dept", column="Product Category").startswith("Remembered")
    assert knowledge.remember(tmp_path, cols, "", term="x", column="Nope").startswith("Not saved")
    knowledge.remember(tmp_path, cols, "Net Sales is in Malaysian ringgit", dataset_name="Jan.xlsx")
    knowledge.remember(tmp_path, cols, "Prefers horizontal bars", scope="general")
    kn = knowledge.for_dataset(tmp_path, cols)
    assert kn["aliases"] == {"dept": "Product Category"} and kn["notes"] and kn["general"]
    assert knowledge.engine_aliases(kn) == {"Product Category": ["dept"]}
    # next month's file: same layout plus one new column → reused as a similar file
    feb = cols + ["Promo Code"]
    kn2 = knowledge.for_dataset(tmp_path, feb)
    assert kn2["from_similar_file"] and kn2["notes"] == ["Net Sales is in Malaysian ringgit"]
    knowledge.adopt_similar(tmp_path, feb)
    assert not knowledge.for_dataset(tmp_path, feb)["from_similar_file"]
    # unrelated data gets only the general preferences
    other = knowledge.for_dataset(tmp_path, ["Patient", "Ward", "Admitted"])
    assert not other["notes"] and other["general"] == ["Prefers horizontal bars"]
    knowledge.forget(tmp_path, cols, "aliases", "dept")
    assert knowledge.for_dataset(tmp_path, cols)["aliases"] == {}


def test_knowledge_strips_control_characters_and_dedupes(tmp_path):
    cols = ["A", "B"]
    knowledge.remember(tmp_path, cols, "line one\nline two\x00")
    assert knowledge.remember(tmp_path, cols, "LINE ONE LINE TWO") == "Already remembered."
    assert knowledge.for_dataset(tmp_path, cols)["notes"] == ["line one line two"]


# ---------------------------------------------------------------- guided assistant
def test_guided_answers_with_pandas(dash):
    cp = _copilot(dash)
    cp.greet(ai=False)
    assert cp.ask("Which region has the highest total sales?") == "guided"
    msg = cp.state["messages"][-1]
    assert msg["role"] == "assistant" and "**North** has the highest" in msg["text"]
    assert msg["steps"][0].startswith("Calculated with pandas")
    assert cp.state["mood"] == "aha"


def test_guided_learns_word_meanings(dash):
    cp = _copilot(dash)
    cp.ask("area means Region")
    assert "Remembered" in cp.state["messages"][-1]["text"]
    cp.ask("total sales by area")
    assert "'Region' values" in cp.state["messages"][-1]["text"]


def test_guided_colours_and_ideas(dash):
    cfg = dash[0]
    cp = _copilot(dash)
    cp.ask("make the colours dark mode please")
    assert cfg["theme"]["preset"] == "Midnight" and cp.changed and cp.state["mood"] == "celebrate"
    n = len(cfg["charts"])
    cp.ask("give me some ideas")
    ids = cp.state["messages"][-1]["suggestions"]
    assert ids and len(cfg["charts"]) == n  # ideas are only drafts until the user adds one
    assert "Added" in cp.add_suggestion(ids[0])
    assert len(cfg["charts"]) == n + 1
    assert cp.add_suggestion(ids[0]) == "That idea is no longer available."


# ---------------------------------------------------------------- AI assistant
def test_llm_tools_calculate_draft_and_recolour(dash):
    cfg, _, _, kdir = dash
    n = len(cfg["charts"])
    llm = FakeLLM([
        {"tools": [("query_data", {"intent": "rank", "metric": "Sales", "metric2": "", "group_by": "Region",
                                   "aggregation": "sum", "top_n": 3, "order": "desc", "date_column": "", "time_grain": "",
                                   "filters": [], "group_by_2": "", "compare_values": []}),
                   ("propose_chart", {"chart": {"type": "box", "x": "Region", "y": "Unit Price", "aggregation": "", "color": "",
                                                "x2": "", "title": "Price spread by region", "why": ""},
                                      "reason": "Shows price differences", "add_now": False}),
                   ("set_theme", {"preset": "Forest", "accent": "", "page_bg": "", "card_bg": "", "palette": []}),
                   ("remember", {"note": "", "scope": "dataset", "term": "area", "column": "Region"})]},
        {"text": "West sells the most. I drafted a chart idea and made it green."},
    ])
    cp = _copilot(dash, llm)
    assert cp.ask("who sells most? make it green, and 'area' means region") == "llm"
    msg = cp.state["messages"][-1]
    assert msg["text"].startswith("West")
    assert len(msg["steps"]) == 4 and msg["steps"][0].startswith("Calculated")
    assert len(msg["suggestions"]) == 1 and len(cfg["charts"]) == n  # drafted, not added
    assert cfg["theme"]["preset"] == "Forest"
    assert knowledge.for_dataset(kdir, list(dash[2].columns))["aliases"] == {"area": "Region"}
    # tool results went back to the model with the data it needs; the knowledge block is in the user message
    sent = llm.calls[1]["messages"]
    results = json.loads(sent[-1]["content"][0]["content"])
    assert results["answered"] and results["table"]
    assert "<knowledge>" in sent[0]["content"][0]["text"]
    assert cp.state["mood"] == "celebrate"


def test_llm_bad_tool_input_is_reported_not_crashing(dash):
    llm = FakeLLM([
        {"tools": [("set_theme", {"preset": "", "accent": "javascript:alert(1)", "page_bg": "", "card_bg": "", "palette": []}),
                   ("propose_chart", {"chart": {"type": "bar", "x": "No Such Column", "y": "Sales", "aggregation": "sum",
                                                "color": "", "x2": "", "title": "", "why": ""}, "reason": "", "add_now": True}),
                   ("delete_everything", {})]},
        {"text": "Sorry, I couldn't do that."},
    ])
    cp = _copilot(dash, llm)
    cp.ask("do weird things")
    results = llm.calls[1]["messages"][-1]["content"]
    assert all(r.get("is_error") for r in results)
    assert dash[0].get("theme") is None


def test_llm_unavailable_falls_back_to_guided(dash):
    class Down:
        name = "down"

        def create(self, *a, **k):
            raise LLMUnavailable("quota used up")

    cp = _copilot(dash, Down())
    assert cp.ask("total sales") == "guided"
    texts = [m["text"] for m in cp.state["messages"] if m["role"] == "assistant"]
    assert "quota used up" in texts[0] and len(texts) == 2


def test_transcript_is_trimmed_at_user_turns(dash):
    state = new_state()
    cp = _copilot(dash, FakeLLM([{"text": "ok"}] * 80), state)
    for i in range(40):
        cp.ask(f"question {i}")
    t = state["transcript"]
    assert len(t) <= 62 and t[0]["role"] == "user" and t[0]["content"][0]["type"] == "text"
