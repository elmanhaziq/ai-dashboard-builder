"""Agentic analyst: workflow, tools, guided agent, LLM orchestrator (with a scripted fake model), persistence."""
import io
import json

import pandas as pd
import pytest

from src.agent import guided
from src.agent.actions import all_decided, record_cleaning_decision
from src.agent.ambiguity import find_ambiguous, readable
from src.agent.cleaning import CleaningError, apply_ops, validate_op
from src.agent.context import AgentContext
from src.agent.llm import AnthropicLLM, LLMReply, LLMUnavailable
from src.agent.orchestrator import Orchestrator, run_turn
from src.agent.pipeline import dashboard_transform
from src.agent.planner import validate_plan
from src.agent.tools import TOOLS, run_tool, validate_args, ToolError
from src.agent.workflow import Stage, WorkflowError, go_back, tools_for
from src.dashboard_service import validate_dashboard_config
from src.chart_renderer import prepare_chart_data
from src.data_loader import load_dataset
from src.data_profiler import apply_overrides, profile_dataset
from src.storage import Storage
from tests import synthetic as syn


# ------------------------------------------------------------------ fixtures

def _xlsx(rows) -> bytes:
    buf = io.BytesIO()
    pd.DataFrame(rows).to_excel(buf, index=False, header=False)
    return buf.getvalue()


@pytest.fixture
def storage(tmp_path):
    return Storage(tmp_path / "uploads", tmp_path / "dash")


@pytest.fixture
def mctx(storage):
    """Context on the difficult Malaysian benchmark report."""
    data = _xlsx(syn.malaysian_sales_rows())
    meta = storage.save_upload("malaysia_report.xlsx", data, ".xlsx")
    ctx = AgentContext.open(storage, meta["id"], "Sheet1")
    ctx._raw_bytes = data
    return ctx


def answer_all(ctx, answers: dict, default="A money amount (RM)", limit=10):
    for _ in range(limit):
        q = ctx.memory.pending()
        if not q:
            return
        run_turn(ctx, None, answers.get(q["column"] or q["kind"], default))


def guided_to_ready(ctx):
    run_turn(ctx, None, event="start")
    answer_all(ctx, {"objective": "Sales performance by branch and region", "UAG_2": "Ultimate Accessories Group 2 sales",
                     "REV_NET": "yes", "QTY_SLD": "yes"})
    for i in ctx.memory.cleaning_issues:
        if i["requires_approval"]:
            record_cleaning_decision(ctx, i["id"], approve=True)
    run_turn(ctx, None, event="cleaning_decided")


# ------------------------------------------------------------------ loading the benchmark

def test_benchmark_loads_cleanly(mctx):
    r = mctx.original
    df = r.df
    assert len(r.removed_rows) == 2  # Subtotal + Grand Total, even though the label is in the 4th column
    assert len(df) == 162  # 160 transactions + 2 exact duplicates (kept until the user decides)
    d = df["DT_TRX"]
    assert d.notna().all() and d.min() == pd.Timestamp("2024-01-01") and d.max() <= pd.Timestamp("2024-06-30")
    assert pd.api.types.is_numeric_dtype(df["REV_NET"]) and "REV_NET" in r.currency_columns
    assert "Disc %" in r.percent_columns


def test_iso_and_day_first_dates_mixed_correctly():
    csv = "Date,Amount\n13/02/2024,1\n2024-03-05,2\n05/04/2024,3\n2024-11-02,4\n".encode()
    df = load_dataset(csv, ".csv").df
    assert list(df["Date"].dt.strftime("%Y-%m-%d")) == ["2024-02-13", "2024-03-05", "2024-04-05", "2024-11-02"]


# ------------------------------------------------------------------ ambiguity

def test_ambiguous_columns_detected_meaningfully(mctx):
    amb = {a["column"]: a for a in mctx.memory.ambiguous}
    assert amb["UAG_2"]["blocking"] and amb["UAG_2"]["priority"] == 3
    assert "REV_NET" in amb and "QTY_SLD" in amb
    for clear in ("DT_TRX", "Region", "Sales Rep", "TRX_ID", "Disc %"):
        assert clear not in amb, clear
    assert list(amb)[0] == "UAG_2"  # most important first
    assert readable("revenue net") == "Net revenue"


# ------------------------------------------------------------------ workflow & tool gating

def test_stages_cannot_be_skipped(mctx):
    assert mctx.memory.stage == Stage.OBJECTIVE.value
    for name in ("apply_approved_transformations", "execute_dashboard_plan", "create_dashboard_plan", "answer_data_question"):
        result, err = run_tool(mctx, name, {})
        assert err and "isn't available" in result["error"]
    assert "execute_dashboard_plan" not in tools_for(Stage.CLARIFICATION)
    assert "apply_approved_transformations" in tools_for(Stage.CLEANING_APPROVAL)


def test_objective_must_come_from_user(mctx):
    result, err = run_tool(mctx, "set_business_objective", {"objective": "sales", "category": "sales"})
    assert err and "must come from the user" in result["error"]


def test_tool_inputs_strictly_validated(mctx):
    with pytest.raises(ToolError):
        validate_args(TOOLS["get_column_profile"].schema, {"column": "Region", "code": "import os"})
    with pytest.raises(ToolError):
        validate_args(TOOLS["get_sample_records"].schema, {"count": "5"})
    result, err = run_tool(mctx, "no_such_tool", {})
    assert err
    for tool in TOOLS.values():
        d = tool.api_definition()
        assert d["strict"] and d["input_schema"]["additionalProperties"] is False


def test_go_back_rules(mctx):
    guided_to_ready(mctx)
    go_back(mctx.memory, Stage.CLARIFICATION, "user changed a definition")
    assert mctx.memory.stage == Stage.CLARIFICATION.value and mctx.memory.plan is None
    with pytest.raises(WorkflowError):
        go_back(mctx.memory, Stage.READY, "forward")
    with pytest.raises(WorkflowError):
        go_back(mctx.memory, Stage.PROFILE, "too far")


# ------------------------------------------------------------------ guided end-to-end on the benchmark

def test_guided_flow_end_to_end(mctx, storage):
    guided_to_ready(mctx)
    m = mctx.memory
    assert m.stage == Stage.READY.value and m.dashboard_id
    kinds = [(q["column"], q["kind"]) for q in m.questions]
    assert kinds[0] == (None, "objective") and ("UAG_2", "meaning") in kinds
    assert len(kinds) == len(set(kinds)), "a question was asked twice"
    # Answers saved as confirmed interpretations
    it = m.interpretations["UAG_2"]
    assert it["confirmed"] and it["source"] == "user" and it["unit"] == "RM" and it["aggregation"] == "sum"
    assert m.interpretations["QTY_SLD"]["unit"] == "units"  # "quantity sold" is a count, no follow-up needed
    assert not any(q["column"] == "QTY_SLD" and q["kind"] == "unit" for q in m.questions)
    # Dashboard uses confirmed names and the user's goal (branch/region)
    cfg = storage.load_dashboard(m.dashboard_id)
    text = json.dumps(cfg["charts"])
    assert "Ultimate Accessories Group 2 sales" in text and "Net revenue" in text and "Region" in text and "BR_CD" in text
    assert "UAG_2" not in [c.get("y") for c in cfg["charts"]]
    assert {o["op"] for o in cfg["cleaning_ops"]} == {"normalize_text", "fill_unknown", "remove_duplicates"}
    assert cfg["renames"]["REV_NET"] == "Net revenue" and cfg["analysis"]["objective"].startswith("Sales performance")


def test_cleaning_requires_user_approval_and_preserves_original(mctx, storage):
    run_turn(mctx, None, event="start")
    answer_all(mctx, {"objective": "Sales by region", "UAG_2": "Accessories sales", "REV_NET": "yes", "QTY_SLD": "yes"})
    m = mctx.memory
    assert m.stage == Stage.CLEANING_APPROVAL.value and not all_decided(mctx)
    result, err = run_tool(mctx, "apply_approved_transformations", {})
    assert err and "Only the user can approve" in result["error"]
    # The model-facing tool set contains no way to approve
    assert not any(n.startswith("approve") or "decision" in n for n in TOOLS)
    assert "record_cleaning_decision" not in TOOLS
    before = mctx.original.df.copy()
    for i in m.cleaning_issues:
        if i["requires_approval"]:
            record_cleaning_decision(mctx, i["id"], approve=i["kind"] != "missing")
    run_turn(mctx, None, event="cleaning_decided")
    pd.testing.assert_frame_equal(mctx.original.df, before)  # original never modified
    assert storage.dataset_file(m.dataset_id).read_bytes() == mctx._raw_bytes
    work, _ = mctx.working()
    assert len(work) == len(before) - 2 and set(work["Region"].dropna()) <= {"Central", "Northern", "Southern", "East Coast"}
    assert work["Sales Rep"].isna().sum() == before["Sales Rep"].isna().sum()  # rejected: missing names kept
    log = m.transformations
    assert all(t["approved_by_user"] for t in log) and any(t["op"] == "keep" for t in log)


def test_no_repeated_questions_after_restart(mctx, storage):
    run_turn(mctx, None, event="start")
    answer_all(mctx, {"objective": "Sales performance", "UAG_2": "Accessories group sales"}, limit=2)
    asked = len(mctx.memory.questions)
    # "Restart": reopen from disk
    ctx2 = AgentContext.open(storage, mctx.memory.dataset_id, "Sheet1")
    assert ctx2.memory.stage == mctx.memory.stage and ctx2.memory.interpretations == mctx.memory.interpretations
    res, err = run_tool(ctx2, "ask_user_clarification", {"question": "What is UAG_2?", "column": "UAG_2", "topic": "meaning", "suggested_answers": []})
    assert not err and res["already_answered"] and "Accessories group sales" in res["answer"]
    assert len(ctx2.memory.questions) == asked


def test_save_interpretation_requires_user_for_ambiguous(mctx):
    run_turn(mctx, None, event="start")
    run_turn(mctx, None, "Sales performance")
    base = {"column": "UAG_2", "meaning": "guess", "display_name": "", "data_type": "number", "unit": "", "business_role": "measure",
            "aggregation": "sum", "kpi_suitable": True}
    res, err = run_tool(mctx, "save_column_interpretation", dict(base, based_on_user_answer=True))
    assert err and "no user answer" in res["error"]
    res, err = run_tool(mctx, "save_column_interpretation", dict(base, based_on_user_answer=False))
    assert err and "Ask the user" in res["error"]
    res, err = run_tool(mctx, "save_column_interpretation", dict(base, column="Region", business_role="measure", based_on_user_answer=False))
    assert err and "can't be used as a measure" in res["error"]


def test_dashboard_reopens_after_restart_with_cleaning(mctx, storage):
    guided_to_ready(mctx)
    cfg = storage.load_dashboard(mctx.memory.dashboard_id)  # read back from disk, as after a restart
    raw = load_dataset(storage.dataset_file(cfg["dataset_id"]), ".xlsx", cfg["sheet"]).df
    df = dashboard_transform(raw, cfg)
    profile, _ = apply_overrides(profile_dataset(df), cfg["column_overrides"])
    clean, warnings = validate_dashboard_config(cfg, profile)
    assert not warnings and len(clean["charts"]) == len(cfg["charts"])
    for spec in clean["charts"]:
        prepare_chart_data(df, spec, profile)  # every chart still calculates
    assert "Net revenue" in df.columns and len(df) == 160


def test_ready_stage_answers_with_engine(mctx):
    guided_to_ready(mctx)
    run_turn(mctx, None, "Which branch has the highest net revenue?")
    msg = mctx.memory.conversation[-1]
    work, _ = mctx.working()
    best = work.groupby("BR_CD")["Net revenue"].sum().idxmax()
    assert msg["kind"] == "answer" and f"**{best}**" in msg["text"]


# ------------------------------------------------------------------ cleaning & plan validation units

def test_cleaning_ops_safe():
    df = pd.DataFrame({"a": [1.0, None, 3.0, 100.0], "c": ["x", " X ", None, "y"]})
    with pytest.raises(CleaningError):
        validate_op({"op": "drop_table"}, list(df.columns), {"a"})
    with pytest.raises(CleaningError):
        validate_op({"op": "fill_mean", "column": "c"}, list(df.columns), {"a"})
    out, log = apply_ops(df, [validate_op({"op": "fill_median", "column": "a"}, list(df.columns), {"a"})])
    assert out["a"].tolist() == [1.0, 3.0, 3.0, 100.0] and 0 not in out["a"].tolist()
    assert df["a"].isna().sum() == 1  # input untouched


def test_plan_validation_uses_interpretations(sales_loaded):
    res, p = sales_loaded
    good = {"title": "Sales in 2024", "summary": "", "business_questions": ["Which region sells most?"],
            "kpis": [{"kind": "agg", "column": "Sales", "aggregation": "sum", "label": "Total sales"}],
            "charts": [{"type": "bar", "x": "Region", "y": "Sales", "agg": "sum", "title": "", "why": ""}], "filters": ["Region"]}
    clean, errors, _ = validate_plan(good, p, {})
    assert not errors and clean["title"] == "Sales in 2024"
    bad = dict(good, title="Sales grew 25%",
               kpis=[{"kind": "agg", "column": "Unit Price", "aggregation": "sum", "label": ""},
                     {"kind": "agg", "column": "Order ID", "aggregation": "sum", "label": ""}],
               charts=[{"type": "bar", "x": "Region", "y": "Unit Price", "agg": "sum", "title": "RM 1,200 in sales", "why": ""}])
    _, errors, _ = validate_plan(bad, p, {"Sales": {"kpi_suitable": False}})
    text = " ".join(errors)
    assert "states a number" in text and "average-type" in text and "identifier" in text
    _, errors, _ = validate_plan(good, p, {"Sales": {"kpi_suitable": False}})
    assert any("headline number" in e for e in errors)


# ------------------------------------------------------------------ LLM orchestrator with a scripted fake model

class FakeLLM:
    name = "fake"

    def __init__(self, script):
        self.script, self.calls = list(script), []

    def create(self, system, messages, tools, allow_tools=True):
        self.calls.append({"tools": [t["name"] for t in tools], "allow_tools": allow_tools,
                           "messages": json.loads(json.dumps(messages))})
        step = self.script.pop(0) if self.script else {"text": "Done."}
        if callable(step):
            step = step(messages)
        content = []
        if step.get("thinking"):
            content.append({"type": "thinking", "thinking": "", "signature": "sig-abc"})
        if step.get("text"):
            content.append({"type": "text", "text": step["text"]})
        for i, (name, args) in enumerate(step.get("tools", [])):
            content.append({"type": "tool_use", "id": f"tu_{len(self.calls)}_{i}", "name": name, "input": args})
        return LLMReply(content, "tool_use" if step.get("tools") else "end_turn")


def _assert_valid_transcript(t):
    assert t[0]["role"] == "user"
    for i, msg in enumerate(t):
        if msg["role"] == "assistant":
            ids = [b["id"] for b in msg["content"] if b.get("type") == "tool_use"]
            if ids:
                nxt = t[i + 1]
                assert nxt["role"] == "user"
                assert sorted(b["tool_use_id"] for b in nxt["content"] if b.get("type") == "tool_result") == sorted(ids)


def test_llm_turn_uses_gated_tools_and_stops_for_question(mctx):
    mctx.memory.consent = True
    llm = FakeLLM([
        {"thinking": True, "tools": [("inspect_dataset", {}), ("apply_approved_transformations", {})]},
        {"text": "I've looked at your report.", "tools": [("ask_user_clarification", {
            "question": "What would you like to understand?", "column": "", "topic": "objective",
            "suggested_answers": ["Sales performance"]})]},
        {"text": "should not be reached"},
    ])
    run_turn(mctx, llm, event="start")
    first = llm.calls[0]
    assert "inspect_dataset" in first["tools"] and "execute_dashboard_plan" not in first["tools"]
    t = mctx.memory.llm["transcript"]
    _assert_valid_transcript(t)
    results = t[2]["content"]
    assert json.loads(results[0]["content"])["rows"] == 162
    assert results[1].get("is_error")  # stage-gated tool refused
    assert t[1]["content"][0] == {"type": "thinking", "thinking": "", "signature": "sig-abc"}  # replayed unchanged
    assert len(llm.calls) == 2 and mctx.memory.pending()  # stopped to wait for the user
    assert "<app_state>" in t[0]["content"][0]["text"]


def test_llm_answer_flow_saves_interpretation(mctx):
    mctx.memory.consent = True
    llm = FakeLLM([
        {"tools": [("ask_user_clarification", {"question": "What do you want to know?", "column": "", "topic": "objective", "suggested_answers": []})]},
        {"tools": [("set_business_objective", {"objective": "Accessory sales by branch", "category": "sales"}),
                   ("ask_user_clarification", {"question": "What is UAG_2?", "column": "UAG_2", "topic": "meaning", "suggested_answers": []})]},
        {"tools": [("save_column_interpretation", {"column": "UAG_2", "meaning": "Ultimate Accessories Group 2 sales",
                                                   "display_name": "Accessories group 2 sales", "data_type": "number", "unit": "RM",
                                                   "business_role": "measure", "aggregation": "sum", "kpi_suitable": True,
                                                   "based_on_user_answer": True})]},
        {"text": "Thanks!"},
    ])
    run_turn(mctx, llm, event="start")
    run_turn(mctx, llm, "Accessory sales by branch")
    run_turn(mctx, llm, "It means Ultimate Accessories Group 2 sales, in RM")
    it = mctx.memory.interpretations["UAG_2"]
    assert it["confirmed"] and it["display_name"] == "Accessories group 2 sales"
    last_user = llm.calls[-2]["messages"][-1]["content"][0]["text"]
    assert "Answer to your question" in last_user and "UAG_2" in last_user
    _assert_valid_transcript(mctx.memory.llm["transcript"])


def test_tool_call_limit_prevents_runaway_loops(mctx):
    mctx.memory.consent = True
    llm = FakeLLM([{"tools": [("inspect_dataset", {})]}] * 50)
    Orchestrator(mctx, llm, max_tool_calls=4).handle(event="start")
    assert len(llm.calls) == 5 and llm.calls[-1]["allow_tools"] is False
    _assert_valid_transcript(mctx.memory.llm["transcript"])


def test_interrupted_turn_is_repaired(mctx, storage):
    mctx.memory.consent = True
    t = mctx.memory.llm["transcript"]
    t.append({"role": "user", "content": [{"type": "text", "text": "start"}]})
    t.append({"role": "assistant", "content": [{"type": "tool_use", "id": "tu_dangling", "name": "inspect_dataset", "input": {}}]})
    mctx.save()  # app crashed here
    ctx2 = AgentContext.open(storage, mctx.memory.dataset_id, "Sheet1")
    llm = FakeLLM([{"text": "Where were we?"}])
    run_turn(ctx2, llm, "hello again")
    t2 = ctx2.memory.llm["transcript"]
    assert t2[2]["content"][0]["tool_use_id"] == "tu_dangling" and t2[2]["content"][0]["is_error"]
    _assert_valid_transcript(t2)


def test_llm_outage_falls_back_to_guided(mctx):
    mctx.memory.consent = True

    class Down:
        name = "down"

        def create(self, *a, **k):
            raise LLMUnavailable("network")
    run_turn(mctx, Down(), event="start")
    texts = [c["text"] for c in mctx.memory.conversation]
    assert any("guided mode" in t for t in texts) and mctx.memory.pending()["kind"] == "objective"


def test_no_consent_means_no_llm_calls(mctx):
    llm = FakeLLM([])
    assert run_turn(mctx, llm, event="start") == "guided"
    assert llm.calls == []


def test_anthropic_request_shape():
    class Block:
        def __init__(self, d):
            self.d = d

        def model_dump(self, **k):
            return self.d

    class Resp:
        content = [Block({"type": "text", "text": "hi"})]
        stop_reason = "end_turn"
        usage = None

    class FakeMessages:
        def __init__(self):
            self.kwargs = None

        def create(self, **kwargs):
            self.kwargs = kwargs
            return Resp()

    class FakeClient:
        def __init__(self):
            self.beta = type("B", (), {"messages": FakeMessages()})()

    client = FakeClient()
    llm = AnthropicLLM(client=client)
    reply = llm.create("system", [{"role": "user", "content": "hi"}], [TOOLS["inspect_dataset"].api_definition()], allow_tools=False)
    kw = client.beta.messages.kwargs
    assert kw["model"] == "claude-opus-5-5" and kw["output_config"] == {"effort": "medium"}
    assert kw["fallbacks"] == "default" and kw["betas"] == ["server-side-fallback-2026-07-01"]
    assert kw["tools"][0]["strict"] is True and kw["tool_choice"] == {"type": "none"}
    assert "thinking" not in kw  # Opus 5.5 always uses adaptive thinking
    assert reply.content == [{"type": "text", "text": "hi"}]


def test_analysis_listed_and_deleted_with_dataset(mctx, storage):
    guided.start(mctx)
    listed = storage.list_analyses()
    assert listed and listed[0]["dataset_id"] == mctx.memory.dataset_id
    storage.delete_dataset(mctx.memory.dataset_id)
    assert storage.list_analyses() == []


def test_clear_business_names_are_not_questioned():
    df = pd.DataFrame({"Shipping Cost (RM)": [1.5, 2.0, 3.0] * 5, "Recognized Revenue (RM)": [10.0, 20.0, 30.5] * 5,
                       "COGS (RM)": [4.0, 5.0, 6.5] * 5, "Unit Price (RM)": [2.5, 3.5, 4.5] * 5, "Region": ["A", "B", "C"] * 5,
                       "UAG_2": [7.0, 8.0, 9.5] * 5})
    asked = [a["column"] for a in find_ambiguous(profile_dataset(df))]
    assert asked == ["UAG_2"]


def test_every_message_is_persisted(mctx, storage):
    guided_to_ready(mctx)
    reloaded = AgentContext.open(storage, mctx.memory.dataset_id, "Sheet1")
    assert reloaded.memory.conversation == mctx.memory.conversation
    assert reloaded.memory.conversation[-1]["kind"] == "dashboard_ready"


# ------------------------------------------------------------------ free / open-source providers

from src.agent import llm as llm_mod  # noqa: E402


def test_openai_message_conversion_round_trip():
    transcript = [
        {"role": "user", "content": [{"type": "text", "text": "start"}]},
        {"role": "assistant", "content": [{"type": "thinking", "thinking": "", "signature": "x"},
                                          {"type": "text", "text": "Looking."},
                                          {"type": "tool_use", "id": "t1", "name": "inspect_dataset", "input": {}}]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "{\"rows\": 5}", "is_error": True}]},
    ]
    msgs = llm_mod.to_openai_messages("SYS", transcript)
    assert msgs[0] == {"role": "system", "content": "SYS"}
    assert msgs[1] == {"role": "user", "content": "start"}
    assert msgs[2]["tool_calls"][0]["function"] == {"name": "inspect_dataset", "arguments": "{}"} and msgs[2]["content"] == "Looking."
    assert msgs[3] == {"role": "tool", "tool_call_id": "t1", "content": "ERROR: {\"rows\": 5}"}


def test_schema_sanitised_for_other_providers():
    s = llm_mod._sanitize_schema(TOOLS["create_dashboard_plan"].api_definition()["input_schema"])
    text = json.dumps(s)
    assert "additionalProperties" not in text and '""' not in text


def test_openai_compatible_client(mctx):
    class Fn:
        def __init__(self, n, a):
            self.name, self.arguments = n, a

    class TC:
        def __init__(self, i, n, a):
            self.id, self.function = i, Fn(n, a)

    class Msg:
        content = "<think>hmm</think>Let me look."
        tool_calls = [TC("c1", "get_column_profile", '{"column": "REV_NET"}'), TC("c2", "inspect_dataset", "not json")]

    class Resp:
        choices = [type("C", (), {"message": Msg()})()]
        usage = None

    class Completions:
        def create(self, **kw):
            self.kw = kw
            return Resp()

    comp = Completions()
    client = type("Cl", (), {"chat": type("Ch", (), {"completions": comp})()})()
    gem = llm_mod.OpenAICompatLLM("gemini", "https://x", "key", "gemini-flash-latest", client=client)
    reply = gem.create("SYS", [{"role": "user", "content": [{"type": "text", "text": "hi"}]}],
                       [TOOLS["get_column_profile"].api_definition()])
    assert reply.content[0] == {"type": "text", "text": "Let me look."}  # reasoning tags stripped
    assert reply.content[1]["input"] == {"column": "REV_NET"} and reply.stop_reason == "tool_use"
    assert "_invalid_arguments" in reply.content[2]["input"]  # bad JSON → rejected by local validation, not executed
    assert comp.kw["tools"][0]["type"] == "function" and comp.kw["tool_choice"] == "auto"
    _, err = run_tool(mctx, "inspect_dataset", reply.content[2]["input"])
    assert err


def test_ollama_native_client():
    class R:
        status_code = 200

        def json(self):
            return {"message": {"content": "", "tool_calls": [{"function": {"name": "inspect_dataset", "arguments": {}}}]}}

    class S:
        def post(self, url, json=None, timeout=None):
            self.url, self.body = url, json
            return R()

    s = S()
    o = llm_mod.OllamaLLM("qwen3:4b", session=s)
    transcript = [{"role": "user", "content": [{"type": "text", "text": "hi"}]},
                  {"role": "assistant", "content": [{"type": "tool_use", "id": "a", "name": "inspect_dataset", "input": {"x": 1}}]},
                  {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "a", "content": "{}"}]}]
    reply = o.create("SYS", transcript, [TOOLS["inspect_dataset"].api_definition()])
    assert s.url.endswith("/api/chat") and s.body["think"] is False and s.body["options"]["num_ctx"] >= 8192
    assert s.body["messages"][2]["tool_calls"][0]["function"]["arguments"] == {"x": 1}  # objects, not strings
    assert reply.content[0]["name"] == "inspect_dataset" and reply.content[0]["id"]


def test_provider_selection(monkeypatch):
    for k in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "GEMINI_API_KEY", "GOOGLE_API_KEY", "GROQ_API_KEY",
              "OPENROUTER_API_KEY", "DASHBOARD_LLM_PROVIDER", "DASHBOARD_LLM_MODEL"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setattr(llm_mod, "_ollama_models", lambda base: ())
    assert provider_status_safe()["enabled"] is False
    monkeypatch.setenv("GEMINI_API_KEY", "test")
    st = provider_status_safe()
    assert st["name"] == "gemini" and "Google" in st["data_note"] and st["model"] == "gemini-flash-lite-latest"
    monkeypatch.setenv("GROQ_API_KEY", "test")
    monkeypatch.setenv("DASHBOARD_LLM_PROVIDER", "groq")
    assert provider_status_safe()["name"] == "groq"
    monkeypatch.setenv("DASHBOARD_LLM_PROVIDER", "ollama")
    assert provider_status_safe()["enabled"] is False  # not running
    monkeypatch.setattr(llm_mod, "_ollama_models", lambda base: ("qwen3:4b",))
    st = provider_status_safe()
    assert st["name"] == "ollama" and st["local"] and st["cost"] == "Free"
    monkeypatch.setenv("DASHBOARD_LLM_PROVIDER", "none")
    assert provider_status_safe()["enabled"] is False


def provider_status_safe():
    return llm_mod.provider_status()


def test_gemini_thought_signature_is_echoed_back():
    reply = llm_mod._blocks(None, [("c1", "inspect_dataset", "{}", {"google": {"thought_signature": "SIG"}})])
    assert reply.content[0]["provider_extra"] == {"google": {"thought_signature": "SIG"}}
    msgs = llm_mod.to_openai_messages("S", [{"role": "user", "content": "hi"}, {"role": "assistant", "content": reply.content}])
    assert msgs[2]["tool_calls"][0]["extra_content"] == {"google": {"thought_signature": "SIG"}}
    # The Claude connector strips the Gemini-only field
    kw = AnthropicLLM(client=object())._request("S", [{"role": "assistant", "content": reply.content}], [], True)
    assert "provider_extra" not in kw["messages"][0]["content"][0]


def test_free_tier_rate_limit_waits_then_retries():
    class RateLimit(Exception):
        pass

    class Other(Exception):
        pass

    fake_openai = type("O", (), {"RateLimitError": RateLimit, "AuthenticationError": Other, "NotFoundError": Other,
                                 "BadRequestError": Other, "APIConnectionError": Other, "APIStatusError": Other})

    class Msg:
        content, tool_calls = "Hello", None

    class Completions:
        calls = 0

        def create(self, **kw):
            self.calls += 1
            if self.calls < 3:
                raise RateLimit("429 ... 'retryDelay': '7s'")
            return type("R", (), {"choices": [type("C", (), {"message": Msg()})()], "usage": None})()

    comp, slept = Completions(), []
    client = type("Cl", (), {"chat": type("Ch", (), {"completions": comp})()})()
    gem = llm_mod.OpenAICompatLLM("gemini", "x", "k", "m", client=client, sleep=slept.append)
    gem._openai = fake_openai
    reply = gem.create("S", [{"role": "user", "content": "hi"}], [])
    assert reply.content[0]["text"] == "Hello" and slept == [8.0, 8.0]
    comp.calls = -100  # keeps failing → gives up within the wait budget with a clear message
    gem2 = llm_mod.OpenAICompatLLM("gemini", "x", "k", "m", client=client, sleep=slept.append, max_wait=20)
    gem2._openai = fake_openai
    with pytest.raises(LLMUnavailable, match="limiting how fast"):
        gem2.create("S", [{"role": "user", "content": "hi"}], [])
