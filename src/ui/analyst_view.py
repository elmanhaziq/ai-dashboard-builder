"""The AI Analyst page: upload → conversation → questions → clean-up approval → dashboard → exploration."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from .. import config
from ..agent.actions import all_decided, record_cleaning_decision, set_assistant
from ..agent.cleaning import describe
from ..agent.context import AgentContext
from ..agent.llm import PRESETS, available_providers, build_llm, describe_choice, list_models, provider_status
from ..agent.orchestrator import run_turn
from ..agent.workflow import STEP_LABELS, Stage, step_index
from ..data_loader import DataLoadError, choose_default_sheet, list_sheets, sheet_dimensions, validate_upload
from ..storage import StorageError
from . import state
from .theme import card, chips, empty_state, esc, human_size, page_header

SESSION_KEY = "analysis"


GUIDED = "__guided__"


@st.cache_resource(show_spinner=False)
def _llm(provider: str, model: str):
    return build_llm(provider, model)


@st.cache_data(show_spinner=False, ttl=3600)
def _models(provider: str) -> list[str]:
    return list(list_models(provider))


def _choice(ctx: AgentContext) -> tuple[str | None, str | None]:
    """The assistant chosen for this analysis (older analyses fall back to the configured default)."""
    m = ctx.memory
    if not m.consent:
        return None, None
    if m.llm_choice.get("provider"):
        return m.llm_choice["provider"], m.llm_choice.get("model")
    status = provider_status()
    return (status["name"], status["model"]) if status["enabled"] else (None, None)


def _assistant_picker(key: str, current: tuple[str | None, str | None] | None) -> tuple[bool, str | None, str | None]:
    """Radio to choose the guided assistant or an AI provider + model. Returns (chosen?, provider, model)."""
    providers = available_providers()
    options = [GUIDED] + providers
    labels = {GUIDED: "Guided assistant — offline, free, fixed rules (no AI service)"}
    for p in providers:
        labels[p] = f"{PRESETS[p]['label']} — {PRESETS[p]['cost']}"
    idx = None if current is None else options.index(current[0] or GUIDED) if (current[0] or GUIDED) in options else 0
    pick = st.radio("Who should help you?", options, index=idx, format_func=labels.get, key=f"{key}_prov")
    if pick is None:
        st.caption("Choose an assistant to continue.")
        return False, None, None
    if pick == GUIDED:
        if providers:
            st.caption("Free-form questions work much better with an AI model.")
        return True, None, None
    models = _models(pick)
    cur_model = current[1] if current and current[0] == pick else None
    model = st.selectbox("Model", models, index=models.index(cur_model) if cur_model in models else 0, key=f"{key}_model",
                         help="The first one is the recommended default. Larger models are smarter but slower and have "
                              "smaller free limits.")
    note = PRESETS[pick]["data_note"]
    if not PRESETS[pick]["local"]:
        note = ("Choosing this sends column names, summary statistics and up to 5 example rows (never the whole file). "
                + note)
    st.caption(note)
    return True, pick, model


def _loader(dataset_id: str, sheet: str | None, header_row: int | None):
    _, result, profile = state.load_for(dataset_id, sheet, header_row)
    return result, profile


def _context(sess: dict) -> AgentContext:
    ctx = AgentContext.open(state.get_storage(), sess["dataset_id"], sess["sheet"], sess.get("header_row"), loader=_loader)
    ctx.sample_rows_allowed = ctx.memory.consent
    return ctx


def _run(ctx: AgentContext, text: str | None = None, event: str | None = None) -> None:
    provider, model = _choice(ctx)
    llm = _llm(provider, model) if provider else None
    with st.spinner("Thinking…" if llm else "Working on it…"):
        run_turn(ctx, llm, text, event)
    st.rerun()


# --------------------------------------------------------------------------- start screen

def _stepper(stage: str) -> None:
    current = step_index(stage)
    parts = []
    for i, (label, _) in enumerate(STEP_LABELS):
        cls = "done" if i < current else "now" if i == current else ""
        parts.append(f'<div class="step {cls}"><span>{i + 1}</span>{esc(label)}</div>')
    st.markdown(f'<div class="stepper">{"".join(parts)}</div>', unsafe_allow_html=True)


def _ai_notice(status: dict) -> None:
    where = "server" if config.ONLINE else "computer"
    if status["enabled"] and status["local"]:
        st.markdown(
            f'<div class="ai-note"><b>AI assistant: {esc(status["provider"])}</b> — model <code>{esc(status["model"])}</code>. '
            f"It runs on this {where}, is free, and nothing is sent anywhere. Small local models can be slow without a "
            "graphics card (often 20–60 seconds per step). If you don't allow it, the faster built-in guided assistant "
            "runs instead.</div>", unsafe_allow_html=True)
    elif status["enabled"]:
        st.markdown(
            f'<div class="ai-note"><b>AI assistant: {esc(status["provider"])}</b> — model <code>{esc(status["model"])}</code> '
            f"({esc(status['cost'])}). If you allow it, the app sends <b>column names, summary statistics and up to 5 example "
            f"rows</b> from your file so the AI can understand the data. {esc(status['data_note'])} The full file is never "
            f"sent, and all numbers are still calculated on this {where}. If you don't allow it, a built-in guided assistant "
            "runs instead and nothing is sent to any AI service.</div>", unsafe_allow_html=True)
    else:
        st.markdown(
            '<div class="ai-note"><b>Guided mode.</b> No AI service is connected, so the built-in assistant guides you with '
            f"fixed rules and nothing leaves this {where}. {esc(status['reason'] or '')} See the README to connect a free "
            "AI (Ollama on your computer, or a Google Gemini key).</div>", unsafe_allow_html=True)


def _start_screen() -> None:
    page_header("AI Analyst", "Hi! I'm your data analyst.",
                "Upload a spreadsheet, tell me what you want to understand, and I'll build the dashboard with you.")
    status = provider_status()
    _ai_notice(status)
    st.write("")
    up = st.file_uploader("Upload an Excel or CSV file", type=["xlsx", "xls", "csv"], key="analyst_upload",
                          help=f"Up to {config.MAX_UPLOAD_MB} MB.")
    if up is not None:
        if st.session_state.get("analyst_upload_id") != up.file_id:
            try:
                ext = validate_upload(up.name, up.size)
                data = up.getvalue()
                sheets = list_sheets(data, ext)
                meta = state.get_storage().save_upload(up.name, data, ext)
                dims = sheet_dimensions(data, ext)
            except (DataLoadError, StorageError) as exc:
                st.error(str(exc), icon=":material/error:")
                return
            st.session_state["analyst_upload_id"] = up.file_id
            st.session_state["analyst_pending"] = {"dataset_id": meta["id"], "name": meta["original_name"], "size": meta["size"],
                                                   "sheets": sheets, "default": choose_default_sheet({s: dims.get(s, (0, 0)) for s in sheets}) or sheets[0]}
        pend = st.session_state.get("analyst_pending")
        if pend:
            with card("analyst_pending"):
                chips([(pend["name"], ""), (human_size(pend["size"]), "")])
                sheet = pend["sheets"][0]
                if len(pend["sheets"]) > 1:
                    sheet = st.selectbox("Which sheet should I look at?", pend["sheets"], index=pend["sheets"].index(pend["default"]))
                chosen, provider, model = _assistant_picker("start", None)
                if st.button("Start analysis", type="primary", icon=":material/auto_awesome:", disabled=not chosen):
                    sess = {"dataset_id": pend["dataset_id"], "sheet": sheet, "header_row": None}
                    try:
                        with st.spinner("Reading your data…"):
                            ctx = _context(sess)
                    except (DataLoadError, StorageError) as exc:
                        st.error(str(exc))
                        return
                    set_assistant(ctx, provider, model)
                    st.session_state[SESSION_KEY] = sess
                    st.session_state.pop("analyst_pending", None)
                    _run(ctx, event="start")

    analyses = state.get_storage().list_analyses()[:6]
    if analyses:
        st.write("")
        st.markdown("### Continue where you left off")
        cols = st.columns(3)
        for i, a in enumerate(analyses):
            with cols[i % 3], card(f"an_{a['dataset_id']}_{i}"):
                st.markdown(f"**{esc(a['dataset_name'])}**")
                step = STEP_LABELS[step_index(a["stage"] or "PROFILE")][0]
                st.caption(f"{a['objective'][:60] or 'No goal yet'} · step: {step} · {str(a['updated_at'])[:10]}")
                if st.button("Continue", key=f"an_open_{i}", icon=":material/chat:"):
                    st.session_state[SESSION_KEY] = {"dataset_id": a["dataset_id"], "sheet": a["sheet"], "header_row": None}
                    st.rerun()


# --------------------------------------------------------------------------- conversation

def _render_message(msg: dict) -> None:
    role = "user" if msg["role"] == "user" else "assistant"
    if msg.get("kind") == "event":
        st.caption(msg["text"])
        return
    with st.chat_message(role, avatar=":material/person:" if role == "user" else ":material/insights:"):
        st.markdown(msg["text"])
        data = msg.get("data") or {}
        if msg.get("kind") == "answer" and data.get("table"):
            with st.expander("How this was calculated"):
                st.dataframe(pd.DataFrame(data["table"]), hide_index=True, width="stretch")
                if data.get("method"):
                    st.caption(f"Method: {data['method']}")


def _question_panel(ctx: AgentContext) -> None:
    q = ctx.memory.pending()
    if not q:
        return
    with card("analyst_question"):
        st.markdown('<div class="panel-label">Question for you</div>', unsafe_allow_html=True)
        st.markdown(q["question"])
        for i, s in enumerate(q.get("suggestions") or []):
            if st.button(s, key=f"sugg_{q['id']}_{i}", width="stretch"):
                _run(ctx, s)
        st.caption("Or type your own answer below.")


def _cleaning_panel(ctx: AgentContext) -> None:
    m = ctx.memory
    pending = [i for i in m.cleaning_issues if i.get("requires_approval") and i["status"] == "pending"]
    if not pending or Stage(m.stage) not in (Stage.CLEANING_REVIEW, Stage.CLEANING_APPROVAL) or not m.cleaning_reviewed:
        return
    with card("analyst_cleaning"):
        st.markdown('<div class="panel-label">Review clean-up suggestions</div>', unsafe_allow_html=True)
        st.caption("Nothing changes until you confirm. Your original file is always kept as it is.")
        choices: dict[str, tuple[bool, int | None]] = {}
        for issue in pending:
            st.markdown(f"**{esc(issue['title'])}**")
            st.markdown(f'<div class="issue-text">{esc(issue["detail"])} {esc(issue["why"])}</div>', unsafe_allow_html=True)
            options = [f"Recommended: {issue['recommended']['label']}"] + [a["label"] for a in issue["alternatives"]]
            keep_idx = next((i for i, a in enumerate(issue["alternatives"], start=1) if a["op"]["op"] == "keep"), None)
            if keep_idx is None:
                options.append("Leave as is")
            pick = st.radio("Decision", options, key=f"dec_{issue['id']}", label_visibility="collapsed")
            idx = options.index(pick)
            if idx == 0:
                choices[issue["id"]] = (True, None)
            elif keep_idx is None and idx == len(options) - 1:
                choices[issue["id"]] = (False, None)
            else:
                alt = issue["alternatives"][idx - 1]
                choices[issue["id"]] = (alt["op"]["op"] != "keep", None if alt["op"]["op"] == "keep" else idx - 1)
            st.divider()
        if st.button("Confirm my decisions", type="primary", icon=":material/check:", width="stretch"):
            for iid, (approve, alt) in choices.items():
                record_cleaning_decision(ctx, iid, approve, alt)
            if all_decided(ctx):
                _run(ctx, event="cleaning_decided")
    info = [i for i in m.cleaning_issues if not i.get("requires_approval")]
    if info:
        with st.expander(f"Other things I noticed ({len(info)}) — no action needed"):
            for i in info:
                st.markdown(f"- **{esc(i['title'])}:** {esc(i['detail'])} {esc(i['why'])}")


def _dashboard_panel(ctx: AgentContext) -> None:
    m = ctx.memory
    if not m.dashboard_id:
        return
    with card("analyst_dashboard"):
        st.markdown('<div class="panel-label">Your dashboard</div>', unsafe_allow_html=True)
        try:
            cfg = state.get_storage().load_dashboard(m.dashboard_id)
        except StorageError as exc:
            st.warning(str(exc))
            return
        st.markdown(f"**{esc(cfg['name'])}**")
        st.caption(f"{len(cfg['kpis'])} key numbers · {len(cfg['charts'])} charts")
        if st.button("Open dashboard", type="primary", icon=":material/space_dashboard:", width="stretch"):
            state.open_dashboard(m.dashboard_id)
            st.switch_page(st.session_state["_pages"]["overview"])
        st.caption("You can keep asking me questions here, or ask me to change the dashboard.")


def _advanced(ctx: AgentContext) -> None:
    m = ctx.memory
    with st.expander("Advanced details", icon=":material/tune:"):
        t1, t2, t3, t4 = st.tabs(["Column meanings", "Changes made", "Questions", "Activity log"])
        with t1:
            if m.interpretations:
                st.dataframe(pd.DataFrame([{"Column": c, "Shown as": it.get("display_name") or c, "Meaning": it.get("meaning"),
                                            "Unit": it.get("unit"), "Used as": it.get("business_role"), "Calculation": it.get("aggregation"),
                                            "Headline number": it.get("kpi_suitable"), "Confirmed by you": it.get("confirmed")}
                                           for c, it in m.interpretations.items()]), hide_index=True, width="stretch")
            else:
                st.caption("No column interpretations yet.")
        with t2:
            if m.transformations:
                st.dataframe(pd.DataFrame([{"Change": t["description"], "Rows affected": t["rows_affected"], "Why": t.get("reason", ""),
                                            "Approved by you": t.get("approved_by_user")} for t in m.transformations]),
                             hide_index=True, width="stretch")
            else:
                st.caption("No changes yet. The original file is never modified; changes apply to a working copy.")
        with t3:
            st.dataframe(pd.DataFrame([{"Question": q["question"], "Column": q["column"] or "—", "Answer": q["answer"] or "(waiting)"}
                                       for q in m.questions]), hide_index=True, width="stretch") if m.questions else st.caption("None yet.")
        with t4:
            st.caption(f"Workflow stage: {m.stage} · assistant: {describe_choice(*_choice(ctx))}")
            st.dataframe(pd.DataFrame([{"When": e["at"][11:19], "Step": e["name"], "OK": e["ok"], "Detail": str(e["detail"])[:160]}
                                       for e in m.events[-60:]][::-1]), hide_index=True, width="stretch", height=260)


def _analysis_screen(sess: dict) -> None:
    try:
        ctx = _context(sess)
    except (DataLoadError, StorageError) as exc:
        st.error(f"This analysis can't be opened: {exc}")
        if st.button("Start over"):
            st.session_state.pop(SESSION_KEY, None)
            st.rerun()
        return
    m = ctx.memory
    provider, model = _choice(ctx)
    head, actions = st.columns([4, 1.6], vertical_alignment="bottom")
    with head:
        page_header("AI Analyst", m.dataset_name)
        chips([(f"Sheet: {m.sheet}", "") if m.sheet and m.sheet != "CSV data" else ("CSV file", ""),
               (describe_choice(provider, model), "accent"),
               (f"Goal: {m.objective[:50]}", "") if m.objective else ("Goal not set yet", "warn")])
    with actions:
        with st.popover("Assistant", icon=":material/smart_toy:", width="stretch",
                        help="Switch between the guided assistant and AI models at any time"):
            chosen, new_provider, new_model = _assistant_picker(f"switch_{m.dataset_id}", (provider, model))
            if chosen and (new_provider, new_model) != (provider, model) and st.button("Use this assistant", type="primary",
                                                                                     width="stretch"):
                set_assistant(ctx, new_provider, new_model)
                m.add_message("assistant", f"Switched to **{describe_choice(new_provider, new_model)}**. "
                                           "Everything we've decided so far is kept.", kind="notice")
                ctx.save()
                st.rerun()
        if st.button("New analysis", icon=":material/add:", width="stretch"):
            st.session_state.pop(SESSION_KEY, None)
            st.session_state.pop("analyst_upload_id", None)
            st.rerun()
    _stepper(m.stage)
    if not m.conversation:
        _run(ctx, event="start")

    chat, side = st.columns([1.55, 1], gap="large")
    with chat:
        with st.container(border=False):
            for msg in m.conversation:
                _render_message(msg)
    with side:
        _question_panel(ctx)
        _cleaning_panel(ctx)
        _dashboard_panel(ctx)
        _advanced(ctx)
    placeholder = ("Type your answer…" if m.pending() else "Ask a question about your data…" if m.dashboard_id
                   else "Type a message…")
    text = st.chat_input(placeholder, key="analyst_chat")
    if text and text.strip():
        _run(ctx, text.strip())


def analyst_page() -> None:
    sess = st.session_state.get(SESSION_KEY)
    if sess:
        _analysis_screen(sess)
    else:
        _start_screen()


__all__ = ["analyst_page", "describe", "empty_state"]
