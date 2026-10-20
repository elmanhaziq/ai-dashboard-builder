"""Dash, the animated assistant on the dashboard page: a floating robot that opens a chat window.

The robot's poses come from ``static/buddy/*.png`` (one horizontal strip of frames per mood, made by
``scripts/make_buddy_sprites.py``) and are animated with CSS only. Its mood follows the conversation: waving while idle,
thinking while the assistant works, "aha" after an answer, presenting ideas, celebrating changes.
"""
from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from ..agent import knowledge
from ..agent.copilot import Copilot, new_state
from ..agent.llm import PRESETS, describe_choice, provider_status
from ..chart_renderer import _style
from ..data_profiler import DatasetProfile
from .. import themes
from . import buddy, state
from .analyst_view import _ai_locked, _assistant_picker, _llm, _usage_text
from .theme import esc

SPRITE = 116  # px per frame on screen
FRAMES = {"idle": 5, "think": 4, "aha": 1, "present": 3, "celebrate": 2}
SPEED = {"idle": 2.4, "think": 1.1, "aha": 1, "present": 1.8, "celebrate": 0.7}
BUBBLES = {"idle": "Hi! Click me to chat, or drag me around", "think": "Hmm, let me think…", "aha": "Aha! I found something",
           "present": "I have ideas for you!", "celebrate": "Done! Looks great"}
AVATAR = "static/buddy/avatar.png"
QUICK = [("Give me ideas", ":material/lightbulb:", "Look at my dashboard and data and give me 3 ideas to make it more useful."),
         ("Summarise", ":material/summarize:", "Summarise the most important things this data shows — good news and bad news."),
         ("Change colours", ":material/palette:", "Suggest a nicer colour theme for this dashboard and apply it."),
         ("What to check next?", ":material/explore:", "What should I look into next in this data, and why?")]

_BASE_CSS = """<style>
.st-key-dash_buddy { position: fixed !important; right: 26px; bottom: 18px; z-index: 1001; width: 132px !important;
  animation: dash-wander 46s ease-in-out infinite; }
.st-key-dash_buddy [data-testid="stButton"] { animation: dash-bob 3.2s ease-in-out infinite; }
.st-key-dash_buddy_btn button { width: %(s)dpx; height: %(s)dpx; min-height: 0; padding: 0; border: none !important;
  background-color: transparent !important; box-shadow: none !important; cursor: pointer;
  background-repeat: no-repeat; filter: drop-shadow(0 6px 10px rgba(16, 40, 80, .28)); transition: filter .2s, transform .2s; }
.st-key-dash_buddy_btn button:hover { filter: drop-shadow(0 0 14px rgba(70, 190, 255, .75)); transform: scale(1.06); }
.st-key-dash_buddy_btn button:focus-visible { outline: 3px solid #46beff; border-radius: 50%%; }
.st-key-dash_buddy_btn button p { display: none; }
.st-key-dash_bubble { position: absolute !important; right: 104px; bottom: 92px; width: max-content !important; max-width: 210px;
  animation: dash-pop 9s ease-in-out infinite; pointer-events: none; }
.st-key-dash_bubble .dash-bubble { background: #ffffff; color: #1c1f24; border: 1px solid #d8e6f5; border-radius: 14px 14px 4px 14px;
  padding: 7px 11px; font-size: .8rem; font-weight: 500; box-shadow: 0 6px 18px rgba(16, 40, 80, .14); }
.st-key-dash_chat { position: fixed !important; right: 22px; bottom: 146px; z-index: 1002;
  width: min(410px, calc(100vw - 32px)) !important; max-height: calc(100vh - 216px); overflow: hidden;
  background: #ffffff; color: #1c1f24; border: 1px solid #dfe5ee; border-radius: 18px; padding: 12px 14px 10px 14px;
  box-shadow: 0 18px 50px rgba(16, 40, 80, .22); animation: dash-open .25s ease-out; }
[data-testid="stLayoutWrapper"]:has(> .st-key-dash_msgs) { height: auto !important; min-height: 0 !important; flex: 0 1 auto !important; }
.st-key-dash_msgs { height: auto !important; max-height: min(420px, calc(100vh - 410px)) !important; min-height: 90px;
  overflow-y: auto !important; }
.st-key-dash_chat p, .st-key-dash_chat li { font-size: .88rem; line-height: 1.5; color: #1c1f24; }
.st-key-dash_chat [data-testid="stChatMessage"] { padding: 6px 4px; background: transparent; }
.st-key-dash_chat [data-testid="stHorizontalBlock"] { flex-wrap: nowrap !important; gap: .5rem; }
.st-key-dash_chat [data-testid="stColumn"] { min-width: 0 !important; width: auto !important; flex: 1 1 0 !important; }
.st-key-dash_chat [data-testid="stColumn"]:has(.dash-title) { flex: 5 1 0 !important; }
.st-key-dash_chat .dash-title { font-weight: 650; font-size: 1rem; color: #1c1f24; }
.st-key-dash_chat .dash-sub { font-size: .74rem; color: #6b7280; margin-top: -2px; }
.st-key-dash_chat .dash-idea { border: 1px dashed #9cc7f0; border-radius: 12px; padding: 8px 10px; background: #f3f8fe; margin: 4px 0; }
.st-key-dash_chat .dash-idea b { color: #1c1f24; }
.st-key-dash_chat .dash-idea span { font-size: .78rem; color: #4b5563; }
@keyframes dash-bob { 0%%, 100%% { transform: translateY(0); } 50%% { transform: translateY(-9px); } }
@keyframes dash-wander { 0%%, 12%%, 100%% { transform: translateX(0); } 30%%, 42%% { transform: translateX(-22vw); }
  60%%, 70%% { transform: translateX(-9vw); } 86%% { transform: translateX(-30vw); } }
@keyframes dash-pop { 0%%, 6%% { opacity: 0; transform: translateY(6px); } 12%%, 70%% { opacity: 1; transform: none; }
  78%%, 100%% { opacity: 0; transform: translateY(6px); } }
@keyframes dash-open { from { opacity: 0; transform: translateY(14px) scale(.98); } to { opacity: 1; transform: none; } }
%(frames)s
@media (max-width: 640px) {
  .st-key-dash_buddy { animation: none; right: 8px; bottom: 8px; transform: scale(.78); transform-origin: bottom right; }
  .st-key-dash_chat { right: 16px; bottom: 112px; max-height: calc(100vh - 180px); }
  .st-key-dash_msgs { max-height: calc(100vh - 370px) !important; }
  .st-key-dash_bubble { display: none; }
}
@media (prefers-reduced-motion: reduce) {
  .st-key-dash_buddy, .st-key-dash_buddy [data-testid="stButton"], .st-key-dash_buddy_btn button, .st-key-dash_bubble { animation: none !important; }
}
</style>"""


def _frames_css() -> str:
    return "\n".join(f"@keyframes dash-f-{m} {{ to {{ background-position: -{n * SPRITE}px 0; }} }}"
                     for m, n in FRAMES.items() if n > 1)


def _mood_css(mood: str, open_: bool) -> str:
    mood = mood if mood in FRAMES else "idle"
    n = FRAMES[mood]
    anim = f"dash-f-{mood} {SPEED[mood]}s steps({n}) infinite" if n > 1 else "none"
    extra = ".st-key-dash_buddy { animation: none; }" if open_ or mood == "think" else ""
    return (f"<style>.st-key-dash_buddy_btn button {{ background-image: url('app/static/buddy/{mood}.png');"
            f" background-size: {n * SPRITE}px {SPRITE}px; background-position: 0 0; animation: {anim}; }}{extra}</style>")


# --------------------------------------------------------------------------- state

def _chat_state(did: str) -> dict:
    return st.session_state.setdefault("dash_copilot", {}).setdefault(did, new_state())


def _choice() -> tuple[str | None, str | None]:
    """The assistant used for Dash: the user's pick, else the configured default AI (if allowed)."""
    if "dash_choice" in st.session_state:
        provider, model = st.session_state["dash_choice"]
    else:
        status = provider_status()
        provider, model = (status["name"], status["model"]) if status["enabled"] else (None, None)
    if provider and (provider not in PRESETS or (_ai_locked() and not PRESETS[provider]["local"])):
        return None, None
    return provider, model


def _knowledge_dir():
    return state.get_storage().dashboard_dir


# --------------------------------------------------------------------------- rendering

def _mini_chart(chart: dict, key: str) -> None:
    rows = pd.DataFrame(chart.get("rows") or [])
    x, y = chart.get("x"), chart.get("y")
    if rows.empty or x not in rows.columns or y not in rows.columns or len(rows) < 2:
        return
    theme = themes.resolve(st.session_state.get("dashboard", {}).get("theme"))
    color = theme["palette"][0]
    if chart.get("kind") == "line":
        fig = px.line(rows, x=x, y=y, markers=True)
        fig.update_traces(line_color=color)
    else:
        fig = px.bar(rows.head(12), x=y, y=x, orientation="h")
        fig.update_traces(marker_color=color)
        fig.update_yaxes(autorange="reversed", type="category", title_text="")
    st.plotly_chart(_style(fig, 200, False), key=key, width="stretch", config={"displayModeBar": False})


def _render_messages(cp: Copilot, did: str) -> None:
    sugg = cp.state["suggestions"]
    for i, m in enumerate(cp.state["messages"]):
        if m["role"] == "user":
            with st.chat_message("user", avatar=":material/person:"):
                st.markdown(esc(m["text"]))
            continue
        with st.chat_message("assistant", avatar=AVATAR):
            if m.get("kind") == "notice":
                st.caption(m["text"])
            else:
                st.markdown(m["text"])
            if m.get("chart"):
                _mini_chart(m["chart"], f"dash_mini_{did}_{i}")
            for sid in m.get("suggestions") or []:
                s = sugg.get(sid)
                if not s:
                    continue
                st.markdown(f'<div class="dash-idea"><b>{esc(s["spec"]["title"])}</b><br><span>{esc(s["reason"] or "")}</span></div>',
                            unsafe_allow_html=True)
                if s["status"] == "open":
                    a, b = st.columns(2)
                    if a.button("Add to dashboard", key=f"dash_add_{did}_{sid}", icon=":material/add_chart:", type="primary",
                                width="stretch"):
                        msg = cp.add_suggestion(sid)
                        st.toast(msg, icon=":material/celebration:")
                        st.rerun()
                    if b.button("No thanks", key=f"dash_no_{did}_{sid}", width="stretch"):
                        cp.dismiss_suggestion(sid)
                        st.rerun()
                else:
                    st.caption({"added": "Added to the dashboard ✓", "dismissed": "Skipped", "failed": "Couldn't be added"}
                               .get(s["status"], ""))
            if m.get("steps"):
                with st.expander("How I worked this out", icon=":material/psychology:"):
                    for step in m["steps"]:
                        st.markdown(f"- {esc(step)}")


def _memory_panel(cp: Copilot) -> None:
    kn = cp.knowledge()
    cols = cp.columns
    if not (kn["aliases"] or kn["notes"] or kn["general"]):
        st.caption("Nothing yet. Tell me what your words or columns mean (e.g. “dept means Product Category”) or how "
                   "you like answers, and I'll remember it for this kind of data.")
        return
    if kn["from_similar_file"]:
        st.caption(f"Learned on a similar file ({esc(kn['similar_name'] or 'earlier upload')}).")
    for term, col in kn["aliases"].items():
        a, b = st.columns([5, 1], vertical_alignment="center")
        a.markdown(f"“{esc(term)}” → **{esc(col)}**")
        if b.button("", key=f"dash_fa_{term}", icon=":material/delete:", help="Forget this"):
            knowledge.forget(cp.kdir, cols, "aliases", term)
            st.rerun()
    for i, note in enumerate(kn["notes"]):
        a, b = st.columns([5, 1], vertical_alignment="center")
        a.markdown(esc(note))
        if not kn["from_similar_file"] and b.button("", key=f"dash_fn_{i}", icon=":material/delete:", help="Forget this"):
            knowledge.forget(cp.kdir, cols, "notes", i)
            st.rerun()
    for i, note in enumerate(kn["general"]):
        a, b = st.columns([5, 1], vertical_alignment="center")
        a.markdown(f"_{esc(note)}_ (all data)")
        if b.button("", key=f"dash_fg_{i}", icon=":material/delete:", help="Forget this"):
            knowledge.forget(cp.kdir, cols, "general", i)
            st.rerun()


def render_copilot(cfg: dict, filtered: pd.DataFrame, profile: DatasetProfile, total_rows: int) -> None:
    did = cfg["id"]
    chat = _chat_state(did)
    open_ = bool(st.session_state.get("dash_open"))
    provider, model = _choice()

    buddy.install()  # lets the user drag Dash anywhere (a drag doesn't open the chat)
    st.markdown(_BASE_CSS % {"s": SPRITE, "frames": _frames_css()}, unsafe_allow_html=True)
    mood_slot = st.empty()
    mood_slot.markdown(_mood_css(chat.get("mood", "idle"), open_), unsafe_allow_html=True)

    with st.container(key="dash_buddy"):
        if not open_:
            with st.container(key="dash_bubble"):
                st.markdown(f'<div class="dash-bubble">{esc(BUBBLES.get(chat.get("mood"), BUBBLES["idle"]))}</div>',
                            unsafe_allow_html=True)
        if st.button("Dash", key="dash_buddy_btn", help="Close the chat" if open_ else "Chat with Dash, your data buddy"):
            st.session_state["dash_open"] = not open_
            st.rerun()

    if not open_:
        return

    cp = Copilot(cfg, filtered, profile, chat, _knowledge_dir(), llm=None, total_rows=total_rows)
    cp.greet(ai=provider is not None)
    with st.container(key="dash_chat"):
        head, tools_col, close = st.columns([5, 1.1, 0.8], vertical_alignment="center")
        with head:
            usage = f" · {_usage_text(provider, model)}" if provider else ""
            st.markdown(f'<div class="dash-title">Dash</div><div class="dash-sub">{esc(describe_choice(provider, model))}'
                        f'{esc(usage)}</div>', unsafe_allow_html=True)
        with tools_col.popover("", icon=":material/tune:", help="Choose the AI, and see what Dash has learned"):
            st.markdown("**Assistant**")
            chosen, new_p, new_m = _assistant_picker("dash", (provider, model))
            if chosen and (new_p, new_m) != (provider, model) and st.button("Use this", type="primary", key="dash_use"):
                st.session_state["dash_choice"] = (new_p, new_m)
                st.rerun()
            st.divider()
            st.markdown("**What I've learned**")
            _memory_panel(cp)
            st.divider()
            if st.button("Clear this chat", key="dash_clear", icon=":material/delete_sweep:"):
                st.session_state["dash_copilot"][did] = new_state()
                st.rerun()
        if close.button("", key="dash_close", icon=":material/close:", help="Close"):
            st.session_state["dash_open"] = False
            st.rerun()

        msgs = st.container(height=400, autoscroll=True, key="dash_msgs", border=False)
        with msgs:
            _render_messages(cp, did)

        pending = st.session_state.pop("dash_pending", None)
        if len(chat["messages"]) <= 2:
            q = st.columns(2)
            for i, (label, icon, prompt) in enumerate(QUICK):
                if q[i % 2].button(label, key=f"dash_q_{i}", icon=icon, width="stretch"):
                    pending = prompt
        text = st.chat_input("Ask Dash about this dashboard…", key=f"dash_input_{did}", max_chars=1000)
        question = text or pending
        if question:
            with msgs:  # show the question straight away, with Dash thinking underneath
                with st.chat_message("user", avatar=":material/person:"):
                    st.markdown(esc(question))
                mood_slot.markdown(_mood_css("think", True), unsafe_allow_html=True)
                cp.llm = _llm(provider, model) if provider else None
                with st.spinner("Dash is thinking…"):
                    cp.ask(question)
            st.rerun()
