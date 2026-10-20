"""The animated robot buddy, shared by the AI Analyst page, the clean-up spreadsheet and the dashboard.

Poses are horizontal strips of frames in ``static/buddy/<mood>.png`` (made by ``scripts/make_buddy_sprites.py``),
animated with CSS. A small script makes buddies draggable (the position is remembered in the browser) and lets the
user poke the free-floating one, which reacts. The script is a fixed string; no user text is ever put into it.
"""
from __future__ import annotations

import html

import streamlit as st

SPRITE = 116
AVATAR = "static/buddy/avatar.png"
FRAMES = {"idle": 5, "think": 4, "aha": 1, "present": 3, "celebrate": 2}
SPEED = {"idle": 2.4, "think": 1.1, "aha": 1, "present": 1.8, "celebrate": 0.7}

POKE_LINES = ["Ouch!", "Hey, that tickles!", "Boop!", "I'm working, I promise!", "Careful, I'm full of data!",
              "Hehe!", "Again?!", "Wheee!", "My numbers are getting dizzy!"]

_CSS = """<style>
.buddy-free { position: fixed; right: 26px; bottom: 104px; z-index: 1001; width: 120px; height: 120px;
  cursor: grab; touch-action: none; user-select: none; -webkit-user-select: none; }
.buddy-free.buddy-dragging, .st-key-dash_buddy.buddy-dragging { cursor: grabbing; }
.buddy-moved { animation: none !important; }
.buddy-sprite { width: %(s)dpx; height: %(s)dpx; background-repeat: no-repeat; pointer-events: none;
  filter: drop-shadow(0 6px 10px rgba(16, 40, 80, .28)); }
.buddy-free .buddy-body { animation: buddy-bob 3.2s ease-in-out infinite; }
.buddy-free:hover .buddy-sprite { filter: drop-shadow(0 0 14px rgba(70, 190, 255, .75)); }
.buddy-free.buddy-hit .buddy-body { animation: buddy-hit .55s cubic-bezier(.3, 1.6, .5, 1); }
.buddy-say { position: absolute; right: 100px; bottom: 96px; width: max-content; max-width: 220px; background: #fff;
  color: #1c1f24; border: 1px solid #d8e6f5; border-radius: 14px 14px 4px 14px; padding: 7px 11px; font-size: .8rem;
  font-weight: 500; line-height: 1.35; box-shadow: 0 6px 18px rgba(16, 40, 80, .14); pointer-events: none; }
.buddy-say:empty { display: none; }
.buddy-left .buddy-say { right: auto; left: 100px; border-radius: 14px 14px 14px 4px; }
.buddy-say-pop { animation: buddy-pop .35s ease-out; }
.buddy-star { position: absolute; left: 50%%; top: 35%%; font-size: 18px; color: #ffc93c; pointer-events: none;
  animation: buddy-star .8s ease-out forwards; text-shadow: 0 0 6px rgba(255, 201, 60, .8); }
.buddy-inline { display: flex; gap: 10px; align-items: flex-end; margin: 2px 0 8px 0; }
.buddy-inline .buddy-sprite { flex: 0 0 auto; width: 84px; height: 84px; }
.buddy-inline .buddy-card { background: #f3f8fe; border: 1px solid #cfe2f6; border-radius: 14px 14px 14px 4px;
  padding: 9px 12px; font-size: .86rem; line-height: 1.45; color: #1c1f24; }
@keyframes buddy-bob { 0%%, 100%% { transform: translateY(0); } 50%% { transform: translateY(-9px); } }
@keyframes buddy-hit { 0%% { transform: scale(1, 1) rotate(0); } 25%% { transform: scale(1.18, .8) rotate(-12deg); }
  55%% { transform: scale(.9, 1.12) rotate(10deg); } 80%% { transform: scale(1.04, .97) rotate(-4deg); } 100%% { transform: none; } }
@keyframes buddy-pop { from { opacity: 0; transform: translateY(6px) scale(.95); } to { opacity: 1; transform: none; } }
@keyframes buddy-star { from { opacity: 1; transform: translate(-50%%, -50%%) scale(.6); }
  to { opacity: 0; transform: translate(calc(-50%% + var(--dx)), calc(-50%% + var(--dy))) scale(1.2) rotate(140deg); } }
%(frames)s
@media (max-width: 640px) { .buddy-free { right: 8px; bottom: 92px; transform: scale(.75); transform-origin: bottom right; }
  .buddy-free .buddy-say { display: none; } .buddy-free .buddy-say.buddy-say-pop { display: block; } }
@media (prefers-reduced-motion: reduce) { .buddy-free .buddy-body, .buddy-sprite { animation: none !important; } }
</style>"""

# Drag (any buddy) and poke (free-floating buddies). Installed once per browser tab; a MutationObserver wires buddies
# that Streamlit renders later. Positions are stored in localStorage per buddy id and clamped to the window.
_JS = """<script>
(function () {
  if (window.__buddyInit) return;
  window.__buddyInit = true;
  var KEY = 'buddy-pos:';
  var LINES = %(lines)s;
  function clamp(el, x, y) {
    var w = el.offsetWidth || 120, h = el.offsetHeight || 120;
    return [Math.max(0, Math.min(window.innerWidth - w, x)), Math.max(0, Math.min(window.innerHeight - h, y))];
  }
  function place(el, x, y) {
    var p = clamp(el, x, y);
    el.style.left = p[0] + 'px'; el.style.top = p[1] + 'px'; el.style.right = 'auto'; el.style.bottom = 'auto';
    el.classList.add('buddy-moved');
    el.classList.toggle('buddy-left', p[0] < 240);  // keep the speech bubble on screen
  }
  function idOf(el) {
    if (el.dataset.buddyId) return el.dataset.buddyId;
    var m = el.className.match(/st-key-[\\w-]+/); return m ? m[0] : 'buddy';
  }
  function restore(el) {
    try { var p = JSON.parse(localStorage.getItem(KEY + idOf(el)) || 'null'); if (p) place(el, p.x, p.y); } catch (e) {}
  }
  function stars(el) {
    for (var i = 0; i < 6; i++) {
      var s = document.createElement('span'); s.className = 'buddy-star'; s.textContent = i %% 2 ? '\\u2726' : '\\u2605';
      var a = Math.PI * 2 * i / 6 + Math.random() * .5, d = 46 + Math.random() * 22;
      s.style.setProperty('--dx', Math.cos(a) * d + 'px'); s.style.setProperty('--dy', Math.sin(a) * d + 'px');
      el.appendChild(s); setTimeout(function (n) { return function () { n.remove(); }; }(s), 850);
    }
  }
  function poke(el) {
    var n = (+el.dataset.hits || 0) + 1; el.dataset.hits = n;
    el.classList.remove('buddy-hit'); void el.offsetWidth; el.classList.add('buddy-hit');
    var say = el.querySelector('.buddy-say');
    if (say) {
      if (say.dataset.base === undefined) say.dataset.base = say.textContent;
      say.textContent = n >= 6 && n %% 6 === 0 ? "Okay, okay! Let's get back to your data!" : LINES[Math.floor(Math.random() * LINES.length)];
      say.classList.remove('buddy-say-pop'); void say.offsetWidth; say.classList.add('buddy-say-pop');
      clearTimeout(el.__sayTimer);
      el.__sayTimer = setTimeout(function () { say.textContent = say.dataset.base; say.classList.remove('buddy-say-pop'); }, 2600);
    }
    stars(el);
  }
  function wire(el) {
    if (el.dataset.buddyWired) return;
    el.dataset.buddyWired = '1';
    restore(el);
    var start = null, moved = false, ox = 0, oy = 0;
    function move(e) {
      if (!start) return;
      if (!moved && Math.hypot(e.clientX - start[0], e.clientY - start[1]) < 6) return;
      moved = true; el.classList.add('buddy-dragging'); place(el, e.clientX - ox, e.clientY - oy); e.preventDefault();
    }
    function up(e) {
      window.removeEventListener('pointermove', move); window.removeEventListener('pointerup', up);
      if (!start) return;
      if (!moved && Math.hypot(e.clientX - start[0], e.clientY - start[1]) >= 6) moved = true;
      if (moved) place(el, e.clientX - ox, e.clientY - oy);
      start = null; el.classList.remove('buddy-dragging');
      if (moved) {
        var r = el.getBoundingClientRect();
        try { localStorage.setItem(KEY + idOf(el), JSON.stringify({x: r.left, y: r.top})); } catch (e) {}
        el.dataset.justDragged = '1'; setTimeout(function () { delete el.dataset.justDragged; }, 300);
      } else if (el.dataset.buddyPoke) { poke(el); }
    }
    el.addEventListener('pointerdown', function (e) {
      if (e.button !== 0) return;
      var r = el.getBoundingClientRect(); start = [e.clientX, e.clientY]; ox = e.clientX - r.left; oy = e.clientY - r.top; moved = false;
      e.preventDefault();  // no text selection while dragging
      window.addEventListener('pointermove', move); window.addEventListener('pointerup', up);
    });
    // A drag must not also count as a click (on the dashboard, a click opens the chat).
    el.addEventListener('click', function (e) { if (el.dataset.justDragged) { e.stopPropagation(); e.preventDefault(); } }, true);
  }
  function scan() { document.querySelectorAll('.buddy-draggable, .st-key-dash_buddy').forEach(wire); }
  new MutationObserver(scan).observe(document.body, {childList: true, subtree: true});
  window.addEventListener('resize', function () { document.querySelectorAll('.buddy-moved').forEach(function (el) {
    var r = el.getBoundingClientRect(); place(el, r.left, r.top); }); });
  scan();
})();
</script>"""


def frames_css() -> str:
    return "\n".join(f"@keyframes buddy-f-{m} {{ to {{ background-position: -{n * SPRITE}px 0; }} }}"
                     for m, n in FRAMES.items() if n > 1)


def sprite_style(mood: str, size: int = SPRITE) -> str:
    """Inline style that shows (and animates) one mood at the given size."""
    mood = mood if mood in FRAMES else "idle"
    n = FRAMES[mood]
    anim = f"animation: buddy-f-{mood}-{size} {SPEED[mood]}s steps({n}) infinite;" if n > 1 else ""
    return (f"background-image: url('app/static/buddy/{mood}.png'); background-size: {n * size}px {size}px;"
            f" background-position: 0 0; width: {size}px; height: {size}px; {anim}")


def _size_keyframes(size: int) -> str:
    return "".join(f"@keyframes buddy-f-{m}-{size} {{ to {{ background-position: -{n * size}px 0; }} }}"
                   for m, n in FRAMES.items() if n > 1)


def install() -> None:
    """Styles and the drag/poke script. Call once per page run (cheap: the script installs itself only once)."""
    st.html(_CSS % {"s": SPRITE, "frames": _size_keyframes(SPRITE) + _size_keyframes(84)})
    st.html(_JS % {"lines": _js_list(POKE_LINES)}, unsafe_allow_javascript=True)


def _js_list(items: list[str]) -> str:
    import json
    return json.dumps(items)


def floating_html(mood: str, say: str, buddy_id: str = "analyst") -> str:
    """A free-floating buddy the user can drag around and poke."""
    return (f'<div class="buddy-free buddy-draggable" data-buddy-id="{html.escape(buddy_id)}" data-buddy-poke="1" '
            f'title="Drag me anywhere, or poke me!"><div class="buddy-say">{html.escape(say)}</div>'
            f'<div class="buddy-body"><div class="buddy-sprite" style="{sprite_style(mood)}"></div></div></div>')


def floating(slot, mood: str, say: str, buddy_id: str = "analyst") -> None:
    """Render (or re-render, e.g. to switch to 'think') a floating buddy into ``slot`` (an ``st.empty()``)."""
    slot.html(floating_html(mood, say, buddy_id))


def inline(mood: str, text_html: str) -> None:
    """A small buddy next to a speech card, inside the page flow. ``text_html`` must already be escaped."""
    st.html(f'<div class="buddy-inline"><div class="buddy-sprite" style="{sprite_style(mood, 84)}"></div>'
            f'<div class="buddy-card">{text_html}</div></div>')
