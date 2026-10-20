"""Dashboard colour themes: page/card/accent colours and the chart palette.

A dashboard stores ``theme`` as plain hex colours. ``clean_theme`` validates anything read from disk, the user or the
AI assistant; ``resolve`` fills in the derived colours (readable text and grid lines for the chosen card colour).
"""
from __future__ import annotations

import re

from .config import SERIES_COLORS

_HEX = re.compile(r"^#[0-9a-fA-F]{6}$")
PALETTE_SIZE = 8

PRESETS: dict[str, dict] = {
    "Classic": {"page_bg": "#f6f7f9", "card_bg": "#ffffff", "accent": "#1f5fbf", "palette": list(SERIES_COLORS[:PALETTE_SIZE])},
    "Ocean": {"page_bg": "#eef6fb", "card_bg": "#ffffff", "accent": "#0b6e99",
              "palette": ["#0b6e99", "#1fa2c7", "#5cc8d7", "#2f4b7c", "#7aa6c2", "#00a39b", "#94d2bd", "#3d5a80"]},
    "Sunset": {"page_bg": "#fff6f0", "card_bg": "#ffffff", "accent": "#d9480f",
               "palette": ["#d9480f", "#f08c00", "#e64980", "#7048e8", "#fab005", "#c2255c", "#ff8787", "#845ef7"]},
    "Forest": {"page_bg": "#f1f7f2", "card_bg": "#ffffff", "accent": "#2b8a3e",
               "palette": ["#2b8a3e", "#74b816", "#087f5b", "#a9a23a", "#5c940d", "#40c057", "#82a87e", "#495057"]},
    "Malaysia": {"page_bg": "#f7f8fb", "card_bg": "#ffffff", "accent": "#010066",
                 "palette": ["#010066", "#cc0001", "#ffcc00", "#3b5bdb", "#e8590c", "#868e96", "#5f3dc4", "#f59f00"]},
    "Pastel": {"page_bg": "#faf8ff", "card_bg": "#ffffff", "accent": "#7c6fd6",
               "palette": ["#9b8cf2", "#7fc8f8", "#ffb5a7", "#a7e8bd", "#fcd581", "#f4a6cd", "#b8c0ff", "#9ad1d4"]},
    "Midnight": {"page_bg": "#0f1724", "card_bg": "#182235", "accent": "#5ab0ff",
                 "palette": ["#5ab0ff", "#ffb74d", "#4dd0a5", "#f06292", "#ba8cff", "#ffe066", "#7dd3fc", "#ff8a65"]},
    "Mono": {"page_bg": "#f5f5f5", "card_bg": "#ffffff", "accent": "#212529",
             "palette": ["#212529", "#495057", "#868e96", "#adb5bd", "#343a40", "#6c757d", "#ced4da", "#1c7ed6"]},
}
DEFAULT = "Classic"


def is_hex(value) -> bool:
    return isinstance(value, str) and bool(_HEX.match(value.strip()))


def clean_theme(raw) -> dict | None:
    """Validated theme or None (= default). Unknown keys and malformed colours are dropped."""
    if not isinstance(raw, dict):
        return None
    preset = raw.get("preset") if raw.get("preset") in PRESETS else None
    base = PRESETS[preset or DEFAULT]
    out = {"preset": preset or "Custom"}
    for key in ("page_bg", "card_bg", "accent"):
        v = raw.get(key)
        out[key] = v.strip().lower() if is_hex(v) else base[key]
    pal = [c.strip().lower() for c in (raw.get("palette") or []) if is_hex(c)][:PALETTE_SIZE] if isinstance(raw.get("palette"), list) else []
    out["palette"] = pal if len(pal) >= 2 else list(base["palette"])
    return out


def from_preset(name: str) -> dict:
    return dict(PRESETS[name], preset=name, palette=list(PRESETS[name]["palette"]))


def _rgb(h: str) -> tuple[int, int, int]:
    h = h.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def luminance(h: str) -> float:
    def ch(c: int) -> float:
        c = c / 255
        return c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4
    r, g, b = _rgb(h)
    return 0.2126 * ch(r) + 0.7152 * ch(g) + 0.0722 * ch(b)


def is_dark(h: str) -> bool:
    return luminance(h) < 0.25


def mix(a: str, b: str, t: float) -> str:
    ra, ga, ba = _rgb(a)
    rb, gb, bb = _rgb(b)
    return "#{:02x}{:02x}{:02x}".format(round(ra + (rb - ra) * t), round(ga + (gb - ga) * t), round(ba + (bb - ba) * t))


def resolve(theme: dict | None) -> dict:
    """Full colour set for rendering: the stored colours plus text/grid colours that stay readable."""
    t = clean_theme(theme) or from_preset(DEFAULT)
    dark = is_dark(t["card_bg"])
    ink = "#eef2f7" if dark else "#1c1f24"
    page_ink = "#eef2f7" if is_dark(t["page_bg"]) else "#1c1f24"
    return {
        **t,
        "dark": dark,
        "ink": ink,
        "ink2": mix(ink, t["card_bg"], 0.3),
        "ink3": mix(ink, t["card_bg"], 0.45),
        "grid": mix(ink, t["card_bg"], 0.88),
        "line": mix(ink, t["card_bg"], 0.82),
        "page_ink": page_ink,
        "page_ink2": mix(page_ink, t["page_bg"], 0.3),
        "accent_soft": mix(t["accent"], t["card_bg"], 0.88),
        "other": mix(ink, t["card_bg"], 0.65),
        "sequential": [[0.0, mix(t["palette"][0], t["card_bg"], 0.9)], [0.5, mix(t["palette"][0], t["card_bg"], 0.4)],
                       [1.0, t["palette"][0]]],
    }


def with_main_color(palette: list[str], color: str | None) -> list[str]:
    """Palette for one chart whose main colour the user changed."""
    if not is_hex(color):
        return palette
    color = color.lower()
    return [color] + [c for c in palette if c != color]


def page_css(theme: dict | None) -> str:
    """CSS that applies the dashboard colours to the page (only the dashboard page calls this)."""
    r = resolve(theme)
    if clean_theme(theme) is None or r["preset"] == DEFAULT:
        return ""
    return f"""<style>
.stApp, [data-testid="stMain"] {{ background: {r['page_bg']} !important; }}
:root {{ --surface: {r['card_bg']}; --ink: {r['ink']}; --ink-2: {r['ink2']}; --ink-3: {r['ink3']}; --line: {r['line']};
        --accent: {r['accent']}; --accent-soft: {r['accent_soft']}; }}
[data-testid="stMain"] h1, [data-testid="stMain"] h2, [data-testid="stMain"] h3,
[data-testid="stMain"] .page-eyebrow {{ color: {r['page_ink']} !important; }}
[data-testid="stMain"] > div [data-testid="stCaptionContainer"] {{ color: {r['page_ink2']}; }}
[class*="st-key-card_"] {{ background: {r['card_bg']} !important; border-color: {r['line']} !important; }}
[class*="st-key-card_"] [data-testid="stMarkdownContainer"] p, [class*="st-key-card_"] [data-testid="stMarkdownContainer"] li,
[class*="st-key-card_"] [data-testid="stWidgetLabel"] p, [class*="st-key-card_"] [data-testid="stCaptionContainer"] {{ color: {r['ink']}; }}
[class*="st-key-card_"] button [data-testid="stMarkdownContainer"] p {{ color: inherit; }}
.kpi {{ border-top: 3px solid {r['accent']} !important; }}
.kpi-value {{ color: {r['ink']} !important; }}
.chip.accent {{ color: {r['accent']}; border-color: {r['accent']}; }}
</style>"""
