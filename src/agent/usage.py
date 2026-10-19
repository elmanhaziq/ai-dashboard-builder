"""Track AI usage per provider and model, and what is known about free limits.

Providers report remaining quota differently:
* Groq / OpenRouter / Anthropic send rate-limit headers on every response → exact "remaining" numbers.
* Google Gemini sends nothing on success; its limits only appear in the error when one is hit. We count
  requests per model per day (Google resets daily quotas at midnight Pacific time) and remember the limit
  Google reports, so the next time we can show "12 of 20 used".
* Ollama runs locally → unlimited.

Counts are stored in ``ai_usage.json`` next to the uploads folder. They count requests made by *this app*;
other apps using the same key also spend the same quota.
"""
from __future__ import annotations

import json
import re
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path

from .. import config

_LOCK = threading.Lock()
USAGE_FILE = config.DATA_DIR / "ai_usage.json"

# Limits observed for free tiers (used until the provider tells us otherwise). They change often.
KNOWN_DAILY_LIMITS = {("gemini", "gemini-flash-latest"): 20}


def _today(provider: str) -> str:
    if provider == "gemini":  # Google resets daily quotas at midnight Pacific time (UTC-8 / UTC-7)
        return (datetime.now(timezone.utc) - timedelta(hours=8)).strftime("%Y-%m-%d")
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def _load(path: Path | None = None) -> dict:
    p = path or USAGE_FILE
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _save(data: dict, path: Path | None = None) -> None:
    from ..storage import atomic_write_text

    try:
        atomic_write_text(path or USAGE_FILE, json.dumps(data, indent=1))
    except OSError:
        pass  # usage tracking must never break the app


def _entry(data: dict, provider: str, model: str) -> dict:
    e = data.setdefault(provider, {}).setdefault(model, {})
    today = _today(provider)
    if e.get("date") != today:  # new day: reset the daily counters but keep learned limits
        e.update(date=today, requests=0, input_tokens=0, output_tokens=0, limited_until=None)
    return e


def _int(v) -> int | None:
    try:
        return int(float(str(v).strip().rstrip("s")))
    except (TypeError, ValueError):
        return None


def record_success(provider: str, model: str, usage: dict | None = None, headers=None, path: Path | None = None) -> None:
    """Count one successful request and read any rate-limit headers the provider sent."""
    with _LOCK:
        data = _load(path)
        e = _entry(data, provider, model)
        e["requests"] = e.get("requests", 0) + 1
        u = usage or {}
        e["input_tokens"] = e.get("input_tokens", 0) + int(u.get("prompt_tokens") or u.get("input_tokens") or 0)
        e["output_tokens"] = e.get("output_tokens", 0) + int(u.get("completion_tokens") or u.get("output_tokens") or 0)
        if headers is not None:
            h = {k.lower(): v for k, v in dict(headers).items()}
            # Groq / OpenAI style
            for src, dst in (("x-ratelimit-remaining-requests", "remaining_requests"), ("x-ratelimit-limit-requests", "limit_requests"),
                             ("x-ratelimit-remaining-tokens", "remaining_tokens"), ("x-ratelimit-limit-tokens", "limit_tokens"),
                             # OpenRouter
                             ("x-ratelimit-remaining", "remaining_requests"), ("x-ratelimit-limit", "limit_requests"),
                             # Anthropic
                             ("anthropic-ratelimit-requests-remaining", "remaining_requests"),
                             ("anthropic-ratelimit-requests-limit", "limit_requests"),
                             ("anthropic-ratelimit-tokens-remaining", "remaining_tokens"),
                             ("anthropic-ratelimit-tokens-limit", "limit_tokens")):
                if src in h and _int(h[src]) is not None:
                    e[dst] = _int(h[src])
            reset = h.get("x-ratelimit-reset-requests") or h.get("anthropic-ratelimit-requests-reset")
            if reset:
                e["reset_requests"] = str(reset)[:40]
            if any(k in h for k in ("x-ratelimit-remaining-requests", "x-ratelimit-remaining", "anthropic-ratelimit-requests-remaining")):
                e["header_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        e["last_used"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        _save(data, path)


def record_limit_error(provider: str, model: str, error_text: str, path: Path | None = None) -> dict:
    """Learn limits from a 429 error (Gemini includes quotaId/quotaValue). Returns what was learned."""
    learned = {}
    for qid, val in re.findall(r"quotaId['\"]?\s*:\s*['\"]([^'\"]+)['\"].{0,400}?quotaValue['\"]?\s*:\s*['\"]?(\d+)", error_text, re.DOTALL):
        if "PerDay" in qid:
            learned["limit_day"] = int(val)
        elif "PerMinute" in qid and "Token" not in qid:
            learned["limit_minute"] = int(val)
    with _LOCK:
        data = _load(path)
        e = _entry(data, provider, model)
        e.update(learned)
        if "PerDay" in error_text:
            e["limited_until"] = "tomorrow"
            if e.get("limit_day"):
                e["requests"] = max(e.get("requests", 0), e["limit_day"])
        e["last_limit_error"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        _save(data, path)
    return learned


def status(provider: str, model: str, path: Path | None = None) -> dict:
    """What to show the user for one provider/model."""
    if provider == "ollama":
        return {"kind": "unlimited", "text": "Unlimited — runs on this computer", "used": None, "limit": None, "left": None}
    data = _load(path)
    e = data.get(provider, {}).get(model, {})
    if e.get("date") != _today(provider):
        e = {"requests": 0, "limit_day": e.get("limit_day"), "limit_minute": e.get("limit_minute")}
    used = e.get("requests", 0)
    if provider == "anthropic":
        text = f"Paid per use — {used} request(s) today"
        if e.get("remaining_requests") is not None:
            text += f"; rate limit: {e['remaining_requests']:,} requests left this minute"
        return {"kind": "paid", "text": text, "used": used, "limit": None, "left": None}
    if e.get("remaining_requests") is not None and e.get("limit_requests"):
        left, lim = e["remaining_requests"], e["limit_requests"]
        return {"kind": "exact", "text": f"{left:,} of {lim:,} free requests left (reported by the provider)",
                "used": lim - left, "limit": lim, "left": left}
    limit = e.get("limit_day") or KNOWN_DAILY_LIMITS.get((provider, model))
    if e.get("limited_until") == "tomorrow":
        return {"kind": "exhausted", "text": f"Daily free limit used up ({limit or used} requests) — resets tomorrow",
                "used": used, "limit": limit, "left": 0}
    if limit:
        left = max(0, limit - used)
        return {"kind": "counted", "text": f"{used} of {limit} free requests used today · about {left} left",
                "used": used, "limit": limit, "left": left}
    return {"kind": "unknown", "text": f"{used} request(s) today · free daily limit not known yet (shown once reached)",
            "used": used, "limit": None, "left": None}


def summary(path: Path | None = None) -> list[dict]:
    """Rows for every provider/model used so far (for the Settings page)."""
    out = []
    for provider, models in _load(path).items():
        for model in models:
            s = status(provider, model, path)
            out.append({"provider": provider, "model": model, **s})
    return out
