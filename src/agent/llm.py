"""LLM provider abstraction.

The agent loop talks to an ``LLMClient`` that returns plain-dict content blocks (text / tool_use), so
providers are swappable and tests can use a scripted fake. Supported providers:

* ``ollama``      – open-source models running on this computer (free, nothing leaves the machine)
* ``gemini``      – Google Gemini through an AI Studio key (free tier available)
* ``groq``        – open-source models hosted by Groq (free tier available)
* ``openrouter``  – many models incl. free open-source ones
* ``openai_compatible`` – any other OpenAI-compatible endpoint (LM Studio, vLLM, …)
* ``anthropic``   – Claude (paid)

With no provider available the app runs the offline guided agent instead.
"""
from __future__ import annotations

import json
import os
import re
import uuid
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Protocol

DEFAULT_MODEL = "claude-opus-5-5"
DEFAULT_EFFORT = "medium"


class LLMUnavailable(Exception):
    """The provider couldn't be reached or refused the request; the caller falls back to guided mode."""


@dataclass
class LLMReply:
    content: list[dict]
    stop_reason: str
    usage: dict = field(default_factory=dict)


class LLMClient(Protocol):
    name: str

    def create(self, system: str, messages: list[dict], tools: list[dict], allow_tools: bool = True) -> LLMReply: ...


# --------------------------------------------------------------------------- provider presets

PRESETS: dict[str, dict] = {
    "ollama": {"label": "Ollama (open-source model on this computer)", "base_url": "http://localhost:11434",
               "key_env": [], "model": "qwen3:4b", "local": True, "cost": "Free",
               "data_note": "Runs entirely on this computer — nothing is sent anywhere."},
    "gemini": {"label": "Google Gemini (AI Studio)", "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
               "key_env": ["GEMINI_API_KEY", "GOOGLE_API_KEY"], "model": "gemini-flash-lite-latest", "local": False,
               "cost": "Free tier with daily limits",
               "data_note": "Sent to Google. On the free tier Google may use what you send to improve its products."},
    "groq": {"label": "Groq (hosted open-source models)", "base_url": "https://api.groq.com/openai/v1",
             "key_env": ["GROQ_API_KEY"], "model": "llama-3.3-70b-versatile", "local": False,
             "cost": "Free tier with rate limits", "data_note": "Sent to Groq."},
    "openrouter": {"label": "OpenRouter", "base_url": "https://openrouter.ai/api/v1", "key_env": ["OPENROUTER_API_KEY"],
                   "model": "meta-llama/llama-3.3-70b-instruct:free", "local": False,
                   "cost": "Free for ':free' models (rate limited)", "data_note": "Sent to OpenRouter and the model's host."},
    "openai_compatible": {"label": "OpenAI-compatible endpoint", "base_url": "", "key_env": ["DASHBOARD_LLM_API_KEY"],
                          "model": "", "local": False, "cost": "Depends on the service",
                          "data_note": "Sent to the configured endpoint."},
    "anthropic": {"label": "Claude (Anthropic)", "base_url": "", "key_env": ["ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN"],
                  "model": DEFAULT_MODEL, "local": False, "cost": "Paid per use", "data_note": "Sent to Anthropic."},
}
AUTO_ORDER = ["anthropic", "gemini", "groq", "openrouter"]


def _env(name: str) -> str:
    return os.environ.get(name, "").strip()


def _key_for(preset: dict) -> str:
    return next((_env(k) for k in preset["key_env"] if _env(k)), "")


@lru_cache(maxsize=4)
def _ollama_models(base_url: str) -> tuple[str, ...]:
    """Models installed in a reachable local Ollama (empty when Ollama isn't running)."""
    try:
        import requests
        r = requests.get(base_url.rstrip("/") + "/api/tags", timeout=1.0)
        r.raise_for_status()
        return tuple(m["name"] for m in r.json().get("models", []))
    except Exception:  # noqa: BLE001 - not running / not installed
        return ()


def _choose_provider() -> tuple[str | None, str | None]:
    """(provider, reason-if-none). An explicit DASHBOARD_LLM_PROVIDER always wins."""
    from .. import config

    wanted = _env("DASHBOARD_LLM_PROVIDER").lower() or "auto"
    if wanted == "none":
        return None, "AI provider switched off (DASHBOARD_LLM_PROVIDER=none)."
    if wanted != "auto":
        if wanted not in PRESETS:
            return None, f"Unknown DASHBOARD_LLM_PROVIDER '{wanted}'."
        preset = PRESETS[wanted]
        if preset["key_env"] and not _key_for(preset) and wanted not in ("openai_compatible",):
            return None, f"{preset['label']} needs {' or '.join(preset['key_env'])}."
        if wanted == "ollama" and not _ollama_models(_env("DASHBOARD_LLM_BASE_URL") or preset["base_url"]):
            return None, "Ollama isn't running on this computer. Start the Ollama app, then reload."
        if wanted == "openai_compatible" and not _env("DASHBOARD_LLM_BASE_URL"):
            return None, "Set DASHBOARD_LLM_BASE_URL for the OpenAI-compatible provider."
        return wanted, None
    for name in AUTO_ORDER:
        if _key_for(PRESETS[name]):
            return name, None
    # Ollama is never picked automatically: on a laptop without a graphics card each step can take minutes,
    # so it must be chosen deliberately with DASHBOARD_LLM_PROVIDER=ollama.
    del config
    return None, "No AI provider is configured (add a free GROQ_API_KEY or GEMINI_API_KEY, or set DASHBOARD_LLM_PROVIDER=ollama)."


def provider_status() -> dict:
    """What the UI should tell the user about the AI provider (never includes secrets)."""
    name, reason = _choose_provider()
    if name is None:
        return {"enabled": False, "name": None, "provider": None, "model": None, "local": False, "cost": None,
                "data_note": None, "effort": None, "reason": reason}
    preset = PRESETS[name]
    model = _env("DASHBOARD_LLM_MODEL") or preset["model"]
    if name == "ollama":
        installed = _ollama_models(_env("DASHBOARD_LLM_BASE_URL") or preset["base_url"])
        if model not in installed and installed:
            model = installed[0]
    return {"enabled": True, "name": name, "provider": preset["label"], "model": model, "local": preset["local"],
            "cost": preset["cost"], "data_note": preset["data_note"],
            "effort": _env("DASHBOARD_LLM_EFFORT").lower() or DEFAULT_EFFORT, "reason": None}


# --------------------------------------------------------------------------- shared conversion helpers

_THINK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def _clean_text(text: str | None) -> str:
    return _THINK.sub("", text or "").strip()


def _sanitize_schema(schema):
    """Drop JSON-Schema features some providers reject (we validate every tool input locally anyway)."""
    if isinstance(schema, dict):
        out = {}
        for k, v in schema.items():
            if k in ("additionalProperties", "strict"):
                continue
            if k == "enum" and isinstance(v, list):
                v = [e for e in v if e != ""] or v
            out[k] = _sanitize_schema(v)
        return out
    if isinstance(schema, list):
        return [_sanitize_schema(s) for s in schema]
    return schema


def to_openai_messages(system: str, messages: list[dict]) -> list[dict]:
    """Convert the agent's (Anthropic-shaped) transcript to OpenAI chat messages."""
    out: list[dict] = [{"role": "system", "content": system}]
    for msg in messages:
        content = msg["content"]
        if isinstance(content, str):
            out.append({"role": msg["role"], "content": content})
            continue
        if msg["role"] == "assistant":
            text = "\n".join(b.get("text", "") for b in content if b.get("type") == "text").strip()
            calls = []
            for b in content:
                if b.get("type") != "tool_use":
                    continue
                call = {"id": b["id"], "type": "function",
                        "function": {"name": b["name"], "arguments": json.dumps(b.get("input") or {})}}
                if b.get("provider_extra"):  # e.g. Gemini's thought_signature, which must be echoed back unchanged
                    call["extra_content"] = b["provider_extra"]
                calls.append(call)
            entry: dict = {"role": "assistant", "content": text or None}
            if calls:
                entry["tool_calls"] = calls
            if text or calls:
                out.append(entry)
            continue
        texts = []
        for b in content:
            if b.get("type") == "tool_result":
                body = b.get("content")
                body = body if isinstance(body, str) else json.dumps(body)
                if b.get("is_error"):
                    body = "ERROR: " + body
                out.append({"role": "tool", "tool_call_id": b["tool_use_id"], "content": body})
            elif b.get("type") == "text":
                texts.append(b.get("text", ""))
        if texts:
            out.append({"role": "user", "content": "\n\n".join(texts)})
    return out


def _parse_args(raw) -> dict:
    if isinstance(raw, dict):
        return raw
    try:
        val = json.loads(raw or "{}")
        return val if isinstance(val, dict) else {"_invalid_arguments": raw}
    except (TypeError, json.JSONDecodeError):
        return {"_invalid_arguments": str(raw)[:500]}  # rejected by local validation → model sees the error


def _blocks(text: str | None, calls: list[tuple]) -> LLMReply:
    """calls: (id, name, arguments[, provider_extra])."""
    content: list[dict] = []
    t = _clean_text(text)
    if t:
        content.append({"type": "text", "text": t})
    for call in calls:
        cid, name, args = call[:3]
        block = {"type": "tool_use", "id": cid or f"call_{uuid.uuid4().hex[:12]}", "name": name, "input": _parse_args(args)}
        if len(call) > 3 and call[3]:
            block["provider_extra"] = call[3]
        content.append(block)
    return LLMReply(content=content, stop_reason="tool_use" if calls else "end_turn")


def _retry_delay(exc) -> float:
    """Seconds the provider asked us to wait (Gemini puts it in the error body; others use Retry-After)."""
    try:
        header = exc.response.headers.get("retry-after")
        if header:
            return float(header)
    except (AttributeError, ValueError):
        pass
    m = re.search(r"retryDelay['\"]?\s*:\s*['\"]?(\d+(?:\.\d+)?)s", str(exc))
    return float(m.group(1)) if m else 20.0


# --------------------------------------------------------------------------- OpenAI-compatible (Gemini, Groq, OpenRouter, …)

class OpenAICompatLLM:
    """Any service speaking the OpenAI chat-completions API with function calling."""

    def __init__(self, name: str, base_url: str, api_key: str, model: str, client=None, sleep=None, max_wait: float = 130.0):
        import time

        import openai

        self._openai = openai
        self.sleep = sleep or time.sleep
        self.max_wait = max_wait
        self.name, self.model = name, model
        self.label = PRESETS.get(name, PRESETS["openai_compatible"])["label"]
        self.client = client or openai.OpenAI(base_url=base_url, api_key=api_key or "none", timeout=180.0, max_retries=0)

    def create(self, system: str, messages: list[dict], tools: list[dict], allow_tools: bool = True) -> LLMReply:
        o = self._openai
        kwargs: dict = {"model": self.model, "messages": to_openai_messages(system, messages), "max_tokens": 4096}
        if tools:
            kwargs["tools"] = [{"type": "function", "function": {"name": t["name"], "description": t["description"],
                                                                 "parameters": _sanitize_schema(t["input_schema"])}} for t in tools]
            kwargs["tool_choice"] = "auto" if allow_tools else "none"
        waited = 0.0
        while True:
            try:
                resp = self.client.chat.completions.create(**kwargs)
                break
            except o.RateLimitError as exc:
                # Free tiers allow only a few requests per minute: wait as asked (up to ~2 minutes in total).
                delay = min(max(_retry_delay(exc), 2.0), 60.0)
                if "PerDay" in str(exc):
                    raise LLMUnavailable(f"{self.label}'s free daily limit for '{self.model}' is used up (it resets "
                                         "tomorrow). Another model may still have quota — see DASHBOARD_LLM_MODEL in the README.") from exc
                if waited + delay > self.max_wait:
                    raise LLMUnavailable(f"{self.label} is limiting how fast requests can be sent. Wait a minute and try again.") from exc
                self.sleep(delay + 1)
                waited += delay + 1
            except o.AuthenticationError as exc:
                raise LLMUnavailable(f"{self.label} didn't accept the API key.") from exc
            except o.NotFoundError as exc:
                raise LLMUnavailable(f"{self.label} doesn't offer the model '{self.model}'. Set DASHBOARD_LLM_MODEL in .env.") from exc
            except o.BadRequestError as exc:
                raise LLMUnavailable(f"{self.label} rejected the request: {getattr(exc, 'message', exc)}") from exc
            except o.APIConnectionError as exc:
                raise LLMUnavailable(f"Couldn't connect to {self.label}. Check the internet connection.") from exc
            except o.APIStatusError as exc:
                raise LLMUnavailable(f"{self.label} returned an error ({exc.status_code}).") from exc
        if not resp.choices:
            raise LLMUnavailable(f"{self.label} returned an empty answer.")
        msg = resp.choices[0].message
        calls = [(tc.id, tc.function.name, tc.function.arguments, (getattr(tc, "model_extra", None) or {}).get("extra_content"))
                 for tc in (msg.tool_calls or []) if tc.function]
        reply = _blocks(msg.content, calls)
        reply.usage = resp.usage.model_dump() if getattr(resp, "usage", None) else {}
        return reply


# --------------------------------------------------------------------------- Ollama (native API: bigger context, thinking off)

class OllamaLLM:
    """Open-source models running locally in Ollama. Uses the native /api/chat endpoint so the context window
    can be enlarged (small defaults would silently cut the prompt) and reasoning mode can be switched off."""

    name = "ollama"

    def __init__(self, model: str, base_url: str = PRESETS["ollama"]["base_url"], num_ctx: int | None = None,
                 think: bool = False, session=None):
        import requests

        self._requests = requests
        self.session = session or requests.Session()
        self.model, self.base_url = model, base_url.rstrip("/")
        self.num_ctx = num_ctx or int(_env("DASHBOARD_OLLAMA_NUM_CTX") or 16384)
        self.think = think or _env("DASHBOARD_OLLAMA_THINK").lower() in ("1", "true", "yes")

    def _messages(self, system: str, messages: list[dict]) -> list[dict]:
        out = []
        for m in to_openai_messages(system, messages):
            m = dict(m)
            if m.get("tool_calls"):
                m["tool_calls"] = [{"function": {"name": c["function"]["name"], "arguments": json.loads(c["function"]["arguments"])}}
                                   for c in m["tool_calls"]]
            if m["role"] == "tool":
                m.pop("tool_call_id", None)
            if m.get("content") is None:
                m["content"] = ""
            out.append(m)
        return out

    def create(self, system: str, messages: list[dict], tools: list[dict], allow_tools: bool = True) -> LLMReply:
        body: dict = {"model": self.model, "messages": self._messages(system, messages), "stream": False,
                      "think": self.think, "options": {"num_ctx": self.num_ctx, "temperature": 0.2}}
        if tools and allow_tools:
            body["tools"] = [{"type": "function", "function": {"name": t["name"], "description": t["description"],
                                                               "parameters": _sanitize_schema(t["input_schema"])}} for t in tools]
        try:
            r = self.session.post(f"{self.base_url}/api/chat", json=body, timeout=600)
        except self._requests.RequestException as exc:
            raise LLMUnavailable("Couldn't reach Ollama on this computer. Is the Ollama app running?") from exc
        if r.status_code == 404:
            raise LLMUnavailable(f"The model '{self.model}' isn't installed in Ollama. Run: ollama pull {self.model}")
        if r.status_code >= 400:
            text = r.text[:300]
            if "does not support tools" in text:
                raise LLMUnavailable(f"The model '{self.model}' can't use tools. Try qwen3:4b or llama3.2.")
            raise LLMUnavailable(f"Ollama returned an error ({r.status_code}): {text}")
        msg = r.json().get("message", {})
        calls = [(None, c["function"]["name"], c["function"].get("arguments") or {}) for c in msg.get("tool_calls") or []
                 if c.get("function", {}).get("name")]
        return _blocks(msg.get("content"), calls)


# --------------------------------------------------------------------------- Anthropic (paid)

class AnthropicLLM:
    """Claude through the Messages API, with strict tools and server-side refusal fallbacks."""

    name = "anthropic"

    def __init__(self, model: str = DEFAULT_MODEL, effort: str = DEFAULT_EFFORT, client=None):
        import anthropic

        self._anthropic = anthropic
        self.client = client or anthropic.Anthropic(max_retries=2, timeout=120.0)
        self.model, self.effort = model, effort
        self._strict = True
        self._fallbacks = True

    def _request(self, system: str, messages: list[dict], tools: list[dict], allow_tools: bool) -> dict:
        tool_defs = [t if self._strict else {k: v for k, v in t.items() if k != "strict"} for t in tools]
        # Fields added for other providers (e.g. Gemini's provider_extra) aren't part of the Messages API.
        messages = [m if isinstance(m["content"], str) else
                    {**m, "content": [{k: v for k, v in b.items() if k != "provider_extra"} for b in m["content"]]}
                    for m in messages]
        kwargs = dict(
            model=self.model, max_tokens=16000, messages=messages,
            # Stable system prompt first so it can be cached across turns.
            system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
            output_config={"effort": self.effort},
        )
        if tool_defs:
            kwargs["tools"] = tool_defs
            kwargs["tool_choice"] = {"type": "auto"} if allow_tools else {"type": "none"}
        if self._fallbacks:
            kwargs["betas"] = ["server-side-fallback-2026-07-01"]
            kwargs["fallbacks"] = "default"
        return kwargs

    def create(self, system: str, messages: list[dict], tools: list[dict], allow_tools: bool = True) -> LLMReply:
        a = self._anthropic
        for _attempt in range(3):
            kwargs = self._request(system, messages, tools, allow_tools)
            try:
                resp = self.client.beta.messages.create(**kwargs)
            except a.BadRequestError as exc:
                text = str(exc).lower()
                if self._strict and ("strict" in text or "schema" in text):
                    self._strict = False  # account/schema doesn't support strict tools: validation still happens locally
                    continue
                if self._fallbacks and ("fallback" in text or "beta" in text):
                    self._fallbacks = False
                    continue
                raise LLMUnavailable(f"The AI service rejected the request: {getattr(exc, 'message', exc)}") from exc
            except a.AuthenticationError as exc:
                raise LLMUnavailable("The AI service didn't accept the API key.") from exc
            except a.PermissionDeniedError as exc:
                raise LLMUnavailable("The API key isn't allowed to use this model.") from exc
            except a.RateLimitError as exc:
                raise LLMUnavailable("The AI service is busy (rate limit). Try again in a minute.") from exc
            except a.APIStatusError as exc:
                raise LLMUnavailable(f"The AI service returned an error ({exc.status_code}).") from exc
            except a.APIConnectionError as exc:
                raise LLMUnavailable("Couldn't connect to the AI service. Check the internet connection.") from exc
            content = [b.model_dump(mode="json", exclude_none=True) for b in resp.content]
            usage = resp.usage.model_dump(mode="json", exclude_none=True) if getattr(resp, "usage", None) else {}
            return LLMReply(content=content, stop_reason=resp.stop_reason or "end_turn", usage=usage)
        raise LLMUnavailable("The AI service rejected the request format.")


# --------------------------------------------------------------------------- choosing provider + model in the app

_NOT_CHAT = ("embed", "tts", "image", "audio", "live", "computer-use", "transcribe", "nano-banana", "robotics", "omni",
             "aqa", "imagen", "veo", "whisper", "guard", "vision-preview", "-native-")
ANTHROPIC_MODELS = ("claude-opus-5-5", "claude-sonnet-5-5", "claude-haiku-5-5")


def available_providers() -> list[str]:
    """Providers the user can pick right now (a key is configured, or Ollama is running locally)."""
    from .. import config

    out = [n for n in ("gemini", "groq", "openrouter", "anthropic") if _key_for(PRESETS[n])]
    if _env("DASHBOARD_LLM_BASE_URL") and _env("DASHBOARD_LLM_PROVIDER").lower() == "openai_compatible":
        out.append("openai_compatible")
    if not config.ONLINE and _ollama_models(PRESETS["ollama"]["base_url"]):
        out.append("ollama")
    return out


@lru_cache(maxsize=16)
def list_models(name: str) -> tuple[str, ...]:
    """Chat models the provider offers to this key (live list where possible), default first."""
    preset = PRESETS[name]
    default = _env("DASHBOARD_LLM_MODEL") if provider_status().get("name") == name and _env("DASHBOARD_LLM_MODEL") else preset["model"]
    models: list[str] = []
    try:
        if name == "ollama":
            models = list(_ollama_models(_env("DASHBOARD_LLM_BASE_URL") or preset["base_url"]))
        elif name == "anthropic":
            models = list(ANTHROPIC_MODELS)
        else:
            import openai
            client = openai.OpenAI(base_url=_env("DASHBOARD_LLM_BASE_URL") or preset["base_url"], api_key=_key_for(preset) or "none",
                                   timeout=10.0, max_retries=0)
            ids = [m.id.removeprefix("models/") for m in client.models.list()]
            models = sorted(i for i in ids if not any(x in i for x in _NOT_CHAT))
            if name == "gemini":
                models = [i for i in models if i.startswith(("gemini", "gemma"))]
            if name == "openrouter":
                models = [i for i in models if i.endswith(":free")]
    except Exception:  # noqa: BLE001 - offline / bad key: fall back to the default only
        models = []
    if default and default not in models:
        models.insert(0, default)
    elif default:
        models.remove(default)
        models.insert(0, default)
    return tuple(models[:80])


def describe_choice(name: str | None, model: str | None) -> str:
    if not name:
        return "Guided assistant (offline)"
    return f"{PRESETS[name]['label']} · {model or PRESETS[name]['model']}"


# --------------------------------------------------------------------------- factory

def build_llm(name: str, model: str) -> LLMClient | None:
    """Client for an explicitly chosen provider + model (None if it can't be created)."""
    preset = PRESETS.get(name)
    if preset is None:
        return None
    try:
        if name == "anthropic":
            return AnthropicLLM(model, _env("DASHBOARD_LLM_EFFORT").lower() or DEFAULT_EFFORT)
        if name == "ollama":
            return OllamaLLM(model, _env("DASHBOARD_LLM_BASE_URL") or preset["base_url"])
        base = _env("DASHBOARD_LLM_BASE_URL") or preset["base_url"]
        return OpenAICompatLLM(name, base, _key_for(preset), model)
    except Exception:  # noqa: BLE001 - SDK missing or misconfigured → offline guided mode
        return None


def get_llm() -> LLMClient | None:
    status = provider_status()
    if not status["enabled"]:
        return None
    name, model = status["name"], status["model"]
    preset = PRESETS[name]
    try:
        if name == "anthropic":
            return AnthropicLLM(model, status["effort"])
        if name == "ollama":
            return OllamaLLM(model, _env("DASHBOARD_LLM_BASE_URL") or preset["base_url"])
        base = _env("DASHBOARD_LLM_BASE_URL") or preset["base_url"]
        return OpenAICompatLLM(name, base, _key_for(preset), model)
    except Exception:  # noqa: BLE001 - SDK missing or misconfigured → offline guided mode
        return None
