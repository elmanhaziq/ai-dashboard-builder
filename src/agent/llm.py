"""LLM provider abstraction.

The agent loop talks to an ``LLMClient`` that returns plain-dict content blocks, so providers are
swappable and tests can use a scripted fake. The built-in provider is Anthropic's Claude via the
official SDK. With no provider configured the app runs the offline guided agent instead.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
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


def _settings() -> tuple[str, str, str]:
    provider = os.environ.get("DASHBOARD_LLM_PROVIDER", "auto").strip().lower()
    model = os.environ.get("DASHBOARD_LLM_MODEL", DEFAULT_MODEL).strip() or DEFAULT_MODEL
    effort = os.environ.get("DASHBOARD_LLM_EFFORT", DEFAULT_EFFORT).strip().lower() or DEFAULT_EFFORT
    return provider, model, effort


def provider_status() -> dict:
    """What the UI should tell the user about the AI provider (no secrets)."""
    provider, model, effort = _settings()
    has_key = bool(os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"))
    enabled = provider == "anthropic" or (provider == "auto" and has_key)
    if provider == "none":
        enabled = False
    return {"enabled": enabled, "provider": "Anthropic (Claude)" if enabled else None, "model": model, "effort": effort,
            "reason": None if enabled else ("AI provider switched off (DASHBOARD_LLM_PROVIDER=none)." if provider == "none"
                                            else "No ANTHROPIC_API_KEY is configured.")}


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


def get_llm() -> LLMClient | None:
    status = provider_status()
    if not status["enabled"]:
        return None
    try:
        return AnthropicLLM(status["model"], status["effort"])
    except Exception:  # noqa: BLE001 - SDK missing or misconfigured → offline guided mode
        return None
