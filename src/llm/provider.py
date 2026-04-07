from __future__ import annotations

import os
import time
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


# ─────────────────────────────────────────────
# Response
# ─────────────────────────────────────────────

@dataclass
class LLMResponse:
    content: str
    model: str
    provider: str
    input_tokens: int   = 0
    output_tokens: int  = 0
    cost_usd: float     = 0.0
    duration_seconds: float = 0.0
    success: bool       = True
    error: str          = ""

    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


# ─────────────────────────────────────────────
# Pricing
# ─────────────────────────────────────────────

_PRICING: Dict[str, Dict[str, float]] = {
    "groq": {
        "llama-3.3-70b-versatile": 0.59,
        "llama-3.1-8b-instant":    0.05,
        "mixtral-8x7b-32768":      0.24,
        "gemma2-9b-it":            0.20,
        "default":                 0.59,
    },
    "openai": {
        "gpt-4o":      5.00,
        "gpt-4o-mini": 0.15,
        "gpt-4-turbo": 10.00,
        "default":     5.00,
    },
    "anthropic": {
        "claude-3-5-sonnet-20241022": 3.00,
        "claude-3-5-haiku-20241022":  0.80,
        "claude-3-opus-20240229":     15.00,
        "default":                    3.00,
    },
    "ollama":  {"default": 0.0},
    "openai_compatible": {"default": 0.0},
}


def _estimate_cost(provider: str, model: str, total_tokens: int) -> float:
    p    = _PRICING.get(provider.lower(), {"default": 0.0})
    rate = p.get(model, p.get("default", 0.0))
    return round((total_tokens / 1_000_000) * rate, 8)


# ─────────────────────────────────────────────
# Base
# ─────────────────────────────────────────────

class BaseProvider(ABC):

    def __init__(self, model: str, max_retries: int = 3, timeout: int = 60):
        self.model       = model
        self.max_retries = max_retries
        self.timeout     = timeout

    @abstractmethod
    def _call(self, messages: List[Dict], **kwargs) -> LLMResponse:
        pass

    def chat(self, messages: List[Dict], **kwargs) -> LLMResponse:
        last_error = ""
        for attempt in range(self.max_retries):
            try:
                return self._call(messages, **kwargs)
            except Exception as e:
                last_error = str(e)
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)
        return LLMResponse(
            content="", model=self.model, provider=self.__class__.__name__,
            success=False, error=last_error,
        )

    def think(self, prompt: str, system: str = "") -> LLMResponse:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        return self.chat(messages)

    def plan(self, task: str, context: str = "") -> Dict[str, Any]:
        system = (
            "You are a Senior AI Engineer. Respond ONLY with valid JSON. "
            "No markdown, no explanation, just the JSON object."
        )
        prompt = f"Task: {task}\nContext: {context}\n\nGenerate a training plan as JSON with keys: architecture, data, training, objective, rationale."
        resp   = self.think(prompt, system=system)
        if not resp.success:
            return {"error": resp.error}
        try:
            text = resp.content.strip()
            if text.startswith("```"):
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]
            return json.loads(text.strip())
        except Exception:
            return {"raw": resp.content}

    def reflect(self, episode_summary: str) -> Dict[str, Any]:
        system = "You are a self-improving AI agent. Respond ONLY with valid JSON."
        prompt = (
            f"Episode summary:\n{episode_summary}\n\n"
            "Reflect on this episode. Respond with JSON: "
            "{next_improvement, gene_hint, failure_mode, confidence}"
        )
        resp = self.think(prompt, system=system)
        if not resp.success:
            return {"next_improvement": "Unknown", "gene_hint": "", "confidence": 0.5}
        try:
            text = resp.content.strip()
            if text.startswith("```"):
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]
            return json.loads(text.strip())
        except Exception:
            return {"next_improvement": resp.content[:200], "gene_hint": "", "confidence": 0.5}

    def trim_context(self, messages: List[Dict], max_tokens: int = 4000) -> List[Dict]:
        total = sum(len(m.get("content", "")) // 4 for m in messages)
        while total > max_tokens and len(messages) > 1:
            if messages[0].get("role") == "system":
                messages.pop(1)
            else:
                messages.pop(0)
            total = sum(len(m.get("content", "")) // 4 for m in messages)
        return messages


# ─────────────────────────────────────────────
# Providers
# ─────────────────────────────────────────────

class GroqProvider(BaseProvider):

    def __init__(self, api_key: str = "", model: str = "llama-3.3-70b-versatile", **kwargs):
        super().__init__(model, **kwargs)
        self.api_key  = api_key or os.getenv("GROQ_API_KEY", "")
        self._client  = None

    def _get_client(self):
        if self._client is None:
            from groq import Groq
            self._client = Groq(api_key=self.api_key)
        return self._client

    def _call(self, messages: List[Dict], **kwargs) -> LLMResponse:
        start  = time.time()
        client = self._get_client()
        resp   = client.chat.completions.create(
            model=self.model,
            messages=messages,
            timeout=self.timeout,
            **kwargs,
        )
        duration = time.time() - start
        it = resp.usage.prompt_tokens
        ot = resp.usage.completion_tokens
        return LLMResponse(
            content=resp.choices[0].message.content,
            model=self.model,
            provider="groq",
            input_tokens=it,
            output_tokens=ot,
            cost_usd=_estimate_cost("groq", self.model, it + ot),
            duration_seconds=round(duration, 3),
        )


class OpenAIProvider(BaseProvider):

    def __init__(self, api_key: str = "", model: str = "gpt-4o-mini", **kwargs):
        super().__init__(model, **kwargs)
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "")
        self._client = None

    def _get_client(self):
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(api_key=self.api_key)
        return self._client

    def _call(self, messages: List[Dict], **kwargs) -> LLMResponse:
        start  = time.time()
        client = self._get_client()
        resp   = client.chat.completions.create(
            model=self.model,
            messages=messages,
            timeout=self.timeout,
            **kwargs,
        )
        duration = time.time() - start
        it = resp.usage.prompt_tokens
        ot = resp.usage.completion_tokens
        return LLMResponse(
            content=resp.choices[0].message.content,
            model=self.model,
            provider="openai",
            input_tokens=it,
            output_tokens=ot,
            cost_usd=_estimate_cost("openai", self.model, it + ot),
            duration_seconds=round(duration, 3),
        )


class AnthropicProvider(BaseProvider):

    def __init__(self, api_key: str = "", model: str = "claude-3-5-haiku-20241022", **kwargs):
        super().__init__(model, **kwargs)
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY", "")
        self._client = None

    def _get_client(self):
        if self._client is None:
            import anthropic
            self._client = anthropic.Anthropic(api_key=self.api_key)
        return self._client

    def _call(self, messages: List[Dict], **kwargs) -> LLMResponse:
        start   = time.time()
        client  = self._get_client()
        system  = ""
        msgs    = []
        for m in messages:
            if m["role"] == "system":
                system = m["content"]
            else:
                msgs.append(m)
        resp = client.messages.create(
            model=self.model,
            max_tokens=1024,
            system=system,
            messages=msgs,
        )
        duration = time.time() - start
        it = resp.usage.input_tokens
        ot = resp.usage.output_tokens
        return LLMResponse(
            content=resp.content[0].text,
            model=self.model,
            provider="anthropic",
            input_tokens=it,
            output_tokens=ot,
            cost_usd=_estimate_cost("anthropic", self.model, it + ot),
            duration_seconds=round(duration, 3),
        )


class OllamaProvider(BaseProvider):

    def __init__(self, model: str = "mistral", base_url: str = "http://localhost:11434", **kwargs):
        super().__init__(model, **kwargs)
        self.base_url = base_url

    def _call(self, messages: List[Dict], **kwargs) -> LLMResponse:
        import urllib.request
        start   = time.time()
        payload = json.dumps({"model": self.model, "messages": messages, "stream": False}).encode()
        req     = urllib.request.Request(
            f"{self.base_url}/api/chat",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            data = json.loads(r.read())
        duration = time.time() - start
        content  = data.get("message", {}).get("content", "")
        return LLMResponse(
            content=content,
            model=self.model,
            provider="ollama",
            duration_seconds=round(duration, 3),
        )


class OpenAICompatibleProvider(BaseProvider):

    def __init__(self, base_url: str, api_key: str = "", model: str = "default", **kwargs):
        super().__init__(model, **kwargs)
        self.base_url = base_url
        self.api_key  = api_key or os.getenv("LLM_API_KEY", "")
        self._client  = None

    def _get_client(self):
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        return self._client

    def _call(self, messages: List[Dict], **kwargs) -> LLMResponse:
        start  = time.time()
        client = self._get_client()
        resp   = client.chat.completions.create(
            model=self.model,
            messages=messages,
            **kwargs,
        )
        duration = time.time() - start
        it = resp.usage.prompt_tokens if resp.usage else 0
        ot = resp.usage.completion_tokens if resp.usage else 0
        return LLMResponse(
            content=resp.choices[0].message.content,
            model=self.model,
            provider="openai_compatible",
            input_tokens=it,
            output_tokens=ot,
            duration_seconds=round(duration, 3),
        )


# ─────────────────────────────────────────────
# ModelProvider — unified interface
# ─────────────────────────────────────────────

class ModelProvider:

    def __init__(
        self,
        provider: str   = "",
        model: str      = "",
        api_key: str    = "",
        base_url: str   = "",
        fallback: Optional["ModelProvider"] = None,
        max_retries: int = 3,
    ):
        self.provider_name = provider or os.getenv("LLM_PROVIDER", "groq")
        self.model_name    = model    or os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")
        self.api_key       = api_key  or os.getenv("LLM_API_KEY", "")
        self.base_url      = base_url or os.getenv("LLM_BASE_URL", "")
        self.fallback      = fallback
        self._backend      = self._build(self.provider_name, self.model_name, max_retries)

    def _build(self, provider: str, model: str, max_retries: int) -> BaseProvider:
        p = provider.lower()
        if p == "groq":
            return GroqProvider(api_key=self.api_key, model=model, max_retries=max_retries)
        if p == "openai":
            return OpenAIProvider(api_key=self.api_key, model=model, max_retries=max_retries)
        if p == "anthropic":
            return AnthropicProvider(api_key=self.api_key, model=model, max_retries=max_retries)
        if p == "ollama":
            return OllamaProvider(model=model, base_url=self.base_url or "http://localhost:11434", max_retries=max_retries)
        return OpenAICompatibleProvider(
            base_url=self.base_url, api_key=self.api_key, model=model, max_retries=max_retries
        )

    def chat(self, messages: List[Dict], **kwargs) -> LLMResponse:
        resp = self._backend.chat(messages, **kwargs)
        if not resp.success and self.fallback:
            resp = self.fallback.chat(messages, **kwargs)
        return resp

    def think(self, prompt: str, system: str = "") -> LLMResponse:
        resp = self._backend.think(prompt, system=system)
        if not resp.success and self.fallback:
            resp = self.fallback.think(prompt, system=system)
        return resp

    def plan(self, task: str, context: str = "") -> Dict[str, Any]:
        return self._backend.plan(task, context)

    def reflect(self, episode_summary: str) -> Dict[str, Any]:
        return self._backend.reflect(episode_summary)

    def record_to_episode(self, episode: Any, resp: LLMResponse, purpose: str) -> None:
        if episode and hasattr(episode, "record_llm_call"):
            episode.record_llm_call(
                purpose=purpose,
                prompt_summary="",
                response_summary=resp.content[:200],
                model=resp.model,
                provider=resp.provider,
                input_tokens=resp.input_tokens,
                output_tokens=resp.output_tokens,
                duration_seconds=resp.duration_seconds,
                success=resp.success,
                error=resp.error,
            )

    @classmethod
    def from_env(cls) -> "ModelProvider":
        return cls(
            provider=os.getenv("LLM_PROVIDER", "groq"),
            model=os.getenv("LLM_MODEL", "llama-3.3-70b-versatile"),
            api_key=os.getenv("LLM_API_KEY", ""),
            base_url=os.getenv("LLM_BASE_URL", ""),
        )

    def __repr__(self) -> str:
        return f"ModelProvider(provider={self.provider_name}, model={self.model_name})"