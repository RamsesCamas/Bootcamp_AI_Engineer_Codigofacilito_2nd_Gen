"""Adaptadores de proveedores de LLM.

Todos los proveedores se consumen vía su API compatible con OpenAI usando `httpx`, sin
SDKs. Cada adaptador solo sabe hablar con su proveedor; el `LLMClient` decide a quién
llamar, cuántas veces reintentar y cuándo hacer fallback.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Protocol

import httpx

from core import pricing
from core.errors import PermanentProviderError, TransientProviderError
from core.llm_client import LLMResponse

if TYPE_CHECKING:
    from pydantic import SecretStr

    from core.config import Settings

TRANSIENT_STATUS_CODES = {429, 500, 502, 503, 504}


class Provider(Protocol):
    name: str

    def generate(self, messages: list[dict], **params) -> LLMResponse: ...


class OpenAICompatibleProvider:
    """Adaptador base para cualquier API compatible con `/chat/completions` de OpenAI.

    Las subclases solo fijan `name` y `base_url` como atributos de clase. También se
    pueden pasar al constructor para sobrescribirlos (por ejemplo, otra URL de Ollama).
    """

    name: str = ""
    base_url: str = ""

    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        timeout_s: float = 30,
        *,
        name: str | None = None,
        base_url: str | None = None,
    ) -> None:
        self.name = name or self.name
        self.base_url = (base_url or self.base_url).rstrip("/")
        self.model = model
        self._api_key = api_key
        self.timeout_s = timeout_s

    def generate(self, messages: list[dict], **params) -> LLMResponse:
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        payload = {"model": self.model, "messages": messages, **params}

        start = time.perf_counter()
        try:
            response = httpx.post(
                f"{self.base_url}/chat/completions",
                json=payload,
                headers=headers,
                timeout=self.timeout_s,
            )
        except httpx.TimeoutException as e:
            raise TransientProviderError(self.name, f"timeout: {e}") from e
        except httpx.TransportError as e:
            raise TransientProviderError(self.name, f"error de conexión: {e}") from e
        latency_ms = (time.perf_counter() - start) * 1000

        if response.status_code in TRANSIENT_STATUS_CODES:
            raise TransientProviderError(self.name, _short(response), response.status_code)
        if response.status_code >= 400:
            raise PermanentProviderError(self.name, _short(response), response.status_code)

        try:
            data = response.json()
            text = data["choices"][0]["message"]["content"] or ""
        except (ValueError, KeyError, IndexError, TypeError) as e:
            raise PermanentProviderError(
                self.name, f"respuesta inesperada: {e!r}", response.status_code
            ) from e

        usage = data.get("usage") or {}
        tokens_in = int(usage.get("prompt_tokens") or 0)
        tokens_out = int(usage.get("completion_tokens") or 0)
        return LLMResponse(
            text=text,
            provider=self.name,
            model=data.get("model") or self.model,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            latency_ms=latency_ms,
            cost_usd=pricing.cost_usd(self.model, tokens_in, tokens_out, provider=self.name),
        )


def _short(response: httpx.Response, limit: int = 200) -> str:
    """Resumen corto del cuerpo de error (nunca incluye los headers de la request)."""
    return " ".join(response.text.split())[:limit] or response.reason_phrase


class GeminiProvider(OpenAICompatibleProvider):
    # Endpoint oficial compatible con OpenAI (https://ai.google.dev/gemini-api/docs/openai),
    # que Google marca como beta. Para proyectos nuevos Google recomienda su API nativa,
    # la Interactions API (/v1beta/interactions). Aquí usamos la compatible con OpenAI para
    # que los tres proveedores compartan el mismo adaptador base.
    name = "gemini"
    base_url = "https://generativelanguage.googleapis.com/v1beta/openai/"


class OllamaProvider(OpenAICompatibleProvider):
    name = "ollama"
    base_url = "http://localhost:11434/v1"


class GroqProvider(OpenAICompatibleProvider):
    name = "groq"
    base_url = "https://api.groq.com/openai/v1"


def build_provider(name: str, settings: Settings) -> Provider:
    """Crea el adaptador de un proveedor a partir de la configuración."""

    def key(secret: SecretStr | None) -> str | None:
        return secret.get_secret_value() if secret is not None else None

    timeout = settings.timeout_s
    if name == "gemini":
        return GeminiProvider(settings.gemini_model, key(settings.gemini_api_key), timeout)
    if name == "groq":
        return GroqProvider(settings.groq_model, key(settings.groq_api_key), timeout)
    if name == "ollama":
        return OllamaProvider(
            settings.ollama_model, None, timeout, base_url=settings.ollama_base_url
        )
    raise ValueError(f"Proveedor desconocido: {name!r}")
