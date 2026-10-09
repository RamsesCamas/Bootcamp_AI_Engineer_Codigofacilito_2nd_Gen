"""Cliente LLM con reintentos, backoff exponencial y fallback entre proveedores."""

from __future__ import annotations

import random
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from core.errors import AllProvidersFailedError, ProviderError, TransientProviderError

if TYPE_CHECKING:
    from core.config import Settings
    from core.logger import CallLogger
    from core.providers import Provider


@dataclass(frozen=True)
class ToolCall:
    """Una herramienta que pidió el modelo. `arguments` es el string JSON tal como llega;
    parsearlo y validarlo le toca a `contracts.tool_loop.execute_tool`."""

    id: str
    name: str
    arguments: str
    # Datos opacos del proveedor que hay que reenviar tal cual en el siguiente turno.
    # Gemini 3 pone aquí su `thought_signature`; sin ella responde 400.
    extra_content: dict | None = field(default=None, compare=False, repr=False)

    def to_openai(self) -> dict:
        call = {
            "id": self.id,
            "type": "function",
            "function": {"name": self.name, "arguments": self.arguments},
        }
        if self.extra_content is not None:
            call["extra_content"] = self.extra_content
        return call


@dataclass
class LLMResponse:
    text: str
    provider: str
    model: str
    tokens_in: int
    tokens_out: int
    latency_ms: float
    cost_usd: float
    tool_calls: list[ToolCall] = field(default_factory=list)
    finish_reason: str | None = None

    def as_message(self) -> dict:
        """Mensaje `assistant` listo para agregarse al historial (formato OpenAI)."""
        message: dict = {"role": "assistant", "content": self.text or None}
        if self.tool_calls:
            message["tool_calls"] = [call.to_openai() for call in self.tool_calls]
        return message


class LLMClient:
    def __init__(
        self,
        providers: list[Provider],
        logger: CallLogger,
        max_retries: int = 3,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if not providers:
            raise ValueError("LLMClient necesita al menos un proveedor.")
        self.providers = providers
        self.logger = logger
        self.max_retries = max_retries
        self._sleep = sleep

    @classmethod
    def from_settings(cls, settings: Settings) -> LLMClient:
        """Arma la cadena primario + fallbacks definida en la configuración."""
        from core.logger import CallLogger
        from core.providers import build_provider

        names = [settings.primary, *settings.fallbacks]
        providers = [build_provider(name, settings) for name in names]
        return cls(providers, CallLogger(settings.log_path), settings.max_retries)

    def generate(
        self,
        messages: list[dict],
        *,
        log_extra: dict | None = None,
        tools: list[dict] | None = None,
        tool_choice: str | dict | None = None,
        response_format: dict | None = None,
        **params,
    ) -> LLMResponse:
        """Pide una respuesta recorriendo los proveedores en orden.

        `log_extra` (por ejemplo `prompt_name` y `prompt_version`) se agrega tal cual a
        cada evento del log de esta request, en todos los intentos.

        `tools`, `tool_choice` y `response_format` se mandan al proveedor solo si no son
        `None`. Si la respuesta pide herramientas, sus nombres quedan en el log (`tool_calls`).

        - `TransientProviderError`: reintenta con backoff 1 s -> 2 s -> 4 s + jitter.
        - `PermanentProviderError`: no reintenta; pasa al siguiente proveedor.
        - Si todos fallan: `AllProvidersFailedError`.
        """
        prompt_chars = sum(len(str(m.get("content") or "")) for m in messages)
        extra = log_extra or {}
        optional = {"tools": tools, "tool_choice": tool_choice, "response_format": response_format}
        params.update({key: value for key, value in optional.items() if value is not None})
        errors: list[ProviderError] = []

        for index, provider in enumerate(self.providers):
            is_fallback = index > 0
            for attempt in range(self.max_retries + 1):
                start = time.perf_counter()
                try:
                    response = provider.generate(messages, **params)
                except ProviderError as e:
                    self._log_failure(provider, e, start, is_fallback, prompt_chars, extra)
                    if isinstance(e, TransientProviderError) and attempt < self.max_retries:
                        self._sleep(2**attempt + random.uniform(0, 0.25))
                        continue
                    errors.append(e)
                    break  # Permanente o reintentos agotados: siguiente proveedor.
                self.logger.log(
                    provider=response.provider,
                    model=response.model,
                    tokens_in=response.tokens_in,
                    tokens_out=response.tokens_out,
                    latency_ms=response.latency_ms,
                    cost_usd=response.cost_usd,
                    fallback=is_fallback,
                    success=True,
                    error_type=None,
                    prompt_chars=prompt_chars,
                    # ttft_ms (tiempo al primer token) requiere streaming; se mide en la
                    # Clase 16. Por ahora queda en null.
                    ttft_ms=None,
                    **{"tool_calls": [c.name for c in response.tool_calls] or None, **extra},
                )
                return response

        raise AllProvidersFailedError(errors)

    def _log_failure(
        self,
        provider: Provider,
        error: ProviderError,
        start: float,
        is_fallback: bool,
        prompt_chars: int,
        extra: dict,
    ) -> None:
        self.logger.log(
            provider=provider.name,
            model=getattr(provider, "model", None),
            tokens_in=0,
            tokens_out=0,
            latency_ms=(time.perf_counter() - start) * 1000,
            cost_usd=0.0,
            fallback=is_fallback,
            success=False,
            error_type=type(error).__name__,
            prompt_chars=prompt_chars,
            ttft_ms=None,
            **extra,
        )
