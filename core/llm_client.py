"""Cliente LLM con reintentos, backoff exponencial y fallback entre proveedores."""

from __future__ import annotations

import random
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

from core.errors import AllProvidersFailedError, ProviderError, TransientProviderError

if TYPE_CHECKING:
    from core.config import Settings
    from core.logger import CallLogger
    from core.providers import Provider


@dataclass
class LLMResponse:
    text: str
    provider: str
    model: str
    tokens_in: int
    tokens_out: int
    latency_ms: float
    cost_usd: float


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

    def generate(self, messages: list[dict], **params) -> LLMResponse:
        """Pide una respuesta recorriendo los proveedores en orden.

        - `TransientProviderError`: reintenta con backoff 1 s -> 2 s -> 4 s + jitter.
        - `PermanentProviderError`: no reintenta; pasa al siguiente proveedor.
        - Si todos fallan: `AllProvidersFailedError`.
        """
        prompt_chars = sum(len(str(m.get("content", ""))) for m in messages)
        errors: list[ProviderError] = []

        for index, provider in enumerate(self.providers):
            is_fallback = index > 0
            for attempt in range(self.max_retries + 1):
                start = time.perf_counter()
                try:
                    response = provider.generate(messages, **params)
                except ProviderError as e:
                    self._log_failure(provider, e, start, is_fallback, prompt_chars)
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
        )
