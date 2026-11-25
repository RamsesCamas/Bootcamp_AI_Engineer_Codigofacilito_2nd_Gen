"""Una caída del proveedor a voluntad, para practicar la reanudación (reto de la Clase 9).

Esperar a que Gemini regrese un 429 en vivo no es un plan. `ClienteConFalla` deja pasar las
primeras llamadas y, desde la número `falla_en`, falla como si todos los proveedores
estuvieran caídos: lo mismo que ve el agente cuando se acaba la cuota.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from core.errors import AllProvidersFailedError, PermanentProviderError
from core.llm_client import LLMClient

if TYPE_CHECKING:
    from core.llm_client import LLMResponse


class ClienteConFalla(LLMClient):
    def __init__(self, base: LLMClient, falla_en: int) -> None:
        super().__init__(base.providers, base.logger, base.max_retries, base._sleep)
        self.falla_en = falla_en
        self.llamadas = 0

    def generate(self, messages: list[dict], **kwargs) -> LLMResponse:
        self.llamadas += 1
        if self.llamadas >= self.falla_en:
            causa = PermanentProviderError(
                "simulada", f"caída simulada en la llamada {self.llamadas}"
            )
            raise AllProvidersFailedError([causa])
        return super().generate(messages, **kwargs)
