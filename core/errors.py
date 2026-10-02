"""Jerarquía de errores de proveedores.

La distinción entre errores transitorios y permanentes decide qué hace el cliente:
los transitorios se reintentan; los permanentes saltan directo al siguiente proveedor.
"""

from __future__ import annotations


class ProviderError(Exception):
    """Error base de cualquier proveedor de LLM."""

    def __init__(self, provider: str, message: str, status_code: int | None = None) -> None:
        super().__init__(message)
        self.provider = provider
        self.status_code = status_code

    def __str__(self) -> str:
        code = f" (HTTP {self.status_code})" if self.status_code is not None else ""
        return f"[{self.provider}]{code} {self.args[0]}"


class TransientProviderError(ProviderError):
    """Error que puede desaparecer si se reintenta: 429, 500, 502, 503, 504, timeouts y red."""


class PermanentProviderError(ProviderError):
    """Error que no se arregla reintentando: llave inválida, request mal formada, etc."""


class AllProvidersFailedError(Exception):
    """Todos los proveedores de la cadena fallaron."""

    def __init__(self, errors: list[ProviderError]) -> None:
        self.errors = errors
        detail = "; ".join(f"{type(e).__name__} {e}" for e in errors) or "sin proveedores"
        super().__init__(f"Todos los proveedores fallaron: {detail}")
