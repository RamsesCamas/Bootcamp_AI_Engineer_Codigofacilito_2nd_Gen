"""Cálculo de costo por llamada a partir de tokens de entrada y salida."""

from __future__ import annotations

import warnings

# Precios ilustrativos. Revisa la tabla vigente de cada proveedor antes de usarlos para decisiones.
# Formato: ID de modelo -> (USD por 1M tokens de entrada, USD por 1M tokens de salida).
PRICES_PER_1M: dict[str, tuple[float, float]] = {
    # Gemini (https://ai.google.dev/gemini-api/docs/pricing, nivel de pago, texto)
    "gemini-3.8-flash": (0.75, 3.75),
    "gemini-3.5-flash-lite": (0.30, 2.50),
    "gemini-2.5-flash": (0.30, 2.50),
    "gemini-2.5-flash-lite": (0.10, 0.40),
    # Groq (https://console.groq.com/docs/models)
    "openai/gpt-oss-20b": (0.075, 0.30),
    "openai/gpt-oss-120b": (0.15, 0.60),
}

_warned_models: set[str] = set()


def cost_usd(model: str, tokens_in: int, tokens_out: int, provider: str | None = None) -> float:
    """Devuelve el costo en USD de una llamada.

    Ollama corre en tu máquina, así que siempre cuesta 0.0. Si el modelo no está en la
    tabla devuelve 0.0 y avisa una sola vez por modelo.
    """
    if provider == "ollama":
        return 0.0
    prices = PRICES_PER_1M.get(model)
    if prices is None:
        if model not in _warned_models:
            _warned_models.add(model)
            warnings.warn(
                f"Modelo sin precio en core/pricing.py: {model!r}. Se registra costo 0.0.",
                stacklevel=2,
            )
        return 0.0
    price_in, price_out = prices
    return (tokens_in * price_in + tokens_out * price_out) / 1_000_000
