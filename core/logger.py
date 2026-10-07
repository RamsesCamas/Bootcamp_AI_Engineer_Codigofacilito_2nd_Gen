"""Registro estructurado de llamadas al LLM en formato JSONL (una línea por intento).

Regla de oro: aquí nunca se escriben llaves, headers, ni el texto del prompt o de la
respuesta. Del prompt solo se guarda su tamaño en caracteres (`prompt_chars`).

Campos opcionales (Clase 2): `prompt_name`, `prompt_version`, `context_tokens` y
`context_truncated`. Solo se escriben cuando vienen; si no, la línea no los incluye.
"""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime
from pathlib import Path


class CallLogger:
    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)

    def log(
        self,
        *,
        provider: str,
        model: str | None,
        tokens_in: int,
        tokens_out: int,
        latency_ms: float,
        cost_usd: float,
        fallback: bool,
        success: bool,
        error_type: str | None,
        prompt_chars: int,
        ttft_ms: float | None = None,
        prompt_name: str | None = None,
        prompt_version: int | None = None,
        context_tokens: dict[str, int] | None = None,
        context_truncated: list[str] | None = None,
    ) -> dict:
        """Agrega un evento al archivo y lo devuelve.

        Cualquier campo opcional que no esté en la firma lanza `TypeError`: así nadie
        puede colar texto del prompt al log por accidente.
        """
        event = {
            "request_id": uuid.uuid4().hex[:8],
            "timestamp": datetime.now(UTC).isoformat(),
            "provider": provider,
            "model": model,
            "tokens_in": tokens_in,
            "tokens_out": tokens_out,
            "latency_ms": round(latency_ms, 1),
            "ttft_ms": ttft_ms,
            "cost_usd": cost_usd,
            "fallback": fallback,
            "success": success,
            "error_type": error_type,
            "prompt_chars": prompt_chars,
        }
        optional = {
            "prompt_name": prompt_name,
            "prompt_version": prompt_version,
            "context_tokens": context_tokens,
            "context_truncated": context_truncated,
        }
        event.update({key: value for key, value in optional.items() if value is not None})
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")
        return event
