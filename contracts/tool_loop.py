"""Ciclo mínimo de herramientas: el modelo pide, ejecutamos, devolvemos el resultado."""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from typing import TYPE_CHECKING

from pydantic import ValidationError

if TYPE_CHECKING:
    from contracts.tool_schemas import ToolSpec
    from core.llm_client import LLMClient, ToolCall

MAX_ITERS = 3

logger = logging.getLogger(__name__)

# Observador opcional del ciclo: recibe ("call", llamada, None) antes de ejecutar y
# ("result", llamada, mensaje_tool) después. Sirve para que la CLI imprima sin que
# contracts/ imprima nada.
ToolEventHandler = Callable[[str, "ToolCall", "dict | None"], None]


class MaxIterationsError(RuntimeError):
    """El modelo siguió pidiendo herramientas después de `max_iters` vueltas."""

    def __init__(self, max_iters: int) -> None:
        super().__init__(
            f"El modelo siguió pidiendo herramientas después de {max_iters} iteraciones."
        )
        self.max_iters = max_iters


def _invalid_args(error: ValidationError) -> str:
    parts = []
    for err in error.errors():
        where = ".".join(str(p) for p in err["loc"])
        parts.append(f"{where}: {err['msg']}" if where else err["msg"])
    return "; ".join(parts)


def execute_tool(call: ToolCall, tools: dict[str, ToolSpec]) -> dict:
    """Ejecuta una herramienta y regresa el mensaje `tool`. Nunca lanza.

    Los errores también se le devuelven al modelo, en `{"error": ...}`, para que pueda
    explicarlos o corregir sus argumentos. Los detalles internos se quedan en el log.
    """

    def tool_message(payload: dict) -> dict:
        content = json.dumps(payload, ensure_ascii=False)
        return {"role": "tool", "tool_call_id": call.id, "content": content}

    spec = tools.get(call.name)
    if spec is None:
        return tool_message({"error": f"La herramienta {call.name} no existe"})
    try:
        args = spec.args_model.model_validate_json(call.arguments or "{}")
    except ValidationError as e:
        return tool_message({"error": f"Argumentos inválidos: {_invalid_args(e)}"})
    try:
        return tool_message(spec.handler(args))
    except Exception:
        logger.warning("La herramienta %s falló", call.name, exc_info=True)
        return tool_message({"error": f"La herramienta {call.name} falló"})


def run_tool_loop(
    client: LLMClient,
    messages: list[dict],
    tools: list[ToolSpec],
    *,
    max_iters: int = MAX_ITERS,
    log_extra: dict | None = None,
    on_event: ToolEventHandler | None = None,
    **params,
) -> str:
    """Llama al modelo hasta que responda sin pedir herramientas, máximo `max_iters` veces.

    No modifica `messages`: trabaja sobre una copia. `params` (por ejemplo `temperature`)
    se pasan en cada llamada.
    """
    history = list(messages)
    by_name = {tool.name: tool for tool in tools}
    definitions = [tool.to_openai() for tool in tools]

    for iteration in range(max_iters):
        extra = {**(log_extra or {}), "tool_iteration": iteration}
        resp = client.generate(history, log_extra=extra, tools=definitions, **params)
        if not resp.tool_calls:
            return resp.text
        history.append(resp.as_message())
        for call in resp.tool_calls:
            if on_event:
                on_event("call", call, None)
            result = execute_tool(call, by_name)
            if on_event:
                on_event("result", call, result)
            history.append(result)

    raise MaxIterationsError(max_iters)
