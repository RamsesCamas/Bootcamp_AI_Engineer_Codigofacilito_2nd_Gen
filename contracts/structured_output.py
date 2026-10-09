"""Salida estructurada: pedir JSON, validarlo con Pydantic y reparar si no cumple."""

from __future__ import annotations

import copy
import json
import re
from typing import TYPE_CHECKING, TypeVar

from pydantic import BaseModel, ValidationError

if TYPE_CHECKING:
    from core.llm_client import LLMClient

T = TypeVar("T", bound=BaseModel)

# Palabras clave que se quitan de la copia del schema que se envía al proveedor:
# - maxLength, minLength y pattern: la página de structured output de Gemini no los lista
#   y la de Groq tampoco; los aplica Pydantic al validar localmente.
# - title y default: ruido para el modelo; `default` además choca con el modo estricto.
# La validación local siempre usa el modelo completo, así que no se pierde ninguna regla.
UNSUPPORTED_KEYWORDS = frozenset({"maxLength", "minLength", "pattern", "title", "default"})

_FENCE = re.compile(r"^```(?:json)?\s*(.*?)\s*```$", re.DOTALL)


class StructuredOutputError(RuntimeError):
    """El modelo no produjo una salida válida después de todas las reparaciones."""

    def __init__(self, schema_name: str, attempts: int, last_error: str) -> None:
        super().__init__(
            f"No se obtuvo un {schema_name} válido después de {attempts} intento(s). "
            f"Último error: {last_error}"
        )
        self.attempts = attempts
        self.last_error = last_error


def sanitize_schema(schema: dict, *, strict: bool = True) -> dict:
    """Devuelve una copia del JSON Schema que aceptan los proveedores.

    Quita `UNSUPPORTED_KEYWORDS` y resuelve `$ref`/`$defs` en línea. Con `strict=True`
    además pone `additionalProperties: false` y marca todas las propiedades como
    `required`, que es lo que exige el modo estricto de Groq.
    """
    schema = copy.deepcopy(schema)
    defs = schema.pop("$defs", {})

    def clean(node):
        if isinstance(node, list):
            return [clean(item) for item in node]
        if not isinstance(node, dict):
            return node
        if "$ref" in node:
            name = node["$ref"].rsplit("/", 1)[-1]
            return clean(copy.deepcopy(defs[name]))
        result = {k: clean(v) for k, v in node.items() if k not in UNSUPPORTED_KEYWORDS}
        if "properties" in node:
            # `properties` es un mapa de nombres a schemas: sus llaves no son palabras clave.
            result["properties"] = {k: clean(v) for k, v in node["properties"].items()}
            if strict:
                result["required"] = list(node["properties"])
                result["additionalProperties"] = False
        return result

    return clean(schema)


def schema_instruction(schema: dict) -> str:
    return (
        "Responde solo con un objeto JSON válido que cumpla este JSON Schema, sin texto "
        "adicional ni bloques de código:\n" + json.dumps(schema, ensure_ascii=False)
    )


def repair_message(error: ValidationError) -> str:
    """Explica los errores de validación en un mensaje que el modelo pueda corregir."""
    lines = []
    for err in error.errors():
        where = ".".join(str(part) for part in err["loc"]) or "respuesta"
        message = err["msg"].removeprefix("Value error, ")
        lines.append(f"- {where}: {message}")
    return (
        "Tu respuesta anterior no cumple el formato pedido. Errores:\n"
        + "\n".join(lines)
        + "\nResponde de nuevo solo con el JSON corregido."
    )


def _with_instruction(messages: list[dict], instruction: str) -> list[dict]:
    messages = [dict(m) for m in messages]
    if messages and messages[0]["role"] == "system":
        messages[0]["content"] = f"{messages[0]['content']}\n\n{instruction}"
    else:
        messages.insert(0, {"role": "system", "content": instruction})
    return messages


def _strip_fences(text: str) -> str:
    text = text.strip()
    match = _FENCE.match(text)
    return match.group(1) if match else text


def generate_structured(
    client: LLMClient,
    messages: list[dict],
    schema: type[T],
    max_repairs: int = 2,
    *,
    log_extra: dict | None = None,
    **params,
) -> T:
    """Pide al modelo un `schema` y lo devuelve validado, reparando hasta `max_repairs` veces.

    La estrategia depende del proveedor que atiende la request (el primero de la cadena):
    1. `json_schema` estricto si `supports_json_schema`.
    2. `json_object` más el schema en el system si `supports_json_object`.
    3. Solo el schema en el system.
    Un error del proveedor (por ejemplo, `PermanentProviderError`) no se repara: se propaga.
    """
    provider = client.providers[0]
    sent_schema = sanitize_schema(schema.model_json_schema())
    history = [dict(m) for m in messages]

    if getattr(provider, "supports_json_schema", False):
        response_format = {
            "type": "json_schema",
            "json_schema": {"name": schema.__name__, "schema": sent_schema, "strict": True},
        }
    elif getattr(provider, "supports_json_object", False):
        response_format = {"type": "json_object"}
        history = _with_instruction(history, schema_instruction(sent_schema))
    else:
        response_format = None
        history = _with_instruction(history, schema_instruction(sent_schema))

    last_error = ""
    for attempt in range(max_repairs + 1):
        extra = {**(log_extra or {}), "structured_schema": schema.__name__}
        extra["repair_attempt"] = attempt
        response = client.generate(
            history, log_extra=extra, response_format=response_format, **params
        )
        try:
            return schema.model_validate_json(_strip_fences(response.text))
        except ValidationError as e:
            last_error = repair_message(e)
            history.append({"role": "assistant", "content": response.text})
            history.append({"role": "user", "content": last_error})

    raise StructuredOutputError(schema.__name__, max_repairs + 1, last_error)
