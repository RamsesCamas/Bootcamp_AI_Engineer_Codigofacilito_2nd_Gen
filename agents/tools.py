"""Herramientas con política: quién puede usar cada una, cuántas veces y con qué tiempo.

`ToolRunner` es un filtro entre el modelo y los handlers. Antes de ejecutar revisa, en
este orden: permiso del rol, llamadas restantes y argumentos. Después ejecuta con timeout
y reintenta solo los errores transitorios de herramientas idempotentes. Nunca lanza: todo
error regresa como `tool_error`, para que el modelo pueda explicarlo o corregirse.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from collections import Counter
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, ValidationError

from contracts.tool_schemas import tool_error

if TYPE_CHECKING:
    from contracts.tool_schemas import ToolSpec
    from core.llm_client import ToolCall

Permiso = Literal["lectura", "escritura", "peligrosa"]
Rol = Literal["lectura", "operador"]

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ToolPolicy:
    permiso: Permiso
    timeout_s: float = 5
    idempotente: bool = True
    max_llamadas: int = 3


POLITICAS: dict[str, ToolPolicy] = {
    "buscar_ticket": ToolPolicy("lectura", timeout_s=2),
    "actualizar_estado": ToolPolicy("escritura"),
    "consultar_proveedor": ToolPolicy("lectura", timeout_s=2),
    "listar_tickets_por_proveedor": ToolPolicy("lectura", timeout_s=2),
    # Escritura idempotente gracias a la idempotency_key: el reintento no duplica.
    "crear_ticket": ToolPolicy("escritura", idempotente=True),
    # Clase 7: embedding de la consulta y, con rerank, una llamada al LLM.
    "buscar_documentos": ToolPolicy("lectura", timeout_s=30),
    # Clase 8: el grafo está en memoria; no llama a ninguna API.
    "consultar_grafo": ToolPolicy("lectura", timeout_s=2),
    # Clase 9: embedding de la consulta (y de los recuerdos que no estén en la caché).
    "consultar_memoria": ToolPolicy("lectura", timeout_s=30),
    # Práctica guiada de la Clase 9: el mismo texto no se duplica, así que reintentar es seguro.
    "guardar_recuerdo": ToolPolicy("escritura", idempotente=True),
}
ROLES: dict[Rol, set[Permiso]] = {"lectura": {"lectura"}, "operador": {"lectura", "escritura"}}

# Herramienta sin política explícita: se trata como peligrosa y se niega (default seguro).
# "peligrosa" no está en ningún rol: la aprobación humana llega en la Clase 11.
SIN_POLITICA = ToolPolicy("peligrosa")
BACKOFF_BASE_S = 0.5  # Esperas entre reintentos: 0.5 s, 1 s, ...


class TransientToolError(Exception):
    """Lo lanza un handler cuando la falla es pasajera (red, servicio saturado)."""


def _invalid_args(error: ValidationError) -> str:
    parts = []
    for err in error.errors():
        where = ".".join(str(p) for p in err["loc"])
        parts.append(f"{where}: {err['msg']}" if where else err["msg"])
    return "Argumentos inválidos: " + "; ".join(parts)


def _normalize(result: dict) -> dict:
    """Convierte el error heredado `{"error": "texto"}` en un `tool_error` de negocio."""
    if isinstance(result.get("error"), str):
        return tool_error("negocio", result["error"])
    return result


class ToolRunner:
    def __init__(
        self,
        tools: list[ToolSpec],
        *,
        rol: Rol = "operador",
        politicas: dict[str, ToolPolicy] = POLITICAS,
        sleep: Callable[[float], None] = time.sleep,
        retries: int = 2,
        log_path: Path | str | None = None,
        request_id: str | None = None,
    ) -> None:
        self.tools = {tool.name: tool for tool in tools}
        # Identifica la petición del usuario. Lo da quien crea el runner (la CLI, el
        # server MCP); si no, uno nuevo por runner. Nunca sale del modelo.
        self.request_id = request_id or uuid.uuid4().hex
        self.rol = rol
        self.politicas = politicas
        self.sleep = sleep
        self.retries = retries
        self.log_path = Path(log_path) if log_path else None
        self.eventos: list[dict] = []  # Lo que se registró, en orden (útil en tests).
        self._llamadas: Counter[str] = Counter()

    def politica(self, name: str) -> ToolPolicy:
        return self.politicas.get(name, SIN_POLITICA)

    def permitida(self, name: str) -> bool:
        return self.politica(name).permiso in ROLES[self.rol]

    def execute(self, call: ToolCall) -> dict:
        """Mismo contrato que `execute_tool`: regresa el mensaje `tool` y nunca lanza."""
        result = self._run(call.name, call.arguments or "{}")
        content = json.dumps(result, ensure_ascii=False)
        return {"role": "tool", "tool_call_id": call.id, "content": content}

    def call(self, name: str, arguments: dict) -> dict:
        """Para el server MCP: regresa el contenido, no el mensaje `tool`."""
        return self._run(name, arguments)

    def _run(self, name: str, arguments: str | dict) -> dict:
        spec = self.tools.get(name)
        if spec is None:
            disponibles = ", ".join(sorted(self.tools))
            result = tool_error(
                "validacion", f"La herramienta {name} no existe", sugerencia=f"Usa: {disponibles}"
            )
            return self._record(name, result, intentos=0)
        politica = self.politica(name)

        # 1. ¿Permitida para el rol?
        if not self.permitida(name):
            result = tool_error(
                "permiso",
                f"El rol {self.rol} no puede usar {name} (permiso: {politica.permiso})",
                sugerencia="Explica al usuario que no tienes permiso para esa acción.",
            )
            return self._record(name, result, intentos=0)

        # 2. ¿Quedan llamadas? El tope es por herramienta y por instancia del runner.
        if self._llamadas[name] >= politica.max_llamadas:
            result = tool_error(
                "limite",
                f"{name} ya se usó {politica.max_llamadas} veces en esta ejecución",
                sugerencia="Responde con lo que ya tienes.",
            )
            return self._record(name, result, intentos=0)
        self._llamadas[name] += 1

        # 3. ¿Argumentos válidos?
        try:
            args = (
                spec.args_model.model_validate(arguments)
                if isinstance(arguments, dict)
                else spec.args_model.model_validate_json(arguments)
            )
        except ValidationError as e:
            result = tool_error("validacion", _invalid_args(e), reintentable=True)
            return self._record(name, result, intentos=0)

        # 4 y 5. Ejecutar con timeout; reintentar solo transitorios idempotentes.
        intentos = 0
        while True:
            intentos += 1
            try:
                result = _normalize(self._with_timeout(spec, args, politica.timeout_s))
                return self._record(name, result, intentos)
            except (FutureTimeoutError, TransientToolError) as e:
                motivo = (
                    f"{name} no respondió en {politica.timeout_s} s"
                    if isinstance(e, FutureTimeoutError)
                    else f"{name} falló de forma transitoria: {e}"
                )
                if not politica.idempotente:
                    # Repetir una escritura no idempotente podría duplicarla.
                    result = tool_error("transitorio", motivo, reintentable=False)
                    return self._record(name, result, intentos)
                if intentos > self.retries:
                    result = tool_error("transitorio", motivo, reintentable=True)
                    return self._record(name, result, intentos)
                self.sleep(BACKOFF_BASE_S * 2 ** (intentos - 1))
            except Exception:
                logger.warning("La herramienta %s falló", name, exc_info=True)
                result = tool_error("permanente", f"La herramienta {name} falló")
                return self._record(name, result, intentos)

    def _with_timeout(self, spec: ToolSpec, args: BaseModel, timeout_s: float) -> dict:
        """Ejecuta el handler en un hilo y espera máximo `timeout_s`.

        Python no puede matar un hilo: si se vence el tiempo, el handler sigue corriendo
        en segundo plano hasta terminar; solo dejamos de esperarlo. Por eso el timeout no
        sustituye a que el handler tenga sus propios límites (por ejemplo, en httpx).
        """
        kwargs = {"request_id": self.request_id} if spec.needs_request_id else {}
        pool = ThreadPoolExecutor(max_workers=1)
        try:
            return pool.submit(spec.handler, args, **kwargs).result(timeout=timeout_s)
        finally:
            pool.shutdown(wait=False)

    def _record(self, name: str, result: dict, intentos: int) -> dict:
        """Registra nombre, permiso, resultado e intentos. Nunca los argumentos."""
        error = result.get("error")
        evento = {
            "tool_name": name,
            "tool_permiso": self.politica(name).permiso,
            "tool_resultado": error["tipo"] if isinstance(error, dict) else "ok",
            "tool_intentos": intentos,
        }
        self.eventos.append(evento)
        logger.info("herramienta %s", evento)
        if self.log_path:
            line = {"timestamp": datetime.now(UTC).isoformat(), **evento}
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with self.log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(line, ensure_ascii=False) + "\n")
        return result
