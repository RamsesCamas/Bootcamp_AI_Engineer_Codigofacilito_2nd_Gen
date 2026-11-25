"""Checkpoint de una ejecución del agente ReAct, escrito a mano (reto de la Clase 9).

Después de cada paso se guarda en var/ejecuciones/ID.json lo necesario para seguir sin
repetir nada: la pregunta, el historial, el request_id y la traza. Si la ejecución se cae
(sin cuota, sin red, Ctrl+C), `--reanudar ID` la retoma desde el último paso guardado. En la
Clase 11 lo comparamos con el checkpointer de LangGraph.
"""

from __future__ import annotations

import json
import os
import re
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from agents.react_agent import AgentStep
from agents.trace import TraceStep

if TYPE_CHECKING:
    from agents.trace import AgentResult

EJECUCIONES_PATH = Path("var") / "ejecuciones"
_ID = re.compile(r"^[\w-]{1,40}$")  # el id es un nombre de archivo, como en las sesiones


def paso_a_dict(step: TraceStep) -> dict:
    """Un paso de la traza en JSON: lo que decidió el modelo, lo que observó y lo que costó."""
    return {
        "paso": step.paso.model_dump(),
        "observacion": step.observacion,
        "tokens": step.tokens,
        "cost_usd": step.cost_usd,
        "latency_ms": step.latency_ms,
    }


def dict_a_paso(data: dict) -> TraceStep:
    return TraceStep(
        paso=AgentStep.model_validate(data["paso"]),
        observacion=data["observacion"],
        tokens=data["tokens"],
        cost_usd=data["cost_usd"],
        latency_ms=data["latency_ms"],
    )


@dataclass
class Ejecucion:
    id: str
    pregunta: str
    historial: str = ""
    # El request_id de la Clase 5. Al reanudar, una escritura que se repite usa la misma
    # llave de idempotencia y no se duplica.
    request_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    pasos: list[dict] = field(default_factory=list)
    motivo: str | None = None  # None: se cortó antes de terminar
    respuesta: str | None = None

    def __post_init__(self) -> None:
        if not _ID.match(self.id):
            raise ValueError(f"id de ejecución inválido: {self.id!r}; usa letras, números, - o _.")

    @property
    def path(self) -> Path:
        return EJECUCIONES_PATH / f"{self.id}.json"

    @property
    def reanudable(self) -> bool:
        """Se reanuda la que se cortó o terminó con error; la que respondió, no."""
        return self.motivo in (None, "error")

    def traza(self) -> list[TraceStep]:
        return [dict_a_paso(paso) for paso in self.pasos]

    def registrar(self, step: TraceStep) -> None:
        """Se llama al terminar cada paso: lo agrega y guarda el archivo."""
        self.pasos.append(paso_a_dict(step))
        self.guardar()

    def terminar(self, resultado: AgentResult) -> None:
        self.motivo, self.respuesta = resultado.motivo, resultado.respuesta
        self.guardar()

    def guardar(self) -> None:
        """Escritura atómica, como en tools/tickets.py: un corte no deja un JSON a medias."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(asdict(self), ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(tmp, self.path)

    @classmethod
    def crear(cls, ejecucion_id: str, pregunta: str, historial: str = "") -> Ejecucion:
        """Una ejecución nueva, guardada antes del primer paso."""
        ejecucion = cls(ejecucion_id, pregunta, historial)
        if ejecucion.path.exists():
            raise ValueError(
                f"ya existe la ejecución {ejecucion_id}; usa --reanudar {ejecucion_id} o elige "
                "otro id."
            )
        ejecucion.guardar()
        return ejecucion

    @classmethod
    def cargar(cls, ejecucion_id: str) -> Ejecucion:
        path = cls(ejecucion_id, pregunta="").path  # valida el id antes de armar la ruta
        if not path.exists():
            raise LookupError(f"no hay una ejecución {ejecucion_id} en {EJECUCIONES_PATH}.")
        return cls(**json.loads(path.read_text(encoding="utf-8")))
