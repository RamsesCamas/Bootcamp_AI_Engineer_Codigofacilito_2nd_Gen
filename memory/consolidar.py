"""Consolidación: de una sesión a la memoria de largo plazo, en segundo plano (Clase 9).

Dos pasos, como en Mem0. El LLM extrae hechos de los turnos que aún no se consolidan; luego,
por cada hecho, ve los recuerdos más parecidos y decide: agregar, actualizar uno, borrar uno
o nada. Corre después de la sesión (`scripts/memoria.py consolidar`), fuera del camino de la
respuesta: el usuario no espera estas llamadas.
"""

from __future__ import annotations

from datetime import date
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, Field, ValidationInfo, model_validator

from contracts.structured_output import generate_structured
from memory.recuerdos import Memoria, Recuerdo, Tipo
from memory.session import COMPANY, PROMPTS_DIR, Sesion, Turno
from prompting import PromptKit

if TYPE_CHECKING:
    from core.llm_client import LLMClient

PARECIDOS = 5  # recuerdos que ve el LLM al decidir qué hacer con un hecho


class Hecho(BaseModel):
    texto: str = Field(min_length=5, max_length=200)
    tipo: Tipo


class Extraccion(BaseModel):
    hechos: list[Hecho] = Field(max_length=5)


class Decision(BaseModel):
    motivo: str = Field(max_length=200)
    operacion: Literal["agregar", "actualizar", "borrar", "nada"]
    id: str | None = None  # el recuerdo que se actualiza o se borra
    texto: str | None = None  # con actualizar: el texto que reemplaza al recuerdo

    @model_validator(mode="after")
    def _id_de_la_lista(self, info: ValidationInfo) -> Decision:
        ids = (info.context or {}).get("ids", [])
        if self.operacion in ("actualizar", "borrar") and self.id not in ids:
            raise ValueError(f"para {self.operacion}, id debe ser uno de {ids}")
        if self.operacion == "actualizar" and not self.texto:
            raise ValueError("actualizar necesita el texto final")
        return self


def extraer(client: LLMClient, resumen: str, turnos: list[Turno], hoy: date) -> list[Hecho]:
    prompt = PromptKit(PROMPTS_DIR).get("memoria_extraer")
    mensajes = prompt.render(
        company=COMPANY,
        hoy=hoy.isoformat(),
        resumen=resumen or "(ninguno)",
        turnos="\n".join(f"{turno.rol}: {turno.texto}" for turno in turnos),
    )
    log_extra = {"prompt_name": prompt.name, "prompt_version": prompt.version}
    extraccion = generate_structured(
        client, mensajes, Extraccion, log_extra=log_extra, **prompt.model_params
    )
    return extraccion.hechos


def decidir(client: LLMClient, hecho: Hecho, parecidos: list[Recuerdo]) -> Decision:
    """Sin recuerdos parecidos no hay nada que comparar: se agrega sin llamar al LLM."""
    if not parecidos:
        return Decision(motivo="La memoria está vacía.", operacion="agregar")
    prompt = PromptKit(PROMPTS_DIR).get("memoria_consolidar")
    mensajes = prompt.render(
        recuerdos="\n".join(f"{r.id} ({r.tipo}): {r.texto}" for r in parecidos),
        nuevo=f"({hecho.tipo}) {hecho.texto}",
    )
    return generate_structured(
        client,
        mensajes,
        Decision,
        log_extra={"prompt_name": prompt.name, "prompt_version": prompt.version},
        validation_context={"ids": [r.id for r in parecidos]},  # ids inventados → reparar
        **prompt.model_params,
    )


def aplicar(memoria: Memoria, hecho: Hecho, decision: Decision, sesion_id: str) -> str:
    """Ejecuta la decisión y la describe en una línea."""
    if decision.operacion == "agregar":
        recuerdo = memoria.agregar(hecho.texto, hecho.tipo, sesion=sesion_id)
    elif decision.operacion == "actualizar":
        recuerdo = memoria.actualizar(decision.id, decision.texto)
    elif decision.operacion == "borrar":
        recuerdo = memoria.borrar(decision.id)
    else:
        return f"nada: {hecho.texto} ({decision.motivo})"
    return f"{decision.operacion} {recuerdo.id}: {recuerdo.texto} ({decision.motivo})"


def consolidar(client: LLMClient, memoria: Memoria, sesion: Sesion) -> list[str]:
    """Pasa a la memoria los turnos de `sesion` que faltan. Quien llama guarda la sesión."""
    nuevos = sesion.turnos[sesion.consolidados :]
    if not nuevos:
        return []
    cambios = []
    for hecho in extraer(client, sesion.resumen, nuevos, memoria.reloj().date()):
        parecidos = [recuerdo for recuerdo, _ in memoria.buscar(hecho.texto, k=PARECIDOS)]
        cambios.append(aplicar(memoria, hecho, decidir(client, hecho, parecidos), sesion.id))
    sesion.consolidados = len(sesion.turnos)
    return cambios
