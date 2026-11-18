"""Router de conocimiento: qué fuente consulta cada pregunta (Clase 9).

Tres fuentes con la misma interfaz `search`: la búsqueda híbrida en documentos, el grafo y
la memoria del usuario. Dos maneras de elegir, con la misma salida (`Ruta`):
- `por_reglas`: la regla del reto de la Clase 8 más una para la memoria. Gratis y
  predecible, pero solo ve palabras.
- `con_llm`: el LLM elige leyendo la descripción de cada fuente, como los selectores de
  LlamaIndex. Entiende paráfrasis, a cambio de una llamada por pregunta.
`Router` elige y busca con la fuente elegida. Sin LLM, o si el LLM falla, decide la regla.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, Field, ValidationInfo, model_validator

from contracts.structured_output import StructuredOutputError, generate_structured
from core.errors import AllProvidersFailedError
from prompting import PromptKit

if TYPE_CHECKING:
    from core.llm_client import LLMClient
    from knowledge.store import Hit

PROMPTS_DIR = Path(__file__).resolve().parents[1] / "prompts"
COMPANY = "Acme Operaciones"

Fuente = Literal["busqueda", "grafo", "memoria"]

# Lo único que el router con LLM sabe de cada fuente: escribirlas bien es el trabajo.
DESCRIPCIONES: dict[str, str] = {
    "busqueda": "documentos de Acme (políticas, runbooks y contratos) con búsqueda híbrida. "
    "Para preguntas que nombran lo que buscan: un proceso, un proveedor, una cláusula.",
    "grafo": "relaciones entre tickets, proveedores, contratos y runbooks. Para preguntas "
    "que nombran un ticket y piden un dato de otra entidad que no nombran, como el contrato "
    "o el runbook de quien lo atiende.",
    "memoria": "lo que el usuario contó en sesiones anteriores: su equipo, su horario, sus "
    "acuerdos, lo que hizo o dejó pendiente y cómo quiere las respuestas.",
}

_TICKET = re.compile(r"\bT-\d{4}\b")
_PROVEEDOR = re.compile(r"\b(proveedor|contrato)\b", re.IGNORECASE)
_PRIMERA_PERSONA = re.compile(r"\b(me|mi|mis|yo|conmigo)\b|recuérdame", re.IGNORECASE)


class Ruta(BaseModel):
    motivo: str = Field(max_length=300)
    fuente: Fuente

    @model_validator(mode="after")
    def _fuente_disponible(self, info: ValidationInfo) -> Ruta:
        fuentes = (info.context or {}).get("fuentes")
        if fuentes and self.fuente not in fuentes:
            raise ValueError(f"fuente debe ser una de {fuentes}")
        return self


def por_reglas(pregunta: str) -> Ruta:
    """Primera persona → memoria; ticket + proveedor o contrato → grafo; si no, búsqueda."""
    if _PRIMERA_PERSONA.search(pregunta):
        return Ruta(motivo="Habla en primera persona.", fuente="memoria")
    if _TICKET.search(pregunta) and _PROVEEDOR.search(pregunta):
        return Ruta(motivo="Nombra un ticket y pide su proveedor o contrato.", fuente="grafo")
    return Ruta(motivo="Ninguna regla aplica.", fuente="busqueda")


def con_llm(client: LLMClient, pregunta: str, fuentes: tuple[str, ...]) -> Ruta:
    """Una llamada: el LLM elige entre `fuentes` con sus descripciones."""
    prompt = PromptKit(PROMPTS_DIR).get("router")
    mensajes = prompt.render(
        company=COMPANY,
        fuentes="\n".join(f"- {fuente}: {DESCRIPCIONES[fuente]}" for fuente in fuentes),
        pregunta=pregunta,
    )
    return generate_structured(
        client,
        mensajes,
        Ruta,
        log_extra={"prompt_name": prompt.name, "prompt_version": prompt.version},
        validation_context={"fuentes": list(fuentes)},
        **prompt.model_params,
    )


class Router:
    """Elige la fuente y busca con ella. Tiene la interfaz `search` de las fuentes, así que
    el RAG (`answer`) lo usa como a cualquier retriever."""

    def __init__(self, fuentes: dict, client: LLMClient | None = None) -> None:
        self.fuentes = fuentes  # nombre → objeto con search(pregunta, k, where)
        self.client = client
        self.ruta: Ruta | None = None  # la última ruta elegida
        self.decidio = ""  # "llm" o "regla"
        self.respaldos = 0  # cuántas veces falló el LLM y decidió la regla

    def elegir(self, pregunta: str) -> Ruta:
        if self.client is not None:
            try:
                ruta = con_llm(self.client, pregunta, tuple(self.fuentes))
                self.decidio = "llm"
                return ruta
            except (AllProvidersFailedError, StructuredOutputError):
                self.respaldos += 1  # decide la regla
        self.decidio = "regla"
        ruta = por_reglas(pregunta)
        if ruta.fuente not in self.fuentes:
            return Ruta(motivo=f"{ruta.motivo} No hay {ruta.fuente}.", fuente="busqueda")
        return ruta

    def search(self, pregunta: str, k: int = 5, where: dict | None = None) -> list[Hit]:
        self.ruta = self.elegir(pregunta)
        return self.fuentes[self.ruta.fuente].search(pregunta, k=k, where=where)
