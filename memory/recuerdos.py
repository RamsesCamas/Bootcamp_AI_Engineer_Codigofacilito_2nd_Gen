"""Memoria de largo plazo: lo que el agente sabe del usuario de una sesión a otra.

Cada recuerdo es una oración con su tipo y sus fechas, en var/memoria.json: un archivo que
puedes abrir y corregir. Los tipos vienen de CoALA: semántico (hechos del usuario y su
mundo), episódico (lo que pasó) y procedimental (cómo quiere que trabaje el agente).

Para buscar, el embedder del RAG convierte la consulta y los recuerdos en vectores. El
puntaje suma la similitud y un poco de recencia, como en Generative Agents. Con cientos de
recuerdos basta comparar contra todos; con millones irían a un índice como el de la Clase 6.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal

from knowledge.chunking import Chunk
from knowledge.embeddings import EmbeddingProvider
from knowledge.store import Hit

MEMORIA_PATH = Path("var") / "memoria.json"
DECAIMIENTO = 0.995  # por hora desde la última actualización, como en Generative Agents
PESO_RECENCIA = 0.1  # la similitud manda; la recencia desempata

Tipo = Literal["semantico", "episodico", "procedimental"]


@dataclass
class Recuerdo:
    id: str
    texto: str
    tipo: Tipo
    creado: str  # ISO 8601, a minutos
    actualizado: str
    sesion: str | None = None  # la sesión de la que salió


class Memoria:
    def __init__(
        self,
        embedder: EmbeddingProvider,
        path: Path = MEMORIA_PATH,
        reloj: Callable[[], datetime] = datetime.now,
    ) -> None:
        self.embedder = embedder
        self.path = Path(path)
        self.reloj = reloj
        datos = {"siguiente": 1, "recuerdos": []}
        if self.path.exists():
            datos = json.loads(self.path.read_text(encoding="utf-8"))
        self.siguiente: int = datos["siguiente"]  # los ids no se reusan aunque borres
        self.recuerdos = {d["id"]: Recuerdo(**d) for d in datos["recuerdos"]}

    def agregar(
        self, texto: str, tipo: Tipo, sesion: str | None = None, fecha: str | None = None
    ) -> Recuerdo:
        fecha = fecha or self._ahora()
        recuerdo = Recuerdo(f"r{self.siguiente}", texto, tipo, fecha, fecha, sesion)
        self.siguiente += 1
        self.recuerdos[recuerdo.id] = recuerdo
        self._guardar()
        return recuerdo

    def actualizar(self, recuerdo_id: str, texto: str) -> Recuerdo:
        recuerdo = self.recuerdos[recuerdo_id]
        recuerdo.texto, recuerdo.actualizado = texto, self._ahora()
        self._guardar()
        return recuerdo

    def borrar(self, recuerdo_id: str) -> Recuerdo:
        recuerdo = self.recuerdos.pop(recuerdo_id)
        self._guardar()
        return recuerdo

    def buscar(self, consulta: str, k: int = 3) -> list[tuple[Recuerdo, float]]:
        """Los k recuerdos de mayor puntaje, cada uno con su similitud coseno a la consulta.

        puntaje = similitud + PESO_RECENCIA × DECAIMIENTO ** (horas desde que se actualizó)
        """
        recuerdos = list(self.recuerdos.values())
        if not recuerdos:
            return []
        vectores = self.embedder.embed_documents([Chunk(r.id, r.texto, {}) for r in recuerdos])
        consulta_v = self.embedder.embed_query(consulta)
        ahora = self.reloj()
        puntuados = []
        for recuerdo, vector in zip(recuerdos, vectores, strict=True):
            similitud = sum(a * b for a, b in zip(consulta_v, vector, strict=True))
            horas = (ahora - datetime.fromisoformat(recuerdo.actualizado)).total_seconds() / 3600
            puntaje = similitud + PESO_RECENCIA * DECAIMIENTO ** max(horas, 0)
            puntuados.append((puntaje, recuerdo, similitud))
        puntuados.sort(key=lambda p: p[0], reverse=True)
        return [(recuerdo, similitud) for _, recuerdo, similitud in puntuados[:k]]

    def search(self, pregunta: str, k: int = 5, where: dict | None = None) -> list[Hit]:
        """La interfaz del índice de documentos: el RAG, las métricas y el router usan la
        memoria como cualquier otra fuente. `where` no aplica: la memoria no tiene filtros."""
        return [
            Hit(r.id, r.texto, {"fuente": f"memoria/{r.id}", "tipo": r.tipo}, 1 - similitud)
            for r, similitud in self.buscar(pregunta, k)
        ]

    def _ahora(self) -> str:
        return self.reloj().isoformat(timespec="minutes")

    def _guardar(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        recuerdos = [asdict(r) for r in self.recuerdos.values()]
        datos = {"siguiente": self.siguiente, "recuerdos": recuerdos}
        self.path.write_text(json.dumps(datos, ensure_ascii=False, indent=2), encoding="utf-8")
