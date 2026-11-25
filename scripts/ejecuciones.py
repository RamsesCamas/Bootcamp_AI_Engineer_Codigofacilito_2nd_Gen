"""Las ejecuciones del agente como grafo, desde la terminal (Clase 11).

    uv run scripts/ejecuciones.py listar
    uv run scripts/ejecuciones.py historial demo
    uv run scripts/ejecuciones.py grafo

Los checkpoints viven en var/checkpoints.db. `listar` muestra cada ejecución con sus pasos y
si terminó; `historial` recorre sus checkpoints del más viejo al más nuevo, con el nodo que
corrió en cada uno; `grafo` imprime el grafo en Mermaid. Ninguno llama al LLM.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import TYPE_CHECKING

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))  # Para importar los paquetes del repo al correr como script.

from agents.grafo_react import (  # noqa: E402
    CHECKPOINTS_PATH,
    AgenteGrafo,
    abrir_checkpointer,
    dibujar,
)

if TYPE_CHECKING:
    from langgraph.checkpoint.sqlite import SqliteSaver


def config(ejecucion: str) -> dict:
    return {"configurable": {"thread_id": ejecucion}}


def resumen(valores: dict, siguiente: tuple[str, ...]) -> str:
    pasos = len(valores.get("pasos", []))
    if siguiente:
        return f"{pasos} pasos · pendiente: {', '.join(siguiente)}"
    return f"{pasos} pasos · terminó: {valores.get('motivo')}"


def listar(agente: AgenteGrafo, saver: SqliteSaver) -> list[str]:
    """Una línea por ejecución. `saver.list(None)` regresa los checkpoints de todos los hilos."""
    ids = dict.fromkeys(t.config["configurable"]["thread_id"] for t in saver.list(None))
    lineas = []
    for ejecucion in ids:
        foto = agente.grafo.get_state(config(ejecucion))
        lineas.append(f"{ejecucion}  {resumen(foto.values, foto.next)}")
    return lineas


def historial(agente: AgenteGrafo, ejecucion: str) -> list[str]:
    """Un renglón por checkpoint. El nodo que corrió es el `next` del checkpoint anterior."""
    fotos = list(agente.grafo.get_state_history(config(ejecucion)))
    if not fotos:
        return []
    lineas = [f"{'paso':>4}  {'fuente':<6}  {'corrió':<10} {'sigue':<10} pasos"]
    anterior: tuple[str, ...] = ()
    for foto in reversed(fotos):  # LangGraph los da del más nuevo al más viejo
        corrio = ", ".join(anterior) or "-"
        sigue = ", ".join(foto.next) or "FIN"
        pasos = len(foto.values.get("pasos", []))
        paso, fuente = foto.metadata["step"], foto.metadata["source"]
        lineas.append(f"{paso:>4}  {fuente:<6}  {corrio:<10} {sigue:<10} {pasos}")
        anterior = foto.next
    return lineas


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Ejecuciones del agente como grafo.")
    parser.add_argument("--db", type=Path, default=CHECKPOINTS_PATH, help=argparse.SUPPRESS)
    sub = parser.add_subparsers(dest="comando", required=True)
    sub.add_parser("listar", help="Cada ejecución, con sus pasos y si terminó.")
    p = sub.add_parser("historial", help="Los checkpoints de una ejecución.")
    p.add_argument("id")
    sub.add_parser("grafo", help="El grafo en Mermaid.")
    args = parser.parse_args(argv)

    if args.comando == "grafo":
        print(dibujar())
        return 0
    if not args.db.exists():
        print(
            f"Error: no hay checkpoints en {args.db}; corre main.py --agent grafo.", file=sys.stderr
        )
        return 1
    with abrir_checkpointer(args.db) as saver:
        # Con el nodo de la práctica guiada: lee las ejecuciones con y sin verificar.
        agente = AgenteGrafo(None, [], checkpointer=saver, verificar=True)
        if args.comando == "listar":
            print("\n".join(listar(agente, saver)) or "No hay ejecuciones.")
            return 0
        lineas = historial(agente, args.id)
    if not lineas:
        print(f"Error: no hay una ejecución {args.id} en {args.db}.", file=sys.stderr)
        return 1
    print("\n".join(lineas))
    return 0


if __name__ == "__main__":
    sys.exit(main())
