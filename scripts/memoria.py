"""La memoria de largo plazo desde la terminal (Clase 9).

    uv run scripts/memoria.py importar data/memorias_demo.jsonl
    uv run scripts/memoria.py listar
    uv run scripts/memoria.py buscar "¿A quién escalé el T-1043?"
    uv run scripts/memoria.py borrar r4

Los recuerdos viven en var/memoria.json. `buscar` usa el embedder del RAG (Gemini con la
caché de .cache/, o `--embedder fake` sin red).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))  # Para importar los paquetes del repo al correr como script.

from knowledge.embeddings import make_embedder  # noqa: E402
from memory.recuerdos import MEMORIA_PATH, Memoria, Recuerdo  # noqa: E402


def linea(recuerdo: Recuerdo) -> str:
    return f"{recuerdo.id} · {recuerdo.tipo} · {recuerdo.actualizado[:10]}  {recuerdo.texto}"


def importar(memoria: Memoria, archivo: Path) -> str:
    """Agrega los recuerdos del JSONL que no estén ya (mismo texto): se puede repetir."""
    existentes = {r.texto for r in memoria.recuerdos.values()}
    nuevos = 0
    for renglon in archivo.read_text(encoding="utf-8").splitlines():
        dato = json.loads(renglon)
        if dato["texto"] not in existentes:
            memoria.agregar(dato["texto"], dato["tipo"], fecha=dato.get("fecha"))
            nuevos += 1
    return f"{nuevos} recuerdos nuevos · {len(memoria.recuerdos)} en {memoria.path}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Lista, importa y busca recuerdos.")
    parser.add_argument("--embedder", choices=("gemini", "fake"), default="gemini")
    parser.add_argument("--path", type=Path, default=MEMORIA_PATH, help=argparse.SUPPRESS)
    sub = parser.add_subparsers(dest="comando", required=True)
    sub.add_parser("listar", help="Todos los recuerdos.")
    p = sub.add_parser("importar", help="Agrega los recuerdos de un JSONL.")
    p.add_argument("archivo", type=Path)
    p = sub.add_parser("buscar", help="Los recuerdos más parecidos a una consulta.")
    p.add_argument("consulta")
    p.add_argument("--k", type=int, default=3)
    p = sub.add_parser("borrar", help="Borra un recuerdo por id.")
    p.add_argument("id")
    args = parser.parse_args(argv)

    memoria = Memoria(make_embedder(args.embedder), path=args.path)
    if args.comando == "listar":
        print("\n".join(map(linea, memoria.recuerdos.values())) or "La memoria está vacía.")
    elif args.comando == "importar":
        print(importar(memoria, args.archivo))
    elif args.comando == "buscar":
        for recuerdo, similitud in memoria.buscar(args.consulta, args.k):
            print(f"{similitud:.2f}  {linea(recuerdo)}")
    elif args.comando == "borrar":
        if args.id not in memoria.recuerdos:
            print(f"Error: no existe el recuerdo {args.id}.", file=sys.stderr)
            return 1
        print(f"Borrado: {linea(memoria.borrar(args.id))}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
