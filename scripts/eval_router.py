"""¿Búsqueda o grafo? Qué fuente elige una regla y qué recall@5 da (reto de la Clase 8).

    uv run scripts/eval_router.py
    uv run scripts/eval_router.py --busqueda bm25 --detalle
    uv run scripts/eval_router.py --report reports/router.md

Cada pregunta de data/golden/router_v1.jsonl trae sus `fuentes_esperadas` y su
`fuente_preferida`: grafo solo si el grafo trajo más fuentes esperadas que la búsqueda; si
empatan, búsqueda. Cada política elige una fuente por pregunta y se mide con lo que trajo esa
fuente. Lee var/grafo.json: créalo con `uv run scripts/build_graph.py`.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))  # Para importar los paquetes del repo al correr como script.

from knowledge.embeddings import make_embedder  # noqa: E402
from knowledge.graph import GRAPH_PATH  # noqa: E402
from knowledge.store import COLLECTION, STORE_PATH, EmbeddingMismatch, open_store  # noqa: E402
from scripts.eval_multihop import recall  # noqa: E402
from scripts.eval_retrieval import load_golden, load_graph, make_retriever  # noqa: E402

DEFAULT_GOLDEN = ROOT / "data" / "golden" / "router_v1.jsonl"
K = 5
FUENTES = ("busqueda", "grafo")
_TICKET = re.compile(r"\bT-\d{4}\b")
_PROVEEDOR = re.compile(r"\b(proveedor|contrato)\b", re.IGNORECASE)


def elegir_fuente(pregunta: str) -> str:
    """Regla base: grafo si nombra un ticket y pregunta por su proveedor o su contrato.

    Es el caso de varios saltos: el ticket está en la pregunta y el dato, en el documento de
    otra entidad que la pregunta no nombra.
    """
    if _TICKET.search(pregunta) and _PROVEEDOR.search(pregunta):
        return "grafo"
    return "busqueda"


def politicas(golden: list[dict]) -> dict[str, dict[str, str]]:
    """La fuente que elige cada política, por pregunta."""
    return {
        "siempre busqueda": {q["id"]: "busqueda" for q in golden},
        "siempre grafo": {q["id"]: "grafo" for q in golden},
        "regla": {q["id"]: elegir_fuente(q["pregunta"]) for q in golden},
        "etiqueta": {q["id"]: q["fuente_preferida"] for q in golden},
    }


def evaluate(retrievers: dict, golden: list[dict], k: int) -> dict[str, dict[str, float]]:
    """recall@k de cada pregunta con cada fuente: recalls[fuente][id]."""
    return {
        fuente: {
            q["id"]: recall(
                [h.metadatos["fuente"] for h in retriever.search(q["pregunta"], k=k)],
                q["fuentes_esperadas"],
            )
            for q in golden
        }
        for fuente, retriever in retrievers.items()
    }


def format_table(
    golden: list[dict], elegidas: dict[str, dict[str, str]], recalls: dict, k: int
) -> str:
    n = len(golden)
    filas = [f"| política | fuente correcta | recall@{k} | completas |", "|---|---|---|---|"]
    for nombre, eleccion in elegidas.items():
        aciertos = sum(eleccion[q["id"]] == q["fuente_preferida"] for q in golden)
        valores = [recalls[eleccion[q["id"]]][q["id"]] for q in golden]
        completas = sum(valor == 1 for valor in valores)
        filas.append(f"| {nombre} | {aciertos}/{n} | {sum(valores) / n:.2f} | {completas}/{n} |")
    return "\n".join(filas)


def format_detail(golden: list[dict], elegidas: dict[str, dict[str, str]], recalls: dict) -> str:
    """Por pregunta: recall de cada fuente, la etiqueta y lo que eligió la regla."""
    filas = ["| id | busqueda | grafo | etiqueta | regla |", "|---|---|---|---|---|"]
    for q in golden:
        regla = elegidas["regla"][q["id"]]
        marca = regla if regla == q["fuente_preferida"] else f"**{regla}**"
        celdas = [f"{recalls[f][q['id']]:.1f}" for f in FUENTES]
        filas.append(
            f"| {q['id']} | " + " | ".join(celdas) + f" | {q['fuente_preferida']} | {marca} |"
        )
    return "\n".join(filas)


def fallas(golden: list[dict], elegidas: dict[str, dict[str, str]], recalls: dict) -> list[str]:
    """Una línea por pregunta donde la regla no eligió la fuente de la etiqueta."""
    lineas = []
    for q in golden:
        regla = elegidas["regla"][q["id"]]
        if regla != q["fuente_preferida"]:
            con_regla, con_etiqueta = (
                recalls[regla][q["id"]],
                recalls[q["fuente_preferida"]][q["id"]],
            )
            lineas.append(
                f"- {q['id']}: eligió {regla} y la etiqueta dice {q['fuente_preferida']} "
                f"(recall {con_regla:.1f} contra {con_etiqueta:.1f}). «{q['pregunta']}»"
            )
    return lineas


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compara reglas para elegir búsqueda o grafo.")
    parser.add_argument("--k", type=int, default=K)
    parser.add_argument("--golden", type=Path, default=DEFAULT_GOLDEN)
    parser.add_argument("--embedder", choices=("gemini", "fake"), default="gemini")
    parser.add_argument(
        "--busqueda",
        choices=("vector", "bm25", "hibrida"),
        default="hibrida",
        help="Modo de búsqueda de la fuente busqueda (hibrida por defecto).",
    )
    parser.add_argument("--detalle", action="store_true", help="Recall por pregunta y fuente.")
    parser.add_argument("--report", type=Path, help="Además, escribe las tablas en este archivo.")
    parser.add_argument("--store", default=STORE_PATH, help=f"Carpeta del índice ({STORE_PATH}).")
    parser.add_argument("--grafo", type=Path, default=GRAPH_PATH, help=f"Grafo ({GRAPH_PATH}).")
    parser.add_argument("--collection", default=COLLECTION, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)

    try:
        store = open_store(make_embedder(args.embedder), path=args.store, name=args.collection)
        grafo = load_graph(["grafo"], args.grafo)
    except (EmbeddingMismatch, ValueError, LookupError) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 2
    if store.count() == 0:
        print(
            "El índice está vacío: corre `uv run scripts/ingest_docs.py data/docs`.",
            file=sys.stderr,
        )
        return 2

    golden = load_golden(args.golden)
    retrievers = {
        "busqueda": make_retriever(store, args.busqueda),
        "grafo": make_retriever(store, "grafo", grafo=grafo),
    }
    recalls = evaluate(retrievers, golden, args.k)
    elegidas = politicas(golden)

    titulo = (
        f"recall@{args.k} sobre {args.golden.name} ({len(golden)} preguntas, "
        f"busqueda = {args.busqueda}, embedder {args.embedder})"
    )
    salida = [titulo, "", format_table(golden, elegidas, recalls, args.k)]
    if args.detalle:
        salida += ["", format_detail(golden, elegidas, recalls)]
    texto = "\n".join(salida)
    print(texto)
    if args.report:
        lineas = fallas(golden, elegidas, recalls) or [
            "- Ninguna: la regla eligió como la etiqueta."
        ]
        cuerpo = "\n".join([texto, "", "## Dónde falla la regla", "", *lineas])
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(f"# ¿Búsqueda o grafo?\n\n{cuerpo}\n", encoding="utf-8")
        print(f"\nReporte: {args.report}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
