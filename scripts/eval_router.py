"""¿Qué fuente responde cada pregunta? La regla y el router con LLM contra la etiqueta.

    uv run scripts/eval_router.py
    uv run scripts/eval_router.py --llm --detalle
    uv run scripts/eval_router.py --llm --report reports/router.md
    uv run scripts/eval_router.py --golden data/golden/router_v1.jsonl   (reto de la Clase 8)

Cada pregunta de data/golden/router_v2.jsonl trae sus `fuentes_esperadas` y su
`fuente_preferida`: busqueda, grafo o memoria. Cada política elige una fuente por pregunta y
se mide con lo que trajo esa fuente. La memoria es la de data/memorias_demo.jsonl, cargada en
una carpeta temporal: tu var/memoria.json no cambia el resultado. Lee var/grafo.json: créalo
con `uv run scripts/build_graph.py`.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))  # Para importar los paquetes del repo al correr como script.

from knowledge.embeddings import make_embedder  # noqa: E402
from knowledge.graph import GRAPH_PATH  # noqa: E402
from knowledge.router import Router, por_reglas  # noqa: E402
from knowledge.store import COLLECTION, STORE_PATH, EmbeddingMismatch, open_store  # noqa: E402
from memory.recuerdos import Memoria  # noqa: E402
from scripts.eval_multihop import recall  # noqa: E402
from scripts.eval_retrieval import (  # noqa: E402
    build_llm_client,
    load_golden,
    load_graph,
    make_retriever,
)
from scripts.memoria import importar  # noqa: E402

DEFAULT_GOLDEN = ROOT / "data" / "golden" / "router_v2.jsonl"
MEMORIAS_DEMO = ROOT / "data" / "memorias_demo.jsonl"
K = 5
FUENTES = ("busqueda", "grafo", "memoria")


def politicas(golden: list[dict], router: Router | None = None) -> dict[str, dict[str, str]]:
    """La fuente que elige cada política, por pregunta. Con `router`, también la del LLM."""
    elegidas = {
        "siempre busqueda": {q["id"]: "busqueda" for q in golden},
        "siempre grafo": {q["id"]: "grafo" for q in golden},
        "regla": {q["id"]: por_reglas(q["pregunta"]).fuente for q in golden},
    }
    if router:
        elegidas["llm"] = {q["id"]: router.elegir(q["pregunta"]).fuente for q in golden}
    elegidas["etiqueta"] = {q["id"]: q["fuente_preferida"] for q in golden}
    return elegidas


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
    """Por pregunta: recall de cada fuente, la etiqueta y lo que eligió cada router."""
    routers = [nombre for nombre in ("regla", "llm") if nombre in elegidas]
    filas = [
        "| id | " + " | ".join(FUENTES) + " | etiqueta | " + " | ".join(routers) + " |",
        "|---" * (len(FUENTES) + len(routers) + 2) + "|",
    ]
    for q in golden:
        celdas = [f"{recalls[f][q['id']]:.1f}" for f in FUENTES] + [q["fuente_preferida"]]
        for nombre in routers:
            fuente = elegidas[nombre][q["id"]]
            celdas.append(fuente if fuente == q["fuente_preferida"] else f"**{fuente}**")
        filas.append(f"| {q['id']} | " + " | ".join(celdas) + " |")
    return "\n".join(filas)


def fallas(
    golden: list[dict], elegidas: dict[str, dict[str, str]], recalls: dict, politica: str
) -> list[str]:
    """Una línea por pregunta donde `politica` no eligió la fuente de la etiqueta."""
    lineas = []
    for q in golden:
        eligio, etiqueta = elegidas[politica][q["id"]], q["fuente_preferida"]
        if eligio != etiqueta:
            lineas.append(
                f"- {q['id']}: eligió {eligio} y la etiqueta dice {etiqueta} "
                f"(recall {recalls[eligio][q['id']]:.1f} contra {recalls[etiqueta][q['id']]:.1f})."
                f" «{q['pregunta']}»"
            )
    return lineas


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compara routers: búsqueda, grafo o memoria.")
    parser.add_argument("--k", type=int, default=K)
    parser.add_argument("--golden", type=Path, default=DEFAULT_GOLDEN)
    parser.add_argument("--embedder", choices=("gemini", "fake"), default="gemini")
    parser.add_argument(
        "--busqueda",
        choices=("vector", "bm25", "hibrida"),
        default="hibrida",
        help="Modo de búsqueda de la fuente busqueda (hibrida por defecto).",
    )
    parser.add_argument("--llm", action="store_true", help="Agrega el router con LLM.")
    parser.add_argument("--detalle", action="store_true", help="Recall por pregunta y fuente.")
    parser.add_argument("--report", type=Path, help="Además, escribe las tablas en este archivo.")
    parser.add_argument("--store", default=STORE_PATH, help=f"Carpeta del índice ({STORE_PATH}).")
    parser.add_argument("--grafo", type=Path, default=GRAPH_PATH, help=f"Grafo ({GRAPH_PATH}).")
    parser.add_argument("--memorias", type=Path, default=MEMORIAS_DEMO, help=argparse.SUPPRESS)
    parser.add_argument("--collection", default=COLLECTION, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)

    embedder = make_embedder(args.embedder)
    try:
        store = open_store(embedder, path=args.store, name=args.collection)
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
    with tempfile.TemporaryDirectory() as carpeta:
        memoria = Memoria(embedder, path=Path(carpeta) / "memoria.json")
        importar(memoria, args.memorias)
        retrievers = {
            "busqueda": make_retriever(store, args.busqueda),
            "grafo": make_retriever(store, "grafo", grafo=grafo),
            "memoria": memoria,
        }
        recalls = evaluate(retrievers, golden, args.k)
        router = Router(retrievers, build_llm_client()) if args.llm else None
        if router:
            with router.client.track() as uso:
                elegidas = politicas(golden, router)
        else:
            elegidas = politicas(golden)

    titulo = (
        f"recall@{args.k} sobre {args.golden.name} ({len(golden)} preguntas, "
        f"busqueda = {args.busqueda}, embedder {args.embedder})"
    )
    salida = [titulo, "", format_table(golden, elegidas, recalls, args.k)]
    if router:
        salida += ["", f"llm: {uso.calls} llamadas · {uso.tokens:,} tokens · ${uso.cost_usd:.6f}"]
        if router.respaldos:  # sin este aviso, la fila llm mezclaría la regla sin decirlo
            salida.append(
                f"aviso: en {router.respaldos} de {len(golden)} preguntas falló el LLM y decidió "
                "la regla; la fila llm no es solo del LLM."
            )
    if args.detalle:
        salida += ["", format_detail(golden, elegidas, recalls)]
    texto = "\n".join(salida)
    print(texto)
    if args.report:
        cuerpo = [texto]
        for politica in ("regla", "llm"):
            if politica in elegidas:
                lineas = fallas(golden, elegidas, recalls, politica) or ["- Ninguna."]
                cuerpo += ["", f"## Dónde falla: {politica}", "", *lineas]
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text("# ¿Qué fuente?\n\n" + "\n".join(cuerpo) + "\n", encoding="utf-8")
        print(f"\nReporte: {args.report}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
