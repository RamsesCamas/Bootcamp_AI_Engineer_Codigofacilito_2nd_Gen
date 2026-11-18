from __future__ import annotations

from scripts.build_graph import main as build_main
from scripts.eval_retrieval import ROOT, load_golden
from scripts.eval_router import DEFAULT_GOLDEN, FUENTES, elegir_fuente
from scripts.eval_router import main as router_main
from scripts.ingest_docs import main as ingest_main


def test_the_rule_needs_a_ticket_and_its_provider_or_contract():
    assert (
        elegir_fuente("¿Qué penalización aplica si el proveedor del T-1042 no cumple?") == "grafo"
    )
    assert elegir_fuente("El T-1043 lleva 3 horas, ¿ya incumplió su contrato?") == "grafo"
    assert elegir_fuente("¿Qué pasó con el T-1016?") == "busqueda"
    assert elegir_fuente("¿Cuál es el tiempo de respuesta de PagosRed según su contrato?") == (
        "busqueda"
    )


def test_the_golden_set_joins_rag_v3_and_multihop():
    golden = load_golden(DEFAULT_GOLDEN)
    assert len(golden) == 26
    assert len({q["id"] for q in golden}) == 26
    assert {q["fuente_preferida"] for q in golden} <= set(FUENTES)
    assert [q["id"] for q in golden if q["fuente_preferida"] == "grafo"] == ["m1", "m2", "m4"]
    for q in golden:
        for fuente in q["fuentes_esperadas"]:
            assert (ROOT / "data" / "docs" / fuente).exists(), fuente


def test_compares_the_rule_with_always_one_source(tmp_path, capsys):
    store = str(tmp_path / "chroma")
    assert ingest_main([str(ROOT / "data" / "docs"), "--embedder", "fake", "--store", store]) == 0
    grafo = tmp_path / "grafo.json"
    assert build_main(["--out", str(grafo)]) == 0
    capsys.readouterr()

    report = tmp_path / "reports" / "router.md"
    args = ["--embedder", "fake", "--store", store, "--grafo", str(grafo), "--busqueda", "bm25"]
    assert router_main([*args, "--report", str(report)]) == 0

    out = capsys.readouterr().out
    assert "| siempre busqueda | 23/26 | 0.94 | 23/26 |" in out
    assert "| siempre grafo | 3/26 | 0.58 | 15/26 |" in out
    assert "| regla | 25/26 | 1.00 | 26/26 |" in out
    assert "- m5: eligió grafo" in report.read_text(encoding="utf-8")
