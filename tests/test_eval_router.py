from __future__ import annotations

import json
import shutil

import pytest

import main
import scripts.eval_router
import scripts.memoria
from core.llm_client import LLMClient
from scripts.build_graph import main as build_main
from scripts.eval_retrieval import ROOT, load_golden
from scripts.eval_router import DEFAULT_GOLDEN, FUENTES, MEMORIAS_DEMO
from scripts.eval_router import main as router_main
from scripts.ingest_docs import main as ingest_main
from tests.conftest import FakeProvider, make_response


@pytest.fixture(scope="module")
def carpeta(tmp_path_factory):
    """Índice con el embedder falso y grafo de reglas, una vez para todo el módulo."""
    carpeta = tmp_path_factory.mktemp("router")
    store = str(carpeta / "chroma")
    assert ingest_main([str(ROOT / "data" / "docs"), "--embedder", "fake", "--store", store]) == 0
    assert build_main(["--out", str(carpeta / "grafo.json")]) == 0
    return carpeta


@pytest.fixture
def fuentes(carpeta) -> list[str]:
    """Los argumentos de eval_router para usar ese índice y ese grafo."""
    store, grafo = str(carpeta / "chroma"), str(carpeta / "grafo.json")
    return ["--embedder", "fake", "--store", store, "--grafo", grafo, "--busqueda", "bm25"]


def test_router_v2_suma_parafrasis_y_preguntas_de_memoria():
    golden = load_golden(DEFAULT_GOLDEN)
    memorias = len(MEMORIAS_DEMO.read_text(encoding="utf-8").splitlines())

    assert len(golden) == len({q["id"] for q in golden}) == 38
    assert {q["fuente_preferida"] for q in golden} == set(FUENTES)
    grafo = [q["id"] for q in golden if q["fuente_preferida"] == "grafo"]
    assert grafo == ["m1", "m2", "m4", "p1", "p2", "p3", "p4"]
    assert [q["id"] for q in golden if q["fuente_preferida"] == "memoria"] == [
        f"u{n}" for n in range(1, 9)
    ]
    for q in golden:
        for fuente in q["fuentes_esperadas"]:
            if fuente.startswith("memoria/r"):
                assert 1 <= int(fuente.removeprefix("memoria/r")) <= memorias
            else:
                assert (ROOT / "data" / "docs" / fuente).exists(), fuente


def test_compara_la_regla_con_una_sola_fuente(fuentes, tmp_path, capsys):
    capsys.readouterr()
    report = tmp_path / "reports" / "router.md"

    assert router_main([*fuentes, "--report", str(report)]) == 0

    out = capsys.readouterr().out
    assert "| siempre busqueda | 23/38 | 0.70 | 23/38 |" in out
    assert "| siempre grafo | 7/38 | 0.50 | 19/38 |" in out
    assert "| regla | 25/38 | 0.74 | 26/38 |" in out
    assert "| etiqueta | 38/38 | 1.00 | 38/38 |" in out
    texto = report.read_text(encoding="utf-8")
    assert "## Dónde falla: regla" in texto
    assert "- q09: eligió memoria y la etiqueta dice busqueda" in texto
    assert "- u1: eligió busqueda y la etiqueta dice memoria" in texto


def test_el_router_con_llm_se_mide_igual(
    fuentes, tmp_path, monkeypatch, logger, fake_sleep, capsys
):
    golden = tmp_path / "router_mini.jsonl"
    lineas = DEFAULT_GOLDEN.read_text(encoding="utf-8").splitlines()
    golden.write_text(
        "\n".join(x for x in lineas if json.loads(x)["id"] in ("q01", "p1", "u1")) + "\n",
        encoding="utf-8",
    )
    respuestas = [
        make_response("gemini", json.dumps({"motivo": "Prueba.", "fuente": fuente}))
        for fuente in ("busqueda", "grafo", "memoria")
    ]
    client = LLMClient([FakeProvider("gemini", respuestas)], logger, sleep=fake_sleep)
    monkeypatch.setattr(scripts.eval_router, "build_llm_client", lambda: client)
    capsys.readouterr()

    assert router_main([*fuentes, "--golden", str(golden), "--llm", "--detalle"]) == 0

    out = capsys.readouterr().out
    assert "| regla | 1/3 | 0.50 | 1/3 |" in out
    assert "| llm | 3/3 | 1.00 | 3/3 |" in out
    assert "llm: 3 llamadas · 45 tokens · $0.000300" in out
    assert "| u1 | 0.0 | 0.0 | 1.0 | memoria | **busqueda** | memoria |" in out


def test_main_modo_auto_responde_con_la_memoria(
    carpeta, tmp_path, monkeypatch, logger, fake_sleep, capsys
):
    # main.py lee .chroma/, var/grafo.json y var/memoria.json desde la carpeta actual.
    shutil.copytree(carpeta / "chroma", tmp_path / ".chroma")
    (tmp_path / "var").mkdir()
    shutil.copy(carpeta / "grafo.json", tmp_path / "var" / "grafo.json")
    monkeypatch.chdir(tmp_path)
    assert scripts.memoria.main(["--embedder", "fake", "importar", str(MEMORIAS_DEMO)]) == 0
    respuestas = [
        json.dumps({"motivo": "Pregunta por algo que hizo el usuario.", "fuente": "memoria"}),
        json.dumps({"encontrado": True, "respuesta": "A Guardia N2 [1].", "citas": [1]}),
    ]
    provider = FakeProvider("gemini", [make_response("gemini", r) for r in respuestas])
    client = LLMClient([provider], logger, sleep=fake_sleep)
    monkeypatch.setattr(main, "build_client", lambda args: client)
    capsys.readouterr()

    pregunta = "¿A quién escalé el T-1043?"
    assert main.main(["--rag", pregunta, "--modo", "auto", "--embedder", "fake"]) == 0

    out = capsys.readouterr().out
    assert "[1] memoria/r3" in out
    assert "ruta: memoria (decidió llm)" in out
    assert "fuentes: busqueda, grafo, memoria" in out
