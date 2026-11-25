"""Clase 11: scripts/ejecuciones.py lee los checkpoints sin llamar al LLM. Sin red."""

from __future__ import annotations

from agents.grafo_react import AgenteGrafo, abrir_checkpointer
from contracts import BUSCAR_TICKET
from core.llm_client import LLMClient
from scripts.ejecuciones import main as ejecuciones_main
from tests.agent_helpers import answer, reply, use_tool
from tests.conftest import FakeProvider


def una_ejecucion(db, logger, fake_sleep) -> None:
    """Dos herramientas y la respuesta: tres veces decidir y dos veces actuar."""
    script = [use_tool(), use_tool(ticket_id="T-1043"), answer("El T-1043 va más atrasado.")]
    provider = FakeProvider("gemini", [reply(paso) for paso in script])
    client = LLMClient([provider], logger, sleep=fake_sleep)
    with abrir_checkpointer(db) as saver:
        grafo = AgenteGrafo(client, [BUSCAR_TICKET], checkpointer=saver)
        grafo.run("Compara el T-1042 y el T-1043", ejecucion="e1")


def test_historial(tmp_path, logger, fake_sleep, capsys):
    db = tmp_path / "checkpoints.db"
    una_ejecucion(db, logger, fake_sleep)

    assert ejecuciones_main(["--db", str(db), "historial", "e1"]) == 0

    renglones = capsys.readouterr().out.splitlines()[1:]  # sin el encabezado
    corrio = [renglon.split()[2] for renglon in renglones]
    assert corrio.count("decidir") == 3 and corrio.count("actuar") == 2
    assert renglones[-1].split()[-2:] == ["FIN", "2"]


def test_listar(tmp_path, logger, fake_sleep, capsys):
    db = tmp_path / "checkpoints.db"
    una_ejecucion(db, logger, fake_sleep)

    assert ejecuciones_main(["--db", str(db), "listar"]) == 0
    assert capsys.readouterr().out.strip() == "e1  2 pasos · terminó: respuesta"


def test_sin_checkpoints_o_sin_esa_ejecucion(tmp_path, logger, fake_sleep, capsys):
    db = tmp_path / "checkpoints.db"
    assert ejecuciones_main(["--db", str(db), "listar"]) == 1

    una_ejecucion(db, logger, fake_sleep)
    assert ejecuciones_main(["--db", str(db), "historial", "otra"]) == 1
    assert "no hay una ejecución otra" in capsys.readouterr().err


def test_grafo_en_mermaid(capsys):
    assert ejecuciones_main(["grafo"]) == 0
    salida = capsys.readouterr().out
    assert "decidir" in salida and "actuar" in salida
