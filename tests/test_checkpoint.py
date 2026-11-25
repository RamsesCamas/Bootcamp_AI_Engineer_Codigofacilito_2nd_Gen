"""Reto de la Clase 9: reanudar una ejecución del agente ReAct que falló. Sin red."""

from __future__ import annotations

import json

import pytest

import main
import tools.tickets
from agents import ReActAgent
from agents.checkpoint import Ejecucion
from agents.tools import ToolRunner
from contracts import BUSCAR_TICKET
from contracts.tool_schemas import CREAR_TICKET
from core.errors import AllProvidersFailedError, PermanentProviderError
from core.fallas import ClienteConFalla
from core.llm_client import LLMClient
from main import parse_args, run_agent
from tests.agent_helpers import answer, reply, use_tool
from tests.conftest import FakeProvider, make_response

PREGUNTA = "Compara el T-1042 y el T-1043"
CAIDA = PermanentProviderError("gemini", "sin cuota")
CREAR = use_tool(
    "crear_ticket", asunto="Falla en pagos", descripcion="Rechaza tarjetas", prioridad="alta"
)


class Caida(Exception):
    """El proceso muere a la mitad de un paso."""


@pytest.fixture(autouse=True)
def _en_tmp(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # var/ejecuciones/ queda en tmp_path
    monkeypatch.setattr(main, "TOOL_LOG_PATH", tmp_path / "tool_calls.jsonl")


def cliente(script, logger, fake_sleep):
    provider = FakeProvider("gemini", [reply(p) if isinstance(p, dict) else p for p in script])
    return provider, LLMClient([provider], logger, sleep=fake_sleep)


def tickets_creados() -> int:
    return len(tools.tickets._nuevos()["tickets"])


def test_guarda_cada_paso(logger, fake_sleep):
    _, client = cliente([use_tool(), answer("Está en revisión.")], logger, fake_sleep)
    ejecucion = Ejecucion.crear("e1", PREGUNTA)

    ejecucion.terminar(
        ReActAgent(client, [BUSCAR_TICKET]).run(PREGUNTA, al_paso=ejecucion.registrar)
    )

    datos = json.loads(ejecucion.path.read_text(encoding="utf-8"))
    assert [p["paso"]["herramienta"] for p in datos["pasos"]] == ["buscar_ticket"]
    assert datos["pasos"][0]["observacion"]["id"] == "T-1042"
    assert (datos["motivo"], datos["respuesta"]) == ("respuesta", "Está en revisión.")
    assert not list(ejecucion.path.parent.glob("*.tmp"))  # escritura atómica


def test_reanuda_sin_repetir_herramientas(logger, fake_sleep):
    _, client = cliente([use_tool(), use_tool(ticket_id="T-1043"), CAIDA], logger, fake_sleep)
    ejecucion = Ejecucion.crear("e2", PREGUNTA)
    primera = ReActAgent(client, [BUSCAR_TICKET]).run(PREGUNTA, al_paso=ejecucion.registrar)
    ejecucion.terminar(primera)
    assert (primera.motivo, len(primera.traza)) == ("error", 2)

    guardada = Ejecucion.cargar("e2")
    provider, client = cliente([answer("El T-1043 es más urgente.")], logger, fake_sleep)
    runner = ToolRunner([BUSCAR_TICKET])
    agente = ReActAgent(client, [BUSCAR_TICKET], runner=runner)
    segunda = agente.run(guardada.pregunta, traza=guardada.traza(), al_paso=guardada.registrar)

    assert guardada.reanudable and segunda.motivo == "respuesta"
    assert len(provider.calls) == 1  # solo la llamada que faltaba
    assert runner.eventos == []  # ninguna herramienta se volvió a ejecutar
    assert "Paso 2 · Observación:" in provider.calls[0]["messages"][-1]["content"]


def test_no_reanuda_una_que_respondio(logger, fake_sleep, capsys):
    _, client = cliente([answer("Listo.")], logger, fake_sleep)

    assert run_agent(client, "react", PREGUNTA, verbose=False, ejecucion="e3") == 0
    assert run_agent(client, "react", None, verbose=False, reanudar="e3") == 1
    assert "ya terminó (respuesta)" in capsys.readouterr().err


def test_la_cli_reanuda_despues_de_la_caida(logger, fake_sleep, capsys):
    _, base = cliente([use_tool(), use_tool(ticket_id="T-1043")], logger, fake_sleep)
    falla = ClienteConFalla(base, 3)

    assert run_agent(falla, "react", PREGUNTA, verbose=False, ejecucion="demo") == 2
    assert "--agent react --reanudar demo" in capsys.readouterr().err

    provider, client = cliente([answer("El T-1043.")], logger, fake_sleep)
    assert run_agent(client, "react", None, verbose=False, reanudar="demo") == 0
    assert "reanudada: 2 pasos guardados" in capsys.readouterr().out
    assert len(provider.calls) == 1


def test_mismo_request_id_al_reanudar_no_duplica_el_ticket(logger, fake_sleep):
    ejecucion = Ejecucion.crear("e5", "Crea un ticket: el portal de pagos rechaza tarjetas")

    def se_cae(step):  # después de crear el ticket y antes de guardar el paso
        raise Caida

    _, client = cliente([CREAR], logger, fake_sleep)
    runner = ToolRunner([CREAR_TICKET], request_id=ejecucion.request_id)
    with pytest.raises(Caida):
        ReActAgent(client, [CREAR_TICKET], runner=runner).run(ejecucion.pregunta, al_paso=se_cae)
    assert tickets_creados() == 1

    guardada = Ejecucion.cargar("e5")
    assert guardada.pasos == []  # el paso no se guardó: el LLM vuelve a pedir el ticket
    _, client = cliente([CREAR, answer("Creé el T-2001.")], logger, fake_sleep)
    runner = ToolRunner([CREAR_TICKET], request_id=guardada.request_id)
    resultado = ReActAgent(client, [CREAR_TICKET], runner=runner).run(guardada.pregunta)

    assert resultado.traza[0].observacion["duplicado_evitado"] is True
    assert tickets_creados() == 1


def test_con_otro_request_id_el_ticket_se_duplica(logger, fake_sleep):
    for _ in range(2):
        _, client = cliente([CREAR, answer("Listo.")], logger, fake_sleep)
        runner = ToolRunner([CREAR_TICKET])  # un request_id nuevo en cada ejecución
        ReActAgent(client, [CREAR_TICKET], runner=runner).run("Crea un ticket")

    assert tickets_creados() == 2


def test_cliente_con_falla(logger, fake_sleep):
    _, base = cliente([make_response("gemini"), make_response("gemini")], logger, fake_sleep)
    client = ClienteConFalla(base, 3)

    client.generate([{"role": "user", "content": "uno"}])
    client.generate([{"role": "user", "content": "dos"}])
    with pytest.raises(AllProvidersFailedError, match="caída simulada en la llamada 3"):
        client.generate([{"role": "user", "content": "tres"}])


@pytest.mark.parametrize("ejecucion_id", ["../secretos", "a/b", "", "x" * 41])
def test_id_invalido(ejecucion_id):
    with pytest.raises(ValueError, match="id de ejecución inválido"):
        Ejecucion.cargar(ejecucion_id)


def test_reanudar_usa_la_pregunta_guardada():
    assert parse_args(["--agent", "react", "--reanudar", "demo"]).reanudar == "demo"
    for argv in (
        ["--agent", "react", "--reanudar", "demo", "otra pregunta"],
        ["--agent", "react", "--ejecucion", "a", "--reanudar", "a"],
        ["--agent", "plan", "--ejecucion", "a", "¿Y el T-1043?"],
        ["--falla-en", "3", "hola"],
    ):
        with pytest.raises(SystemExit):
            parse_args(argv)
