"""Clase 11: el agente ReAct como grafo de LangGraph. Sin red."""

from __future__ import annotations

import json

import pytest
from langgraph.checkpoint.memory import InMemorySaver

import main
import tools.tickets
from agents import AgentBudget
from agents.grafo_react import AgenteGrafo, abrir_checkpointer
from agents.tools import ToolRunner
from contracts import BUSCAR_TICKET
from contracts.tool_schemas import CREAR_TICKET
from core.errors import PermanentProviderError
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
    """El proceso muere a la mitad de un nodo."""


@pytest.fixture(autouse=True)
def _en_tmp(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # var/ queda en tmp_path
    monkeypatch.setattr(main, "TOOL_LOG_PATH", tmp_path / "tool_calls.jsonl")


def cliente(script, logger, fake_sleep):
    provider = FakeProvider("gemini", [reply(p) if isinstance(p, dict) else p for p in script])
    return provider, LLMClient([provider], logger, sleep=fake_sleep)


def agente(script, logger, fake_sleep, **kwargs):
    provider, client = cliente(script, logger, fake_sleep)
    return provider, AgenteGrafo(client, [BUSCAR_TICKET], **kwargs)


def tickets_creados() -> int:
    return len(tools.tickets._nuevos()["tickets"])


def test_responde_como_el_react(logger, fake_sleep):
    eventos = []
    _, grafo = agente(
        [use_tool(), answer("Está en revisión.")],
        logger,
        fake_sleep,
        on_event=lambda kind, data: eventos.append(kind),
    )

    resultado = grafo.run("¿Cómo va el T-1042?")

    assert (resultado.motivo, resultado.respuesta) == ("respuesta", "Está en revisión.")
    [paso] = resultado.traza
    assert paso.paso.herramienta == "buscar_ticket"
    assert paso.observacion["estado"] == "en revisión"
    assert (resultado.llamadas, resultado.tokens) == (2, 30)
    assert eventos == ["paso", "observacion", "paso", "fin"]


def test_el_segundo_paso_lleva_la_traza(logger, fake_sleep):
    provider, grafo = agente([use_tool(), answer("Listo.")], logger, fake_sleep)

    grafo.run("¿Cómo va el T-1042?")

    contexto = provider.calls[1]["messages"][-1]["content"]
    assert "<traza>" in contexto and "Paso 1 · Observación:" in contexto


def test_presupuesto(logger, fake_sleep):
    script = [use_tool(), use_tool(ticket_id="T-1043")]
    provider, grafo = agente(script, logger, fake_sleep, budget=AgentBudget(1))

    resultado = grafo.run(PREGUNTA)

    assert resultado.motivo == "max_pasos"
    assert len(provider.calls) == 1  # el presupuesto se revisa antes de llamar al LLM


def test_el_estado_es_json(logger, fake_sleep):
    _, grafo = agente([use_tool(), answer("Listo.")], logger, fake_sleep)

    grafo.run("¿Cómo va el T-1042?", ejecucion="e1")

    valores = grafo.grafo.get_state({"configurable": {"thread_id": "e1"}}).values
    assert json.loads(json.dumps(valores)) == valores
    assert valores["pasos"][0]["paso"]["herramienta"] == "buscar_ticket"
    assert valores["pendiente"] is None


def test_dos_nodos_y_una_arista_condicional(logger, fake_sleep):
    _, grafo = agente([], logger, fake_sleep)

    dibujo = grafo.grafo.get_graph()

    assert set(dibujo.nodes) == {"__start__", "decidir", "actuar", "__end__"}
    condicionales = {(e.source, e.target) for e in dibujo.edges if e.conditional}
    assert condicionales == {("decidir", "actuar"), ("decidir", "__end__")}


def test_los_checkpoints_sobreviven_al_proceso(logger, fake_sleep, tmp_path):
    db = tmp_path / "checkpoints.db"
    _, client = cliente([use_tool(), answer("Listo.")], logger, fake_sleep)
    with abrir_checkpointer(db) as saver:
        AgenteGrafo(client, [BUSCAR_TICKET], checkpointer=saver).run(
            "¿Y el T-1042?", ejecucion="e1"
        )

    with abrir_checkpointer(db) as saver:  # otro proceso: una conexión nueva al mismo archivo
        foto = AgenteGrafo(None, [], checkpointer=saver).grafo.get_state(
            {"configurable": {"thread_id": "e1"}}
        )

    assert foto.next == ()  # terminó
    assert [p["paso"]["herramienta"] for p in foto.values["pasos"]] == ["buscar_ticket"]


def test_un_id_es_una_ejecucion(logger, fake_sleep):
    _, grafo = agente([answer("Listo.")], logger, fake_sleep)
    grafo.run("Hola", ejecucion="e1")

    with pytest.raises(ValueError, match="ya existe la ejecución e1"):
        grafo.run("Otra pregunta", ejecucion="e1")


def test_agent_grafo_en_la_cli(logger, fake_sleep, capsys):
    provider, client = cliente([use_tool(), answer("Está en revisión.")], logger, fake_sleep)

    assert parse_args(["--agent", "grafo", "¿Cómo va el T-1042?"]).agent == "grafo"
    assert run_agent(client, "grafo", "¿Cómo va el T-1042?", verbose=False) == 0
    salida = capsys.readouterr().out
    assert "Está en revisión." in salida
    assert "checkpoints en var" in salida
    assert len(provider.calls) == 2


def test_la_cli_rechaza_un_id_repetido(logger, fake_sleep, capsys):
    _, client = cliente([answer("Listo.")], logger, fake_sleep)

    assert run_agent(client, "grafo", "Hola", verbose=False, ejecucion="g0") == 0
    assert "ejecución g0" in capsys.readouterr().out
    assert run_agent(client, "grafo", "Otra", verbose=False, ejecucion="g0") == 1
    assert "ya existe la ejecución g0" in capsys.readouterr().err


def test_falla_y_reanuda_sin_repetir_herramientas(logger, fake_sleep):
    saver = InMemorySaver()  # el mismo checkpointer para la ejecución que falla y la que sigue
    script = [use_tool(), use_tool(ticket_id="T-1043"), CAIDA]
    _, grafo = agente(script, logger, fake_sleep, checkpointer=saver)
    primera = grafo.run(PREGUNTA, ejecucion="e1")
    assert (primera.motivo, len(primera.traza)) == ("error", 2)
    assert grafo.grafo.get_state({"configurable": {"thread_id": "e1"}}).next == ("decidir",)

    runner = ToolRunner([BUSCAR_TICKET])
    provider, grafo = agente(
        [answer("El T-1043 va más atrasado.")],
        logger,
        fake_sleep,
        checkpointer=saver,
        runner=runner,
    )
    segunda = grafo.reanudar("e1")

    assert (segunda.motivo, len(segunda.traza)) == ("respuesta", 2)
    assert len(provider.calls) == 1  # solo el decidir que había fallado
    assert runner.eventos == []  # ninguna herramienta se volvió a ejecutar


def test_reanudar_valida_la_ejecucion(logger, fake_sleep):
    _, grafo = agente([answer("Listo.")], logger, fake_sleep)
    grafo.run("Hola", ejecucion="e1")

    with pytest.raises(ValueError, match="ya terminó"):
        grafo.reanudar("e1")
    with pytest.raises(LookupError, match="no hay una ejecución e9"):
        grafo.reanudar("e9")


def test_si_se_cae_al_escribir_repite_la_misma_accion(logger, fake_sleep):
    saver = InMemorySaver()

    def se_cae(kind, data):  # después de crear el ticket y antes del checkpoint de actuar
        if kind == "observacion":
            raise Caida

    _, client = cliente([CREAR], logger, fake_sleep)
    runner = ToolRunner([CREAR_TICKET])
    grafo = AgenteGrafo(client, [CREAR_TICKET], runner=runner, checkpointer=saver, on_event=se_cae)
    with pytest.raises(Caida):
        grafo.run("Crea un ticket: el portal de pagos rechaza tarjetas", ejecucion="e1")
    assert tickets_creados() == 1

    provider, client = cliente([answer("Creé el T-2001.")], logger, fake_sleep)
    runner = ToolRunner([CREAR_TICKET])  # otro proceso: su request_id sale del estado
    grafo = AgenteGrafo(client, [CREAR_TICKET], runner=runner, checkpointer=saver)
    resultado = grafo.reanudar("e1")

    assert len(provider.calls) == 1  # la acción ya estaba en el estado: el LLM no decide otra vez
    assert resultado.traza[0].observacion["duplicado_evitado"] is True
    assert tickets_creados() == 1


def test_la_cli_reanuda_el_grafo(logger, fake_sleep, capsys):
    _, base = cliente([use_tool(), use_tool(ticket_id="T-1043")], logger, fake_sleep)
    falla = ClienteConFalla(base, 3)

    assert run_agent(falla, "grafo", PREGUNTA, verbose=False, ejecucion="g1") == 2
    assert "--agent grafo --reanudar g1" in capsys.readouterr().err

    provider, client = cliente([answer("El T-1043.")], logger, fake_sleep)
    assert run_agent(client, "grafo", None, verbose=False, reanudar="g1") == 0
    salida = capsys.readouterr().out
    assert "reanudada: 2 pasos guardados" in salida and "El T-1043." in salida
    assert len(provider.calls) == 1


def test_reanudar_el_grafo_desde_la_cli():
    assert parse_args(["--agent", "grafo", "--reanudar", "g1"]).reanudar == "g1"
    with pytest.raises(SystemExit):
        parse_args(["--agent", "grafo", "--mcp", "--ejecucion", "g1", "¿Y el T-1043?"])


def verificacion(aprobada: bool, *problemas: str) -> dict:
    return {"problemas": list(problemas), "aprobada": aprobada}


def test_verificar_aprueba(logger, fake_sleep):
    script = [use_tool(), answer("Está en revisión."), verificacion(True)]
    _, grafo = agente(script, logger, fake_sleep, verificar=True)

    resultado = grafo.run("¿Cómo va el T-1042?", ejecucion="v1")

    valores = grafo.grafo.get_state({"configurable": {"thread_id": "v1"}}).values
    assert resultado.respuesta == "Está en revisión."
    assert valores["verificacion"] == {"problemas": [], "aprobada": True}


def test_verificar_corrige(logger, fake_sleep):
    script = [
        use_tool(),
        answer("Ya está resuelto."),
        verificacion(False, "La traza dice que está en revisión."),
        make_response("gemini", "Está en revisión."),
    ]
    _, grafo = agente(script, logger, fake_sleep, verificar=True)

    resultado = grafo.run("¿Cómo va el T-1042?", ejecucion="v1")

    valores = grafo.grafo.get_state({"configurable": {"thread_id": "v1"}}).values
    assert resultado.respuesta == "Está en revisión."
    assert valores["verificacion"]["problemas"] == ["La traza dice que está en revisión."]


def test_el_nodo_verificar_es_opcional(logger, fake_sleep):
    _, sin = agente([], logger, fake_sleep)
    _, con = agente([], logger, fake_sleep, verificar=True)

    assert "verificar" not in sin.grafo.get_graph().nodes
    assert "verificar" in con.grafo.get_graph().nodes
