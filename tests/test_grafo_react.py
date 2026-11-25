"""Clase 11: el agente ReAct como grafo de LangGraph. Sin red."""

from __future__ import annotations

import json

import pytest

import main
from agents import AgentBudget
from agents.grafo_react import AgenteGrafo
from contracts import BUSCAR_TICKET
from core.llm_client import LLMClient
from main import parse_args, run_agent
from tests.agent_helpers import answer, reply, use_tool
from tests.conftest import FakeProvider

PREGUNTA = "Compara el T-1042 y el T-1043"


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


def test_agent_grafo_en_la_cli(logger, fake_sleep, capsys):
    provider, client = cliente([use_tool(), answer("Está en revisión.")], logger, fake_sleep)

    assert parse_args(["--agent", "grafo", "¿Cómo va el T-1042?"]).agent == "grafo"
    assert run_agent(client, "grafo", "¿Cómo va el T-1042?", verbose=False) == 0
    assert "Está en revisión." in capsys.readouterr().out
    assert len(provider.calls) == 2
