from __future__ import annotations

import json
from datetime import datetime

import pytest

from agents import ReActAgent
from agents.tools import ToolRunner
from core.llm_client import LLMClient, ToolCall
from knowledge.embeddings import FakeEmbedder
from main import parse_args
from memory.recuerdos import Memoria
from scripts.memoria import ROOT, importar
from tests.agent_helpers import answer, reply, use_tool
from tests.conftest import FakeProvider
from tools.memoria import consultar_memoria_tool, guardar_recuerdo_tool


@pytest.fixture
def memoria(tmp_path) -> Memoria:
    memoria = Memoria(
        FakeEmbedder(), tmp_path / "memoria.json", reloj=lambda: datetime(2026, 11, 20)
    )
    importar(memoria, ROOT / "data" / "memorias_demo.jsonl")
    return memoria


def run_tool(runner: ToolRunner, name: str, arguments: dict) -> dict:
    message = runner.execute(ToolCall("paso-1", name, json.dumps(arguments)))
    return json.loads(message["content"])


def run(runner: ToolRunner, arguments: dict) -> dict:
    return run_tool(runner, "consultar_memoria", arguments)


def test_trae_los_recuerdos_mas_parecidos_con_su_fuente(memoria):
    runner = ToolRunner([consultar_memoria_tool(memoria)], rol="lectura")

    salida = run(runner, {"consulta": "pendiente con PagosRed"})

    assert len(salida["recuerdos"]) == 3
    assert salida["recuerdos"][0] == (
        "[memoria/r4] (episodico, 2026-11-13) Quedó de revisar si PagosRed entregó el reporte "
        "de conciliación del T-1014."
    )
    assert runner.eventos[-1]["tool_permiso"] == "lectura"


def test_con_la_memoria_vacia_lo_dice(tmp_path):
    runner = ToolRunner([consultar_memoria_tool(Memoria(FakeEmbedder(), tmp_path / "m.json"))])
    assert run(runner, {"consulta": "T-1043"}) == {
        "recuerdos": [],
        "nota": "La memoria está vacía.",
    }


GUARDIA = {"texto": "Cubre la guardia N2 los jueves.", "tipo": "semantico"}


def test_guardar_recuerdo_regresa_el_id_nuevo(memoria):
    runner = ToolRunner([guardar_recuerdo_tool(memoria)], rol="operador")

    salida = run_tool(runner, "guardar_recuerdo", GUARDIA)

    assert salida == {"id": "r9", "guardado": "Cubre la guardia N2 los jueves."}
    assert memoria.recuerdos["r9"].texto == "Cubre la guardia N2 los jueves."
    assert runner.eventos[-1]["tool_permiso"] == "escritura"


def test_el_mismo_texto_no_se_duplica(memoria):
    runner = ToolRunner([guardar_recuerdo_tool(memoria)], rol="operador")
    run_tool(runner, "guardar_recuerdo", GUARDIA)

    segunda = run_tool(runner, "guardar_recuerdo", GUARDIA)

    assert segunda == {"id": "r9", "nota": "Ya estaba en la memoria."}
    assert len(memoria.recuerdos) == 9


def test_con_el_rol_lectura_no_se_guarda(memoria):
    runner = ToolRunner([guardar_recuerdo_tool(memoria)], rol="lectura")

    salida = run_tool(
        runner,
        "guardar_recuerdo",
        {"texto": "Prefiere respuestas cortas.", "tipo": "procedimental"},
    )

    assert salida["error"]["tipo"] == "permiso"
    assert len(memoria.recuerdos) == 8


def test_el_agente_responde_con_lo_que_recuerda(memoria, logger, fake_sleep):
    script = [
        reply(use_tool("consultar_memoria", consulta="T-1043 escalado")),
        reply(answer("Lo escalaste a Guardia N2 el 12 de noviembre [memoria/r3].")),
    ]
    provider = FakeProvider("gemini", script)
    tools = [consultar_memoria_tool(memoria)]
    agent = ReActAgent(LLMClient([provider], logger, sleep=fake_sleep), tools)

    result = agent.run("¿A quién escalé el T-1043?")

    assert result.respuesta == "Lo escalaste a Guardia N2 el 12 de noviembre [memoria/r3]."
    assert "[memoria/r3]" in provider.calls[1]["messages"][-1]["content"]


def test_la_cli_solo_agrega_la_memoria_a_un_agente():
    args = parse_args(["--agent", "react", "--memoria", "--embedder", "fake", "¿Y el T-1043?"])
    assert args.memoria and args.embedder == "fake"
    with pytest.raises(SystemExit):
        parse_args(["--memoria", "¿Y el T-1043?"])
