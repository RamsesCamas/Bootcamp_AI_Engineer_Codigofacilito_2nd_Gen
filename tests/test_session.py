from __future__ import annotations

import pytest

from agents import ReActAgent
from contracts import BUSCAR_TICKET
from core.llm_client import LLMClient
from main import parse_args, run_agent
from memory.session import RECIENTES, Sesion
from tests.agent_helpers import answer, reply
from tests.conftest import FakeProvider, make_response


def client_for(script, logger, fake_sleep):
    provider = FakeProvider("gemini", script)
    return provider, LLMClient([provider], logger, sleep=fake_sleep)


def larga(n: int) -> Sesion:
    """Una sesión de n preguntas con sus respuestas: 2n turnos, más de 300 tokens."""
    sesion = Sesion("demo")
    for i in range(n):
        sesion.agregar("usuario", f"Pregunta {i} sobre el T-104{i}. " + "Contexto. " * 30)
        sesion.agregar("agente", f"Respuesta {i}.")
    return sesion


def test_historial_lleva_el_resumen_y_los_turnos_vigentes():
    sesion = Sesion("demo", resumen="Escaló el T-1043 a Guardia N2.")
    for rol, texto in [
        ("usuario", "¿Cómo va el T-1043?"),
        ("agente", "Abierto."),
        ("usuario", "¿Y quién lo atiende?"),
        ("agente", "NubeHost."),
    ]:
        sesion.agregar(rol, texto)
    sesion.resumidos = 2

    assert sesion.historial() == (
        "Resumen de lo anterior: Escaló el T-1043 a Guardia N2.\n"
        "usuario: ¿Y quién lo atiende?\n"
        "agente: NubeHost."
    )


def test_guardar_y_cargar(tmp_path):
    sesion = larga(1)
    sesion.resumen, sesion.resumidos = "Nada aún.", 1
    path = sesion.guardar(tmp_path)

    assert path == tmp_path / "demo.json"
    assert Sesion.cargar("demo", tmp_path) == sesion
    assert Sesion.cargar("otra", tmp_path) == Sesion("otra")


@pytest.mark.parametrize("sesion_id", ["../secretos", "a/b", "", "x" * 41])
def test_id_invalido(tmp_path, sesion_id):
    with pytest.raises(ValueError, match="id de sesión inválido"):
        Sesion.cargar(sesion_id, tmp_path)


def test_no_resume_si_el_historial_cabe(logger, fake_sleep):
    provider, client = client_for([], logger, fake_sleep)
    sesion = Sesion("demo")
    sesion.agregar("usuario", "¿Cómo va el T-1043?")
    sesion.agregar("agente", "Abierto.")

    assert sesion.compactar(client) == 0
    assert provider.calls == []


def test_compactar_resume_lo_viejo_y_deja_los_recientes(logger, fake_sleep):
    provider, client = client_for(
        [make_response("gemini", " Preguntó por el T-1040 y el T-1041. ")], logger, fake_sleep
    )
    sesion = larga(4)

    assert sesion.compactar(client) == 8 - RECIENTES
    assert sesion.resumidos == 4
    assert sesion.resumen == "Preguntó por el T-1040 y el T-1041."
    assert len(sesion.vigentes) == RECIENTES
    assert len(sesion.turnos) == 8  # el archivo conserva todo
    enviado = provider.calls[0]["messages"][-1]["content"]
    assert "<resumen_anterior>(ninguno)</resumen_anterior>" in enviado
    assert "Pregunta 1" in enviado and "Pregunta 2" not in enviado


def test_el_resumen_anterior_entra_al_siguiente(logger, fake_sleep):
    script = [make_response("gemini", "R1"), make_response("gemini", "R2")]
    provider, client = client_for(script, logger, fake_sleep)
    sesion = larga(4)
    sesion.compactar(client)
    for i in range(4, 7):
        sesion.agregar("usuario", f"Pregunta {i}. " + "Contexto. " * 30)
        sesion.agregar("agente", f"Respuesta {i}.")

    assert sesion.compactar(client) == 6
    assert "<resumen_anterior>R1</resumen_anterior>" in provider.calls[1]["messages"][-1]["content"]
    assert (sesion.resumen, sesion.resumidos) == ("R2", 14 - RECIENTES)


def test_el_agente_recibe_el_historial(logger, fake_sleep):
    provider = FakeProvider("gemini", [reply(answer("Sigue abierto."))])
    agent = ReActAgent(LLMClient([provider], logger, sleep=fake_sleep), [BUSCAR_TICKET])

    agent.run("¿Y ese ticket?", historial="usuario: ¿Cómo va el T-1043?\nagente: Abierto.")

    enviado = provider.calls[0]["messages"][-1]["content"]
    assert enviado.startswith("<historial>\nusuario: ¿Cómo va el T-1043?\nagente: Abierto.\n")


def test_la_cli_guarda_cada_turno(logger, fake_sleep, tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)  # var/sesiones/ queda en tmp_path
    script = [reply(answer("El T-1043 sigue abierto.")), reply(answer("Lo atiende NubeHost."))]
    provider, client = client_for(script, logger, fake_sleep)

    run_agent(client, "react", "¿Cómo va el T-1043?", verbose=False, sesion="demo")
    run_agent(client, "react", "¿Y quién lo atiende?", verbose=False, sesion="demo")

    assert (
        "<historial>\nusuario: ¿Cómo va el T-1043?" in provider.calls[1]["messages"][-1]["content"]
    )
    sesion = Sesion.cargar("demo")
    assert [t.texto for t in sesion.turnos][-1] == "Lo atiende NubeHost."
    assert "sesión demo: 2 turnos" in capsys.readouterr().out


def test_sesion_requiere_react():
    assert parse_args(["--agent", "react", "--sesion", "demo", "¿Y el T-1043?"]).sesion == "demo"
    with pytest.raises(SystemExit):
        parse_args(["--agent", "plan", "--sesion", "demo", "¿Y el T-1043?"])
