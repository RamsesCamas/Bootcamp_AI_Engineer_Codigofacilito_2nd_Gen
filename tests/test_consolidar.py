from __future__ import annotations

import json
from datetime import datetime

import pytest

import scripts.memoria
from core.llm_client import LLMClient
from knowledge.embeddings import FakeEmbedder
from memory.consolidar import consolidar
from memory.recuerdos import Memoria
from memory.session import Sesion
from tests.conftest import FakeProvider, make_response

AHORA = datetime(2026, 11, 20, 9, 0)


def reply(**payload):
    return make_response("gemini", json.dumps(payload, ensure_ascii=False))


def hechos(*pares: tuple[str, str]):
    return reply(hechos=[{"texto": texto, "tipo": tipo} for texto, tipo in pares])


def decision(operacion: str, id: str | None = None, texto: str | None = None):
    return reply(motivo="Prueba.", operacion=operacion, id=id, texto=texto)


@pytest.fixture
def memoria(tmp_path) -> Memoria:
    return Memoria(FakeEmbedder(), tmp_path / "memoria.json", reloj=lambda: AHORA)


@pytest.fixture
def sesion() -> Sesion:
    sesion = Sesion("demo")
    sesion.agregar("usuario", "Desde esta semana cubro la guardia N2 los jueves.")
    sesion.agregar("agente", "Anotado.")
    return sesion


def client_for(script, logger, fake_sleep):
    provider = FakeProvider("gemini", script)
    return provider, LLMClient([provider], logger, sleep=fake_sleep)


def test_con_la_memoria_vacia_agrega_sin_preguntar(memoria, sesion, logger, fake_sleep):
    script = [hechos(("Cubre la guardia N2 los jueves.", "semantico"))]
    provider, client = client_for(script, logger, fake_sleep)

    cambios = consolidar(client, memoria, sesion)

    assert cambios == ["agregar r1: Cubre la guardia N2 los jueves. (La memoria está vacía.)"]
    assert memoria.recuerdos["r1"].sesion == "demo"
    assert len(provider.calls) == 1  # solo la extracción
    enviado = provider.calls[0]["messages"]
    assert "Hoy es 2026-11-20." in enviado[0]["content"]
    assert "usuario: Desde esta semana cubro la guardia N2 los jueves." in enviado[-1]["content"]


def test_actualiza_el_recuerdo_que_cambio(memoria, sesion, logger, fake_sleep):
    memoria.agregar("Cubre la guardia N2 los martes.", "semantico", fecha="2026-11-09T10:20")
    memoria.agregar("Prefiere respuestas en viñetas.", "procedimental")
    script = [
        hechos(("Cubre la guardia N2 los jueves.", "semantico")),
        decision("actualizar", "r1", "Cubre la guardia N2 los jueves (antes, los martes)."),
    ]
    provider, client = client_for(script, logger, fake_sleep)

    consolidar(client, memoria, sesion)

    r1 = memoria.recuerdos["r1"]
    assert (r1.texto, r1.creado, r1.actualizado) == (
        "Cubre la guardia N2 los jueves (antes, los martes).",
        "2026-11-09T10:20",
        "2026-11-20T09:00",
    )
    assert len(memoria.recuerdos) == 2
    vistos = provider.calls[1]["messages"][-1]["content"]
    assert "r1 (semantico): Cubre la guardia N2 los martes." in vistos
    assert "<nuevo>(semantico) Cubre la guardia N2 los jueves.</nuevo>" in vistos


def test_un_id_inventado_se_repara(memoria, sesion, logger, fake_sleep):
    memoria.agregar("Cubre la guardia N2 los martes.", "semantico")
    script = [
        hechos(("Ya no cubre guardias.", "semantico")),
        decision("borrar", "r7"),
        decision("borrar", "r1"),
    ]
    provider, client = client_for(script, logger, fake_sleep)

    [cambio] = consolidar(client, memoria, sesion)

    assert cambio.startswith("borrar r1: Cubre la guardia N2 los martes.")
    assert memoria.recuerdos == {}
    assert "para borrar, id debe ser uno de ['r1']" in provider.calls[2]["messages"][-1]["content"]


def test_nada_y_solo_consolida_turnos_nuevos(memoria, sesion, logger, fake_sleep):
    memoria.agregar("Cubre la guardia N2 los jueves.", "semantico")
    script = [hechos(("Cubre la guardia N2 los jueves.", "semantico")), decision("nada")]
    provider, client = client_for(script, logger, fake_sleep)

    assert consolidar(client, memoria, sesion) == [
        "nada: Cubre la guardia N2 los jueves. (Prueba.)"
    ]
    assert sesion.consolidados == 2
    assert consolidar(client, memoria, sesion) == []
    assert len(provider.calls) == 2


def test_cli_consolida_y_guarda_la_sesion(tmp_path, monkeypatch, logger, fake_sleep, capsys):
    sesiones = tmp_path / "sesiones"
    sesion = Sesion("demo")
    sesion.agregar("usuario", "Desde esta semana cubro la guardia N2 los jueves.")
    sesion.agregar("agente", "Anotado.")
    sesion.guardar(sesiones)
    script = [hechos(("Cubre la guardia N2 los jueves.", "semantico"))]
    _, client = client_for(script, logger, fake_sleep)
    monkeypatch.setattr(scripts.memoria, "build_llm_client", lambda: client)
    base = ["--embedder", "fake", "--path", str(tmp_path / "memoria.json")]

    assert (
        scripts.memoria.main([*base, "consolidar", "--sesion", "demo", "--sesiones", str(sesiones)])
        == 0
    )
    assert (
        scripts.memoria.main([*base, "consolidar", "--sesion", "demo", "--sesiones", str(sesiones)])
        == 0
    )

    salida = capsys.readouterr().out.splitlines()
    assert salida[0].startswith("agregar r1: Cubre la guardia N2 los jueves.")
    assert salida[1] == "La sesión demo no tiene turnos nuevos."
    assert Sesion.cargar("demo", sesiones).consolidados == 2
