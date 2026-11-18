from __future__ import annotations

import json

import pytest

from core.errors import PermanentProviderError
from core.llm_client import LLMClient
from knowledge.router import Router, con_llm, por_reglas
from knowledge.store import Hit
from main import parse_args
from tests.conftest import FakeProvider, make_response


def ruta(fuente: str, motivo: str = "Prueba."):
    return make_response("gemini", json.dumps({"motivo": motivo, "fuente": fuente}))


def client_for(script, logger, fake_sleep):
    provider = FakeProvider("gemini", script)
    return provider, LLMClient([provider], logger, sleep=fake_sleep)


class Fuente:
    """Una fuente falsa: anota cada consulta y regresa un hit con su nombre."""

    def __init__(self, nombre: str) -> None:
        self.nombre = nombre
        self.consultas: list[str] = []

    def search(self, pregunta: str, k: int = 5, where: dict | None = None) -> list[Hit]:
        self.consultas.append(pregunta)
        return [Hit(f"{self.nombre}#0", "texto", {"fuente": self.nombre}, 0.2)]


@pytest.mark.parametrize(
    ("pregunta", "fuente"),
    [
        ("¿Qué me falta para cerrar el T-1022?", "memoria"),
        ("¿Qué penalización aplica si el proveedor del T-1042 no cumple?", "grafo"),
        ("¿Cuándo escalo un ticket de prioridad alta?", "busqueda"),
        # Lo que la regla no ve: en español el sujeto va en el verbo...
        ("¿A quién escalé el T-1043?", "busqueda"),
        # ...y «mi» no siempre pide memoria.
        ("¿Cómo restablezco mi contraseña?", "memoria"),
    ],
)
def test_por_reglas(pregunta, fuente):
    assert por_reglas(pregunta).fuente == fuente


def test_con_llm_solo_describe_las_fuentes_disponibles(logger, fake_sleep):
    provider, client = client_for(
        [ruta("memoria", "Pregunta por algo que hizo.")], logger, fake_sleep
    )

    elegida = con_llm(client, "¿A quién escalé el T-1043?", ("busqueda", "memoria"))

    assert (elegida.fuente, elegida.motivo) == ("memoria", "Pregunta por algo que hizo.")
    system = provider.calls[0]["messages"][0]["content"]
    assert "- memoria: lo que el usuario contó" in system
    assert "- grafo:" not in system


def test_una_fuente_que_no_existe_se_repara(logger, fake_sleep):
    provider, client = client_for([ruta("grafo"), ruta("busqueda")], logger, fake_sleep)

    elegida = con_llm(client, "¿Qué pasó con el T-1016?", ("busqueda", "memoria"))

    assert elegida.fuente == "busqueda"
    reparacion = provider.calls[1]["messages"][-1]["content"]
    assert "fuente debe ser una de ['busqueda', 'memoria']" in reparacion


def test_el_router_busca_solo_en_la_fuente_elegida(logger, fake_sleep):
    fuentes = {nombre: Fuente(nombre) for nombre in ("busqueda", "grafo", "memoria")}
    _, client = client_for([ruta("memoria")], logger, fake_sleep)
    router = Router(fuentes, client)

    [hit] = router.search("¿A quién escalé el T-1043?", k=3)

    assert hit.metadatos["fuente"] == "memoria"
    assert (router.ruta.fuente, router.decidio) == ("memoria", "llm")
    assert fuentes["busqueda"].consultas == fuentes["grafo"].consultas == []


def test_si_el_llm_falla_decide_la_regla(logger, fake_sleep):
    fuentes = {nombre: Fuente(nombre) for nombre in ("busqueda", "grafo")}
    error = PermanentProviderError("gemini", "llave inválida", 401)
    _, client = client_for([error], logger, fake_sleep)
    router = Router(fuentes, client)

    router.search("¿Qué me falta para cerrar el T-1022?")

    # La regla diría memoria, pero esta vez no hay memoria: búsqueda.
    assert (router.ruta.fuente, router.decidio) == ("busqueda", "regla")
    assert router.ruta.motivo == "Habla en primera persona. No hay memoria."
    assert router.respaldos == 1
    assert fuentes["busqueda"].consultas == ["¿Qué me falta para cerrar el T-1022?"]


def test_modo_auto_en_la_cli():
    assert parse_args(["--rag", "¿A quién escalé el T-1043?", "--modo", "auto"]).modo == "auto"
    with pytest.raises(SystemExit):
        parse_args(["--rag", "¿A quién escalé el T-1043?", "--modo", "auto", "--rerank"])
