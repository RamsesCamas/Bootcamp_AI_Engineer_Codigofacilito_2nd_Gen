from __future__ import annotations

from datetime import datetime

from knowledge.embeddings import FakeEmbedder
from memory.recuerdos import Memoria
from scripts.memoria import ROOT
from scripts.memoria import main as memoria_main

AHORA = datetime(2026, 11, 20, 9, 0)


def memoria(tmp_path) -> Memoria:
    return Memoria(FakeEmbedder(), tmp_path / "memoria.json", reloj=lambda: AHORA)


def test_agregar_actualizar_y_borrar_persisten(tmp_path):
    m = memoria(tmp_path)
    m.agregar("Trabaja en Finanzas.", "semantico", fecha="2026-11-09T10:20")
    m.agregar("Cubre la guardia N2 los martes.", "semantico", sesion="demo")
    m.actualizar("r1", "Trabaja en Operaciones.")
    m.borrar("r2")

    otra = memoria(tmp_path)
    [r1] = otra.recuerdos.values()
    assert (r1.texto, r1.creado, r1.actualizado) == (
        "Trabaja en Operaciones.",
        "2026-11-09T10:20",
        "2026-11-20T09:00",
    )
    assert otra.agregar("Prefiere viñetas.", "procedimental").id == "r3"  # r2 no se reusa


def test_buscar_ordena_por_similitud(tmp_path):
    m = memoria(tmp_path)
    m.agregar("Prefiere las respuestas en viñetas.", "procedimental")
    m.agregar("Escaló el T-1043 a Guardia N2 porque NubeHost no respondía.", "episodico")

    (primero, similitud), (segundo, _) = m.buscar("¿Qué pasó con NubeHost y el T-1043?", k=2)

    assert (primero.id, segundo.id) == ("r2", "r1")
    assert similitud > 0.3


def test_la_recencia_desempata(tmp_path):
    m = memoria(tmp_path)
    # Para el embedder falso, N2 y N3 son la misma bolsa de palabras: empatan en similitud.
    m.agregar("El T-1043 se escaló a Guardia N2.", "episodico", fecha="2026-11-12T16:40")
    m.agregar("El T-1043 se escaló a Guardia N3.", "episodico", fecha="2026-11-19T18:00")

    assert [r.id for r, _ in m.buscar("¿A quién se escaló el T-1043?")] == ["r2", "r1"]


def test_search_regresa_hits_de_la_fuente_memoria(tmp_path):
    m = memoria(tmp_path)
    m.agregar("Va a cerrar el T-1022 cuando se cancele el pedido duplicado.", "episodico")

    [hit] = m.search("¿Qué falta para cerrar el T-1022?", k=5)

    assert (hit.chunk_id, hit.metadatos) == ("r1", {"fuente": "memoria/r1", "tipo": "episodico"})
    assert 0 < hit.distancia < 1


def test_memoria_vacia(tmp_path):
    assert memoria(tmp_path).buscar("¿A quién escalé el T-1043?") == []


def test_cli_importa_una_vez_y_busca(tmp_path, capsys):
    base = ["--embedder", "fake", "--path", str(tmp_path / "memoria.json")]
    demo = str(ROOT / "data" / "memorias_demo.jsonl")

    assert memoria_main([*base, "importar", demo]) == 0
    assert memoria_main([*base, "importar", demo]) == 0
    assert memoria_main([*base, "buscar", "¿Qué me falta para cerrar el T-1022?", "--k", "1"]) == 0
    assert memoria_main([*base, "borrar", "r99"]) == 1

    salida = capsys.readouterr().out.splitlines()
    assert salida[0].startswith("8 recuerdos nuevos · 8 en")
    assert salida[1].startswith("0 recuerdos nuevos · 8 en")
    assert salida[2].endswith(
        "r8 · episodico · 2026-11-17  Va a cerrar el T-1022 cuando se cancele el pedido duplicado."
    )
