"""`consultar_memoria`: el agente busca en lo que el usuario contó en otras sesiones (Clase 9).

Con `buscar_documentos`, `consultar_grafo` y `consultar_memoria`, el agente ReAct ya es un
router: en cada paso elige la fuente por la descripción de la herramienta, como el router con
LLM, y puede consultar varias. A cambio, gasta más llamadas que el router, que elige una vez.
"""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING

from contracts.tool_schemas import ConsultarMemoriaArgs, GuardarRecuerdoArgs, ToolSpec

if TYPE_CHECKING:
    from memory.recuerdos import Memoria

LIMITE = 3


def consultar_memoria(args: ConsultarMemoriaArgs, *, memoria: Memoria) -> dict:
    encontrados = memoria.buscar(args.consulta, k=LIMITE)
    if not encontrados:
        return {"recuerdos": [], "nota": "La memoria está vacía."}
    return {
        "recuerdos": [
            f"[memoria/{r.id}] ({r.tipo}, {r.actualizado[:10]}) {r.texto}" for r, _ in encontrados
        ]
    }


def guardar_recuerdo(args: GuardarRecuerdoArgs, *, memoria: Memoria) -> dict:
    """Escribe en el camino de la respuesta: lo que el usuario pidió recordar, al momento."""
    for recuerdo in memoria.recuerdos.values():
        if recuerdo.texto == args.texto:  # el mismo texto no se duplica: se puede reintentar
            return {"id": recuerdo.id, "nota": "Ya estaba en la memoria."}
    recuerdo = memoria.agregar(args.texto, args.tipo)
    return {"id": recuerdo.id, "guardado": recuerdo.texto}


def consultar_memoria_tool(memoria: Memoria) -> ToolSpec:
    """La herramienta lista para el agente, ligada a una memoria ya abierta."""
    return ToolSpec(
        name="consultar_memoria",
        description=(
            "Busca en la memoria de largo plazo lo que el usuario contó en sesiones "
            "anteriores: su equipo y horario, sus acuerdos, lo que hizo o dejó pendiente y "
            f"cómo quiere las respuestas. Regresa los {LIMITE} recuerdos más parecidos, con "
            "fecha; puede que ninguno venga al caso. Es lo que pasó, no el estado actual: si "
            "un dato de ticket importa, confírmalo con buscar_ticket. Cita las fuentes entre "
            "corchetes."
        ),
        args_model=ConsultarMemoriaArgs,
        handler=partial(consultar_memoria, memoria=memoria),
    )


def guardar_recuerdo_tool(memoria: Memoria) -> ToolSpec:
    """Práctica guiada de la Clase 9: escribir en la memoria sin esperar a `consolidar`."""
    return ToolSpec(
        name="guardar_recuerdo",
        description=(
            "Guarda un recuerdo de largo plazo. Úsala solo cuando el usuario te pide que "
            "recuerdes algo («recuerda que…», «anota que…»); lo demás de la sesión lo guarda "
            "la consolidación. Escribe una oración en tercera persona con lo que pidió."
        ),
        args_model=GuardarRecuerdoArgs,
        handler=partial(guardar_recuerdo, memoria=memoria),
    )
