"""Contratos de las herramientas: qué recibe cada una y cómo se le describe al modelo."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, Field

from contracts.structured_output import sanitize_schema
from tools.proveedores import consultar_proveedor
from tools.tickets import (
    actualizar_estado,
    buscar_ticket,
    crear_ticket,
    listar_tickets_por_proveedor,
)

ErrorTipo = Literal["validacion", "negocio", "transitorio", "permanente", "permiso", "limite"]
EstadoTicket = Literal["abierto", "en revisión", "esperando proveedor", "resuelto"]
Prioridad = Literal["baja", "media", "alta"]


def tool_error(
    tipo: ErrorTipo, mensaje: str, *, reintentable: bool = False, sugerencia: str | None = None
) -> dict:
    """Error de herramienta con forma fija, para que el modelo sepa qué hacer después.

    `reintentable` dice si repetir la misma llamada podría funcionar; `sugerencia`, qué
    probar en su lugar.
    """
    return {
        "error": {
            "tipo": tipo,
            "mensaje": mensaje,
            "reintentable": reintentable,
            "sugerencia": sugerencia,
        }
    }


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    args_model: type[BaseModel]
    handler: Callable[..., dict]
    # Clase 5 (reto): el handler recibe además el `request_id` de quien llama. Lo pasa el
    # ToolRunner; nunca sale de los argumentos del modelo.
    needs_request_id: bool = False

    def to_openai(self) -> dict:
        """Definición de la herramienta en el formato `tools` de la API compatible con OpenAI.

        `parameters` sale del modelo Pydantic, sin `pattern`, `maxLength`, etc. (ver
        `sanitize_schema`): esas reglas se validan localmente en `execute_tool`.
        """
        parameters = sanitize_schema(self.args_model.model_json_schema(), strict=False)
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": parameters,
            },
        }


class BuscarTicketArgs(BaseModel):
    ticket_id: str = Field(pattern=r"^T-\d{4}$")


class ActualizarEstadoArgs(BaseModel):
    ticket_id: str = Field(pattern=r"^T-\d{4}$")
    estado: EstadoTicket


class ConsultarProveedorArgs(BaseModel):
    nombre: str = Field(
        min_length=2, description="Nombre del proveedor, tal como aparece en el ticket"
    )


class ListarTicketsPorProveedorArgs(BaseModel):
    nombre: str = Field(min_length=2, description="Nombre del proveedor, por ejemplo PagosRed")


class CrearTicketArgs(BaseModel):
    asunto: str = Field(min_length=5, max_length=120, description="Resumen en una línea")
    descripcion: str = Field(min_length=10, description="Qué pasa, desde cuándo y a quién afecta")
    prioridad: Prioridad


class BuscarDocumentosArgs(BaseModel):
    consulta: str = Field(
        min_length=3,
        description=(
            "Qué buscar, con las palabras del documento: IDs de ticket, proveedores, "
            "cláusulas, nombres de guardias"
        ),
    )
    tipo: Literal["runbook", "politica", "contrato", "faq"] | None = Field(
        default=None,
        description="Solo si la pregunta pide un tipo de documento, por ejemplo contrato",
    )
    proveedor: str | None = Field(
        default=None,
        description="Solo para contratos de un proveedor: PagosRed, NubeHost o Transportes Sierra",
    )


class ConsultarMemoriaArgs(BaseModel):
    consulta: str = Field(
        min_length=3,
        description=(
            "Qué recordar, con las palabras del usuario: un ticket, un acuerdo, una preferencia"
        ),
    )


class ConsultarGrafoArgs(BaseModel):
    entidad: str = Field(
        min_length=2,
        description=(
            "La entidad por la que preguntas: un ID de ticket (T-1043), un proveedor "
            "(NubeHost), un equipo (Finanzas), una guardia (N2) o un servicio"
        ),
    )
    hasta: str | None = Field(
        default=None,
        description="Otra entidad, solo si la pregunta es qué conecta a las dos",
    )


BUSCAR_TICKET = ToolSpec(
    name="buscar_ticket",
    description=(
        "Obtiene estado e historial de un ticket por su ID. "
        "Úsala solo si el usuario menciona un ID."
    ),
    args_model=BuscarTicketArgs,
    handler=buscar_ticket,
)

CONSULTAR_PROVEEDOR = ToolSpec(
    name="consultar_proveedor",
    description=(
        "Obtiene servicio, SLA en horas, estado y contacto de un proveedor por su nombre. "
        "Úsala solo con un nombre que venga de un ticket (buscar_ticket) o que el usuario "
        "haya dado. No adivines nombres de proveedores."
    ),
    args_model=ConsultarProveedorArgs,
    handler=consultar_proveedor,
)

ACTUALIZAR_ESTADO = ToolSpec(
    name="actualizar_estado",
    description=(
        "Fija el estado de un ticket por su ID (abierto, en revisión, esperando proveedor o "
        "resuelto). Fija el valor indicado, no lo alterna: repetir la llamada no cambia nada. "
        "No la uses si el usuario no pidió un cambio de estado explícito."
    ),
    args_model=ActualizarEstadoArgs,
    handler=actualizar_estado,
)

LISTAR_TICKETS_POR_PROVEEDOR = ToolSpec(
    name="listar_tickets_por_proveedor",
    description=(
        "Lista los tickets de un proveedor por su nombre: id, asunto y estado, máximo 10, "
        "más el total encontrado. Solo lectura. Úsala cuando pregunten qué tickets tiene un "
        "proveedor; para el detalle de un ticket usa buscar_ticket."
    ),
    args_model=ListarTicketsPorProveedorArgs,
    handler=listar_tickets_por_proveedor,
)

CREAR_TICKET = ToolSpec(
    name="crear_ticket",
    description=(
        "Crea un ticket nuevo con asunto, descripción y prioridad (baja, media o alta) y "
        "regresa su ID. Úsala solo si el usuario pide crear o levantar un ticket. Si la "
        "repites con el mismo asunto y descripción en la misma conversación, no se crea otro: "
        "regresa el mismo ticket con duplicado_evitado = true."
    ),
    args_model=CrearTicketArgs,
    handler=crear_ticket,
    needs_request_id=True,
)
