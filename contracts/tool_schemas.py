"""Contratos de las herramientas: qué recibe cada una y cómo se le describe al modelo."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from pydantic import BaseModel, Field

from contracts.structured_output import sanitize_schema
from tools.tickets import buscar_ticket


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    args_model: type[BaseModel]
    handler: Callable[[BaseModel], dict]

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


BUSCAR_TICKET = ToolSpec(
    name="buscar_ticket",
    description=(
        "Obtiene estado e historial de un ticket por su ID. "
        "Úsala solo si el usuario menciona un ID."
    ),
    args_model=BuscarTicketArgs,
    handler=buscar_ticket,
)

# TODO(clase-3): crea ConsultarProveedorArgs y CONSULTAR_PROVEEDOR.
# 1. Un modelo Pydantic con el argumento que necesita la herramienta.
# 2. Una descripción que diga qué hace, cuándo usarla y cuándo NO.
# 3. Un handler en tools/ que lea data/proveedores.json.
# 4. Agrégala a TOOLS en main.py junto a BUSCAR_TICKET.
