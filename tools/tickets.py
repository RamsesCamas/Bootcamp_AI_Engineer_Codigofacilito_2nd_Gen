"""Herramienta `buscar_ticket` sobre data/tickets_db.json."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from contracts.tool_schemas import BuscarTicketArgs

# Ruta fija: nunca se construye a partir de los argumentos del modelo.
TICKETS_DB = Path(__file__).resolve().parents[1] / "data" / "tickets_db.json"
HISTORY_LIMIT = 3


@lru_cache(maxsize=1)
def _tickets() -> dict[str, dict]:
    tickets = json.loads(TICKETS_DB.read_text(encoding="utf-8"))
    return {ticket["id"]: ticket for ticket in tickets}


def buscar_ticket(args: BuscarTicketArgs) -> dict:
    """Estado, equipo, proveedor y últimas entradas del historial de un ticket."""
    ticket = _tickets().get(args.ticket_id)
    if ticket is None:
        return {"error": f"ticket {args.ticket_id} no existe"}
    return {
        "id": ticket["id"],
        "asunto": ticket["asunto"],
        "estado": ticket["estado"],
        "equipo": ticket["equipo"],
        "prioridad": ticket["prioridad"],
        "proveedor": ticket["proveedor"],
        "historial": ticket["historial"][-HISTORY_LIMIT:],
    }
