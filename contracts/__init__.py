"""Contratos de entrada y salida: salida estructurada, herramientas y su ciclo."""

from contracts.structured_output import StructuredOutputError, generate_structured
from contracts.ticket import TicketClassification
from contracts.tool_loop import MAX_ITERS, MaxIterationsError, execute_tool, run_tool_loop
from contracts.tool_schemas import BUSCAR_TICKET, BuscarTicketArgs, ToolSpec

__all__ = [
    "BUSCAR_TICKET",
    "MAX_ITERS",
    "BuscarTicketArgs",
    "MaxIterationsError",
    "StructuredOutputError",
    "TicketClassification",
    "ToolSpec",
    "execute_tool",
    "generate_structured",
    "run_tool_loop",
]
