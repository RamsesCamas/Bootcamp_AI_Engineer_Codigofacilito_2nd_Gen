"""Contratos de entrada y salida: salida estructurada, herramientas y su ciclo."""

from contracts.structured_output import StructuredOutputError, generate_structured
from contracts.ticket import TicketClassification

__all__ = ["StructuredOutputError", "TicketClassification", "generate_structured"]
