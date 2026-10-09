"""Contrato de salida del triage de tickets."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator


class TicketClassification(BaseModel):
    # `razon` va primero a propósito: el modelo escribe en orden, así que justifica antes
    # de decidir la categoría y la prioridad.
    razon: str = Field(description="Justificación breve, máx. 2 oraciones")
    categoria: Literal["falla", "solicitud", "proveedor", "facturacion", "otro"]
    prioridad: Literal["baja", "media", "alta"]
    resumen: str = Field(max_length=280)
    requiere_humano: bool

    # Regla simplificada para la demo: busca "caíd" en el texto. Una regla real usaría un
    # campo explícito (por ejemplo `servicio_caido: bool`) en vez de adivinar por palabras.
    @model_validator(mode="after")
    def caida_es_alta(self):
        texto = (self.razon + self.resumen).lower()
        if "caíd" in texto and self.prioridad != "alta":
            raise ValueError("Una caída debe ser prioridad alta")
        return self
