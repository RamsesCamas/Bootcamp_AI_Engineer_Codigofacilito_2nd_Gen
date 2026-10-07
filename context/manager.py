"""Armado del contexto con presupuesto de tokens por sección.

Orden de salida: `system` y `examples` van en el mensaje system; `history`, `documents`
y `user` en el mensaje user, cada uno en su etiqueta, con la pregunta al final.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field

SECTIONS = ("system", "examples", "history", "documents", "user")
RESERVED_SECTIONS = ("output",)  # Se reserva en el presupuesto, pero no se llena.
TRUNCATION_MARK = "[…recortado]"

_TAGS = {"examples": "ejemplos", "history": "historial", "documents": "documentos"}


def estimate_tokens(text: str) -> int:
    """Estima tokens como `ceil(caracteres / 4)`.

    Es una estimación por caracteres: cada proveedor tokeniza distinto. Sirve para
    repartir el presupuesto; para cobrar vale el `usage` que regresa la API.
    """
    return math.ceil(len(text) / 4)


@dataclass
class ContextReport:
    tokens_by_section: dict[str, int]
    truncated_sections: list[str]
    total_tokens: int


@dataclass
class _Item:
    content: str
    priority: int
    order: int  # Orden de llegada: menor = más antiguo.
    tokens: int = field(default=0)


class ContextManager:
    def __init__(
        self, budget: dict[str, int], count_tokens: Callable[[str], int] = estimate_tokens
    ) -> None:
        unknown = set(budget) - set(SECTIONS) - set(RESERVED_SECTIONS)
        if unknown:
            raise ValueError(f"Secciones desconocidas en el presupuesto: {sorted(unknown)}")
        self.budget = dict(budget)
        self.count_tokens = count_tokens
        self._items: dict[str, list[_Item]] = {section: [] for section in SECTIONS}
        self._counter = 0
        self._last_report: ContextReport | None = None

    def add(self, section: str, content: str, priority: int = 0) -> None:
        """Agrega un elemento. A mayor `priority`, más tarde se descarta."""
        if section not in SECTIONS:
            raise ValueError(f"Sección desconocida: {section!r}. Usa una de: {SECTIONS}.")
        if section not in self.budget:
            raise ValueError(f"La sección {section!r} no tiene presupuesto asignado.")
        self._items[section].append(_Item(content, priority, self._counter))
        self._counter += 1

    @property
    def last_report(self) -> ContextReport | None:
        return self._last_report

    def build(self) -> list[dict]:
        """Aplica los presupuestos y devuelve los mensajes `system` y `user`."""
        kept: dict[str, list[str]] = {}
        tokens_by_section: dict[str, int] = {}
        truncated: list[str] = []

        for section in SECTIONS:
            items = self._items[section]
            if not items:
                continue
            for item in items:
                item.tokens = self.count_tokens(item.content)
            budget = self.budget[section]
            if section == "system":
                used = sum(item.tokens for item in items)
                if used > budget:
                    raise ValueError(
                        f"El system ocupa {used} tokens y su presupuesto es {budget}. "
                        "El system nunca se recorta: acórtalo o sube su presupuesto."
                    )
                fitted, was_cut = [item.content for item in items], False
            else:
                fitted, was_cut = self._fit(items, budget)
            kept[section] = fitted
            tokens_by_section[section] = sum(self.count_tokens(text) for text in fitted)
            if was_cut:
                truncated.append(section)

        self._last_report = ContextReport(
            tokens_by_section=tokens_by_section,
            truncated_sections=truncated,
            total_tokens=sum(tokens_by_section.values()),
        )
        return self._messages(kept)

    def _fit(self, items: list[_Item], budget: int) -> tuple[list[str], bool]:
        """Descarta por prioridad (y antigüedad) y, si hace falta, recorta el último."""
        remaining = list(items)
        was_cut = False
        while len(remaining) > 1 and sum(i.tokens for i in remaining) > budget:
            victim = min(remaining, key=lambda i: (i.priority, i.order))
            remaining.remove(victim)
            was_cut = True

        texts = [item.content for item in sorted(remaining, key=lambda i: i.order)]
        if remaining and remaining[0].tokens > budget:  # Queda uno solo y no cabe.
            cut = self._truncate(remaining[0].content, budget)
            texts = [cut] if cut else []
            was_cut = True
        return texts, was_cut

    def _truncate(self, text: str, budget: int) -> str | None:
        """Recorta por el final el prefijo más largo que cabe junto con la marca."""

        def fits(length: int) -> bool:
            candidate = f"{text[:length].rstrip()} {TRUNCATION_MARK}"
            return self.count_tokens(candidate) <= budget

        if not fits(0):
            return None
        low, high = 0, len(text)
        while low < high:
            mid = (low + high + 1) // 2
            if fits(mid):
                low = mid
            else:
                high = mid - 1
        return f"{text[:low].rstrip()} {TRUNCATION_MARK}"

    @staticmethod
    def _messages(kept: dict[str, list[str]]) -> list[dict]:
        system_parts = list(kept.get("system", []))
        if kept.get("examples"):
            system_parts.append(_wrap("examples", kept["examples"]))

        user_parts = [_wrap(s, kept[s]) for s in ("history", "documents") if kept.get(s)]
        user_parts.extend(kept.get("user", []))  # La pregunta va al final.

        messages = []
        if system_parts:
            messages.append({"role": "system", "content": "\n\n".join(system_parts)})
        if user_parts:
            messages.append({"role": "user", "content": "\n\n".join(user_parts)})
        return messages


def _wrap(section: str, texts: list[str]) -> str:
    tag = _TAGS[section]
    body = "\n\n".join(texts)
    return f"<{tag}>\n{body}\n</{tag}>"
