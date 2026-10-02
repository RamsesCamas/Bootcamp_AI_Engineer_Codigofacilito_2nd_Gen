"""Reporte de llamadas al LLM a partir de logs/llm_calls.jsonl.

Uso:
    uv run scripts/report.py [ruta/al/archivo.jsonl]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

DEFAULT_PATH = Path("logs/llm_calls.jsonl")


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    path = Path(argv[0]) if argv else DEFAULT_PATH

    lines = path.read_text(encoding="utf-8").splitlines()
    events = [json.loads(line) for line in lines if line.strip()]
    print(f"Llamadas registradas: {len(events)}")

    # TODO(clase-1): completa el reporte. Debe imprimir:
    #   - costo total en USD,
    #   - latencia p50 y p95 (solo llamadas exitosas; usa statistics.quantiles),
    #   - % de llamadas con fallback,
    #   - número de llamadas por proveedor.
    # Usa solo la librería estándar y maneja con un mensaje el caso de que el archivo
    # no exista o esté vacío.
    return 0


if __name__ == "__main__":
    sys.exit(main())
