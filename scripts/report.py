"""Reporte de llamadas al LLM a partir de logs/llm_calls.jsonl.

Uso:
    uv run scripts/report.py [ruta/al/archivo.jsonl]
"""

from __future__ import annotations

import json
import statistics
import sys
from collections import Counter
from pathlib import Path

DEFAULT_PATH = Path("logs/llm_calls.jsonl")


def load_events(path: Path) -> list[dict]:
    if not path.exists() or path.stat().st_size == 0:
        sys.exit("No hay llamadas registradas todavía. Corre main.py primero.")
    lines = path.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def percentiles(values: list[float]) -> tuple[float, float]:
    """p50 y p95. quantiles() exige al menos 2 datos."""
    if len(values) == 1:
        return values[0], values[0]
    q = statistics.quantiles(values, n=100, method="inclusive")
    return q[49], q[94]


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    path = Path(argv[0]) if argv else DEFAULT_PATH

    lines = path.read_text(encoding="utf-8").splitlines()
    events = [json.loads(line) for line in lines if line.strip()]
    print(f"Llamadas registradas: {len(events)}")

    events = load_events(path)
    ok = [e for e in events if e["success"]]
    total_cost = sum(e["cost_usd"] for e in events)
    print(f"Eventos en el log:     {len(events)}  ({len(ok)} respuestas)")
    print(f"Costo total:           ${total_cost:.5f}")
    if ok:
        # Latencia y fallback: solo respuestas exitosas.
        p50, p95 = percentiles([e["latency_ms"] for e in ok])
        fallbacks = sum(e["fallback"] for e in ok)
        print(f"Latencia p50 / p95:    {p50:,.0f} ms / {p95:,.0f} ms")
        print(
            f"Fallback:              {fallbacks / len(ok) * 100:.1f} %  ({fallbacks} de {len(ok)})"
        )
    else:
        print("Sin respuestas exitosas: no hay latencia ni fallback que calcular.")

    counts = Counter((e["provider"], e["success"]) for e in events)
    print(f"\n{'Proveedor':<10}  {'éxitos':>6}  {'errores':>7}")
    for provider in sorted({e["provider"] for e in events}):
        print(f"{provider:<10}  {counts[(provider, True)]:>6}  {counts[(provider, False)]:>7}")
    return 0


if __name__ == "__main__":
    main()
