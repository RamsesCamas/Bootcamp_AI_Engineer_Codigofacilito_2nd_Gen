"""Demo del ContextManager: historial largo con presupuesto pequeño. No llama a ningún LLM.

Uso:
    uv run scripts/context_demo.py
    uv run scripts/context_demo.py --budget-history 400
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))  # Para importar context/ y prompting/ al correr como script.

from context import ContextManager  # noqa: E402
from prompting import PromptKit  # noqa: E402

BUDGET = {"system": 300, "history": 250, "documents": 120, "user": 80, "output": 100}
TURNS = 30

DOCUMENTS = [
    (
        2,
        "Procedimiento PRV-07 (portal de proveedores): si el portal no responde, revisar el "
        "estado del balanceador, reiniciar el servicio de autenticación y avisar a compras. "
        "Si la falla dura más de 2 horas, ampliar el periodo de recepción de facturas un día.",
    ),
    (
        0,
        "Bitácora de cambios de septiembre: se actualizó el certificado TLS del portal, se "
        "migró la base de datos de cotizaciones a un servidor nuevo, se cambió el proveedor "
        "de correo transaccional y se ajustaron las reglas del firewall para la red de "
        "invitados. También se programó mantenimiento del aire acondicionado del site.",
    ),
]
QUESTION = "¿Qué hago si el portal de proveedores sigue sin cargar después de las 11:00?"


def history() -> list[tuple[int, str]]:
    """30 turnos ficticios. Las decisiones tienen prioridad 2 y los datos clave 1."""
    turns = []
    for i in range(1, TURNS + 1):
        if i % 10 == 0:
            priority, text = 2, "decisión: se escala el caso al equipo de infraestructura."
        elif i % 4 == 0:
            priority, text = 1, "dato: el error es 503 y empezó a las 9:00."
        else:
            priority, text = 0, "seguimiento: el usuario confirma que sigue igual, sin cambios."
        role = "usuario" if i % 2 else "agente"
        turns.append((priority, f"[turno {i:02d}] {role}: {text}"))
    return turns


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Muestra el recorte del ContextManager.")
    parser.add_argument(
        "--budget-history",
        type=int,
        default=BUDGET["history"],
        help=f"Presupuesto de tokens del historial (por defecto {BUDGET['history']}).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    budget = {**BUDGET, "history": args.budget_history}

    system = PromptKit(ROOT / "prompts").get("ticket_classifier", 2)
    system_text = system.render(company="Acme Operaciones", ticket="")[0]["content"]

    cm = ContextManager(budget)
    cm.add("system", system_text)
    for priority, turn in history():
        cm.add("history", turn, priority=priority)
    for priority, document in DOCUMENTS:
        cm.add("documents", document, priority=priority)
    cm.add("user", QUESTION)
    messages = cm.build()
    report = cm.last_report

    print("Presupuesto:", ", ".join(f"{k}={v}" for k, v in budget.items()))
    print()
    print(f"{'sección':<10} {'tokens':>6} {'presupuesto':>11}  recortada")
    for section, tokens in report.tokens_by_section.items():
        mark = "sí" if section in report.truncated_sections else ""
        print(f"{section:<10} {tokens:>6} {budget[section]:>11}  {mark}")
    print(f"{'total':<10} {report.total_tokens:>6} {'':>11}  (+{budget['output']} de salida)")

    user_message = messages[-1]["content"]
    survivors = re.findall(r"\[turno (\d+)\]", user_message)
    print()
    print(f"Turnos que sobrevivieron: {len(survivors)}/{TURNS} -> {', '.join(survivors)}")
    print(f"Secciones recortadas: {', '.join(report.truncated_sections) or 'ninguna'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
