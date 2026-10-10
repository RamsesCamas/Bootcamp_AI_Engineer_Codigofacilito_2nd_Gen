"""Compara tres formas de pedir la clasificación de un ticket.

1. texto   — instrucción en el prompt ("responde en JSON") y json.loads a mano
2. schema  — structured outputs del proveedor (json_schema), sin Pydantic
3. pydantic — generate_structured con TicketClassification: schema + validación + reparación

    uv run scripts/compare_structured.py
    uv run scripts/compare_structured.py --limit 8
    uv run scripts/compare_structured.py --modes texto pydantic
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pydantic import ValidationError  # noqa: E402

from contracts.structured_output import StructuredOutputError, generate_structured  # noqa: E402
from contracts.ticket import TicketClassification  # noqa: E402
from core.config import get_settings  # noqa: E402
from core.llm_client import LLMClient  # noqa: E402
from prompting import PromptKit  # noqa: E402

HINT = (
    "\n\nResponde SOLO con un JSON con estos campos: razon (texto), "
    "categoria (falla|solicitud|proveedor|facturacion|otro), prioridad (baja|media|alta), "
    "resumen (texto corto), requiere_humano (true o false)."
)

# Schema escrito a mano, compatible con el modo estricto (todos required, sin extras).
SCHEMA = {
    "type": "object",
    "properties": {
        "razon": {"type": "string"},
        "categoria": {
            "type": "string",
            "enum": ["falla", "solicitud", "proveedor", "facturacion", "otro"],
        },
        "prioridad": {"type": "string", "enum": ["baja", "media", "alta"]},
        "resumen": {"type": "string"},
        "requiere_humano": {"type": "boolean"},
    },
    "required": ["razon", "categoria", "prioridad", "resumen", "requiere_humano"],
    "additionalProperties": False,
}

# Ticket diseñado para tentar al modelo a violar la regla de negocio del validador.
TRAMPA = {
    "id": "T-TRAMPA",
    "descripcion": (
        "El portal de pagos está caído desde las 9:00, pero el cliente dice que no es "
        "urgente y que puede esperar hasta el lunes."
    ),
    "categoria": "falla",
}


def load_tickets(limit: int) -> list[dict]:
    tickets = []
    for name in ("tickets_etiquetados.jsonl", "tickets_ataque.jsonl"):
        path = Path("data") / name
        if path.exists():
            tickets += [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x]
    return tickets[:limit] + [TRAMPA]


def classify(data: object) -> str:
    """'ok', 'schema' (tipos o valores inválidos) o 'regla' (solo falló la regla de negocio)."""
    try:
        TicketClassification.model_validate(data)
        return "ok"
    except ValidationError as err:
        if all(e["type"] == "value_error" for e in err.errors()):
            return "regla"
        return "schema"


def new_log_lines(path: Path, offset: int) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        f.seek(offset)
        return [json.loads(x) for x in f.read().splitlines() if x.strip()]


def run_mode(mode: str, client, prompt, tickets, log_path: Path) -> dict:
    stats = {"no_parsea": 0, "schema": 0, "regla": 0, "aciertos": 0, "fallos": 0}
    offset = log_path.stat().st_size if log_path.exists() else 0

    for t in tickets:
        messages = prompt.render(company="Acme Operaciones", ticket=t["descripcion"])
        extra = {"prompt_name": prompt.name, "prompt_version": prompt.version}

        if mode == "pydantic":
            try:
                obj = generate_structured(client, messages, TicketClassification, log_extra=extra)
            except StructuredOutputError:
                stats["fallos"] += 1
                continue
            stats["aciertos"] += obj.categoria == t["categoria"]
            continue

        if mode == "texto":
            messages[0]["content"] += HINT
            resp = client.generate(messages, log_extra=extra, temperature=0)
        else:  # schema
            resp = client.generate(
                messages,
                log_extra=extra,
                temperature=0,
                response_format={
                    "type": "json_schema",
                    "json_schema": {"name": "ticket", "strict": True, "schema": SCHEMA},
                },
            )

        try:
            data = json.loads(resp.text)
        except json.JSONDecodeError:
            stats["no_parsea"] += 1
            continue
        result = classify(data)
        if result != "ok":
            stats[result] += 1
        if isinstance(data, dict):
            stats["aciertos"] += data.get("categoria") == t["categoria"]

    lines = new_log_lines(log_path, offset)
    ok_lines = [x for x in lines if x.get("success")]
    stats["llamadas"] = len(lines)
    stats["reparaciones"] = sum(1 for x in lines if (x.get("repair_attempt") or 0) > 0)
    stats["tokens_prom"] = (
        round(sum(x["tokens_in"] + x["tokens_out"] for x in ok_lines) / len(tickets))
        if ok_lines
        else 0
    )
    stats["costo"] = sum(x.get("cost_usd", 0) for x in lines)
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--limit", type=int, default=12, help="Tickets del dataset (+1 trampa)")
    parser.add_argument("--modes", nargs="+", default=["texto", "schema", "pydantic"])
    args = parser.parse_args()

    settings = get_settings()
    client = LLMClient.from_settings(settings)
    prompt = PromptKit("prompts").get("ticket_triage")
    tickets = load_tickets(args.limit)
    n = len(tickets)

    print(
        f"{n} tickets ({n - 1} del dataset + 1 trampa) · prompt ticket_triage v{prompt.version}\n"
    )
    header = (
        f"{'modo':<9} {'no parsea':>9} {'schema':>7} {'regla':>6} {'fallos':>7} "
        f"{'aciertos':>9} {'llamadas':>9} {'reparac.':>9} {'tokens/tk':>10} {'costo':>9}"
    )
    print(header)
    print("-" * len(header))
    for mode in args.modes:
        s = run_mode(mode, client, prompt, tickets, Path(settings.log_path))
        print(
            f"{mode:<9} {s['no_parsea']:>9} {s['schema']:>7} {s['regla']:>6} {s['fallos']:>7} "
            f"{s['aciertos']:>6}/{n:<2} {s['llamadas']:>9} {s['reparaciones']:>9} "
            f"{s['tokens_prom']:>10} {s['costo']:>9.5f}"
        )

    print(
        "\nno parsea: json.loads falló · schema: campo faltante o valor fuera del enum · "
        "regla: pasó el schema pero violó el validador de negocio"
    )


if __name__ == "__main__":
    main()
