"""CLI del Agente Operador.

Uso:
    uv run main.py "Resume este ticket: el portal de proveedores no carga desde las 9:00"
    uv run main.py --provider groq "hola"
    uv run main.py --demo
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from pydantic import ValidationError

from core.config import KNOWN_PROVIDERS, Settings, get_settings
from core.errors import AllProvidersFailedError
from core.llm_client import LLMClient

SYSTEM_PROMPT = (
    "Eres un asistente de operaciones de una empresa mediana. "
    "Responde siempre en español, de forma clara y en máximo 3 oraciones."
)

# --demo: solo Gemini, sin fallbacks. Funciona aunque GroqProvider siga con su TODO.
DEMO_TICKET_ID = "T-1043"
TICKETS_PATH = Path(__file__).parent / "data" / "tickets_ejemplo.jsonl"

GRAY = "\033[90m"
RESET = "\033[0m"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Agente Operador: pregunta algo al LLM.")
    parser.add_argument(
        "prompt",
        nargs="?",
        help=f"Texto a enviar al modelo (con --demo, por defecto el ticket {DEMO_TICKET_ID}).",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--provider",
        choices=KNOWN_PROVIDERS,
        help="Fuerza un proveedor y desactiva los fallbacks.",
    )
    mode.add_argument(
        "--demo",
        action="store_true",
        help="Demo de la clase: usa solo Gemini, sin fallbacks (no necesita Groq).",
    )
    args = parser.parse_args(argv)
    if args.prompt is None and not args.demo:
        parser.error("falta el texto a enviar (o usa --demo).")
    return args


def demo_prompt() -> str:
    """Arma el prompt de la demo a partir del ticket de ejemplo."""
    for line in TICKETS_PATH.read_text(encoding="utf-8").splitlines():
        ticket = json.loads(line)
        if ticket["id"] == DEMO_TICKET_ID:
            return f"Resume este ticket: {ticket['asunto']}. {ticket['descripcion']}"
    raise LookupError(f"No encontré el ticket {DEMO_TICKET_ID} en {TICKETS_PATH}")


def fail(message: str) -> int:
    print(f"Error: {message}", file=sys.stderr)
    return 1


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    try:
        if args.demo:
            settings = Settings(primary="gemini", fallbacks=[])
        elif args.provider:
            settings = Settings(primary=args.provider, fallbacks=[])
        else:
            settings = get_settings()
    except ValidationError as e:
        details = "; ".join(err["msg"].removeprefix("Value error, ") for err in e.errors())
        return fail(f"configuración inválida. {details}")

    try:
        client = LLMClient.from_settings(settings)
    except NotImplementedError as e:
        return fail(str(e))

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": args.prompt or demo_prompt()},
    ]
    params: dict = {}

    try:
        response = client.generate(messages, **params)
    except AllProvidersFailedError as e:
        lines = "\n".join(f"  - {type(err).__name__}: {err}" for err in e.errors)
        return fail(f"ningún proveedor pudo responder.\n{lines}")

    print(response.text.strip())
    print(
        f"{GRAY}{response.provider} · {response.model} · "
        f"{response.tokens_in}/{response.tokens_out} tokens · "
        f"{response.latency_ms:.0f} ms · ${response.cost_usd:.6f}{RESET}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
