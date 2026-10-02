"""CLI del Agente Operador.

Uso:
    uv run main.py "Resume este ticket: el portal de proveedores no carga desde las 9:00"
    uv run main.py --provider groq "hola"
"""

from __future__ import annotations

import argparse
import sys

from pydantic import ValidationError

from core.config import KNOWN_PROVIDERS, Settings, get_settings
from core.errors import AllProvidersFailedError
from core.llm_client import LLMClient

SYSTEM_PROMPT = (
    "Eres un asistente de operaciones de una empresa mediana. "
    "Responde siempre en español, de forma clara y en máximo 3 oraciones."
)

GRAY = "\033[90m"
RESET = "\033[0m"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Agente Operador: pregunta algo al LLM.")
    parser.add_argument("prompt", help="Texto que quieres enviar al modelo.")
    parser.add_argument(
        "--provider",
        choices=KNOWN_PROVIDERS,
        help="Fuerza un proveedor y desactiva los fallbacks.",
    )
    return parser.parse_args(argv)


def fail(message: str) -> int:
    print(f"Error: {message}", file=sys.stderr)
    return 1


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    try:
        if args.provider:
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
        {"role": "user", "content": args.prompt},
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
