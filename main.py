"""CLI del Agente Operador.

Uso:
    uv run main.py "Resume este ticket: el portal de proveedores no carga desde las 9:00"
    uv run main.py --provider groq "hola"
    uv run main.py --demo
    uv run main.py --prompt ticket_classifier --ticket T-1099
    uv run main.py --prompt ticket_classifier --prompt-version 1 "No abre el ERP"
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
from prompting import PromptKit, PromptNotFoundError

# TODO(clase-2): migra este system prompt a prompts/operator_assistant/v1.yaml y cárgalo con PromptKit. # noqa: E501
SYSTEM_PROMPT = (
    "Eres un asistente de operaciones de una empresa mediana. "
    "Responde siempre en español, de forma clara y en máximo 3 oraciones."
)

# --demo: solo Gemini, sin fallbacks. Funciona aunque GroqProvider siga con su TODO.
DEMO_TICKET_ID = "T-1043"
TICKETS_PATH = Path(__file__).parent / "data" / "tickets_ejemplo.jsonl"

DATA_DIR = Path(__file__).parent / "data"
PROMPTS_DIR = Path(__file__).parent / "prompts"
# Variables fijas por prompt; `ticket` sale de --ticket o del texto de la línea de comandos.
PROMPT_VARIABLES = {"ticket_classifier": {"company": "Acme Operaciones"}}

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
    parser.add_argument(
        "--prompt",
        dest="prompt_name",
        metavar="NOMBRE",
        help="Usa un prompt versionado de prompts/ (versión activa por defecto).",
    )
    parser.add_argument(
        "--prompt-version", type=int, metavar="N", help="Versión del prompt (requiere --prompt)."
    )
    parser.add_argument(
        "--ticket", metavar="ID", help="Usa la descripción de un ticket de data/*.jsonl."
    )
    args = parser.parse_args(argv)
    if args.prompt is None and not (args.demo or args.ticket):
        parser.error("falta el texto a enviar (o usa --demo o --ticket).")
    if args.prompt_version is not None and not args.prompt_name:
        parser.error("--prompt-version requiere --prompt.")
    return args


def find_ticket(ticket_id: str) -> dict:
    """Busca un ticket por `id` en todos los data/*.jsonl."""
    for path in sorted(DATA_DIR.glob("*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip() and (ticket := json.loads(line)).get("id") == ticket_id:
                return ticket
    raise LookupError(f"No encontré el ticket {ticket_id} en {DATA_DIR}/*.jsonl.")


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


def config_error(e: ValidationError) -> str:
    details = "; ".join(err["msg"].removeprefix("Value error, ") for err in e.errors())
    return f"configuración inválida. {details}"


def build_client(args: argparse.Namespace) -> LLMClient:
    """Arma el cliente. Con --prompt o --ticket, si la cadena de fallbacks no se puede
    construir (por ejemplo, GroqProvider pendiente), sigue solo con el primario."""
    if args.demo:
        return LLMClient.from_settings(Settings(primary="gemini", fallbacks=[]))
    if args.provider:
        return LLMClient.from_settings(Settings(primary=args.provider, fallbacks=[]))
    if not (args.prompt_name or args.ticket):
        return LLMClient.from_settings(get_settings())
    try:
        return LLMClient.from_settings(get_settings())
    except (ValidationError, NotImplementedError) as e:
        reason = config_error(e) if isinstance(e, ValidationError) else str(e)
        settings = Settings(fallbacks=[])
        print(
            f"{GRAY}Aviso: se usa solo {settings.primary}, sin fallback ({reason}){RESET}",
            file=sys.stderr,
        )
        return LLMClient.from_settings(settings)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    try:
        text = find_ticket(args.ticket)["descripcion"] if args.ticket else args.prompt
    except LookupError as e:
        return fail(str(e))

    params: dict = {}
    log_extra: dict = {}
    if args.prompt_name:
        try:
            prompt = PromptKit(PROMPTS_DIR).get(args.prompt_name, args.prompt_version)
            variables = PROMPT_VARIABLES.get(prompt.name, {})
            messages = prompt.render(**variables, ticket=text)
        except (PromptNotFoundError, ValueError) as e:
            return fail(str(e))
        cli_params: dict = {}  # Flags de la CLI (si existen) tienen prioridad.
        params = {**prompt.model_params, **cli_params}
        log_extra = {"prompt_name": prompt.name, "prompt_version": prompt.version}
    else:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": text or demo_prompt()},
        ]

    try:
        client = build_client(args)
    except ValidationError as e:
        return fail(config_error(e))
    except NotImplementedError as e:
        return fail(str(e))

    try:
        response = client.generate(messages, log_extra=log_extra, **params)
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
