"""Compara versiones de un prompt sobre el mini dataset etiquetado y los tickets de ataque.

Uso:
    uv run scripts/compare_prompts.py ticket_classifier --versions 1 2
    uv run scripts/compare_prompts.py ticket_classifier --versions 2 --no-attack

Usa solo un proveedor (el primario, o el de --provider), sin fallback.
"""

from __future__ import annotations

import argparse
import json
import string
import sys
import unicodedata
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))  # Para importar core/ y prompting/ al correr como script.

from pydantic import ValidationError  # noqa: E402

from core.config import KNOWN_PROVIDERS, Settings  # noqa: E402
from core.errors import AllProvidersFailedError  # noqa: E402
from core.llm_client import LLMClient  # noqa: E402
from prompting import CATEGORIES, PromptKit, PromptNotFoundError  # noqa: E402

COMPANY = "Acme Operaciones"
DEFAULT_DATASET = ROOT / "data" / "tickets_etiquetados.jsonl"
DEFAULT_ATTACK = ROOT / "data" / "tickets_ataque.jsonl"
_STRIP = string.whitespace + string.punctuation + "¿¡«»“”‘’…"
# Una evaluación por lotes choca con el límite por minuto del free tier (HTTP 429).
# Con 6 reintentos el backoff espera hasta ~1 min (1+2+4+8+16+32 s) y la ventana se libera.
BATCH_MAX_RETRIES = 6


# --- Lógica pura (se prueba sin red) -------------------------------------------------


def normalize(text: str) -> str:
    """Minúsculas, sin acentos y sin espacios ni puntuación alrededor."""
    decomposed = unicodedata.normalize("NFKD", text)
    no_accents = "".join(c for c in decomposed if not unicodedata.combining(c))
    return no_accents.lower().strip(_STRIP)


@dataclass
class CallResult:
    ticket_id: str
    expected: str
    raw: str
    tokens_in: int = 0
    tokens_out: int = 0
    cost_usd: float = 0.0
    attack: bool = False
    error: str | None = None  # Falla del proveedor (no es culpa del prompt).

    @property
    def prediction(self) -> str:
        return normalize(self.raw)

    @property
    def in_format(self) -> bool:
        return self.error is None and self.prediction in CATEGORIES

    @property
    def correct(self) -> bool:
        return self.in_format and self.prediction == self.expected


@dataclass
class VersionSummary:
    version: int
    correct: int
    total: int
    out_of_format: int
    avg_tokens: float
    cost_per_1k: float
    attacks_resisted: int
    attacks_total: int
    errors: int = 0


def summarize(version: int, results: list[CallResult]) -> VersionSummary:
    """Agrega los resultados de una versión. Tokens y costo se promedian sobre el dataset."""
    dataset = [r for r in results if not r.attack]
    attacks = [r for r in results if r.attack]
    n = len(dataset)
    return VersionSummary(
        version=version,
        correct=sum(r.correct for r in dataset),
        total=n,
        out_of_format=sum(not r.in_format and r.error is None for r in dataset),
        avg_tokens=sum(r.tokens_in + r.tokens_out for r in dataset) / n if n else 0.0,
        cost_per_1k=1000 * sum(r.cost_usd for r in dataset) / n if n else 0.0,
        attacks_resisted=sum(r.correct for r in attacks),
        attacks_total=len(attacks),
        errors=sum(r.error is not None for r in results),
    )


def format_table(summaries: list[VersionSummary]) -> str:
    headers = [
        "versión",
        "aciertos",
        "fuera de formato",
        "tokens prom.",
        "costo/1k tickets",
        "ataques resistidos",
    ]
    rows = [
        [
            f"v{s.version}",
            f"{s.correct}/{s.total}",
            str(s.out_of_format),
            f"{s.avg_tokens:.0f}",
            f"${s.cost_per_1k:.4f}",
            f"{s.attacks_resisted}/{s.attacks_total}" if s.attacks_total else "-",
        ]
        for s in summaries
    ]
    widths = [max(len(row[i]) for row in [headers, *rows]) for i in range(len(headers))]
    lines = [" | ".join(cell.ljust(w) for cell, w in zip(row, widths, strict=True)) for row in rows]
    header = " | ".join(h.ljust(w) for h, w in zip(headers, widths, strict=True))
    notes = [
        f"Aviso: v{s.version} tuvo {s.errors} llamada(s) con error del proveedor; "
        "cuentan como fallo, no como fuera de formato."
        for s in summaries
        if s.errors
    ]
    return "\n".join([header, "-+-".join("-" * w for w in widths), *lines, *notes])


def describe(result: CallResult) -> str:
    if result.error is not None:
        return f"error del proveedor ({result.error})"
    if result.in_format:
        return result.prediction
    return f"{result.raw.strip()[:30]!r} (fuera de formato)"


def failures(results: dict[int, list[CallResult]]) -> list[str]:
    """Líneas con los tickets que alguna versión falló y la predicción de cada versión."""
    by_ticket: dict[str, dict[int, CallResult]] = {}
    for version, version_results in results.items():
        for r in version_results:
            by_ticket.setdefault(r.ticket_id, {})[version] = r
    lines = []
    for ticket_id, per_version in by_ticket.items():
        if all(r.correct for r in per_version.values()):
            continue
        first = next(iter(per_version.values()))
        label = " [ataque]" if first.attack else ""
        preds = " · ".join(f"v{v}: {describe(r)}" for v, r in per_version.items())
        lines.append(f"{ticket_id}{label} (esperada: {first.expected}) -> {preds}")
    return lines


def load_jsonl(path: Path) -> list[dict]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


# --- Ejecución con el LLM ------------------------------------------------------------


def run_version(
    client: LLMClient, kit: PromptKit, name: str, version: int, tickets: list[dict]
) -> list[CallResult]:
    prompt = kit.get(name, version)
    log_extra = {"prompt_name": prompt.name, "prompt_version": prompt.version}
    results = []
    for ticket in tickets:
        messages = prompt.render(company=COMPANY, ticket=ticket["descripcion"])
        result = CallResult(ticket["id"], ticket["categoria"], "", attack="ataque" in ticket)
        try:
            response = client.generate(messages, log_extra=log_extra, **prompt.model_params)
        except AllProvidersFailedError as e:
            last = e.errors[-1] if e.errors else None
            code = getattr(last, "status_code", None)
            result.error = f"HTTP {code}" if code else type(last).__name__
        else:
            result.raw = response.text
            result.tokens_in = response.tokens_in
            result.tokens_out = response.tokens_out
            result.cost_usd = response.cost_usd
        print("." if result.correct else "x", end="", flush=True, file=sys.stderr)
        results.append(result)
    print(file=sys.stderr)
    return results


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compara versiones de un prompt.")
    parser.add_argument("name", help="Nombre del prompt, por ejemplo ticket_classifier.")
    parser.add_argument("--versions", type=int, nargs="+", help="Versiones (por defecto todas).")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--attack", type=Path, default=DEFAULT_ATTACK)
    parser.add_argument("--no-attack", action="store_true", help="No corre los ataques.")
    parser.add_argument(
        "--provider",
        choices=KNOWN_PROVIDERS,
        help="Proveedor a usar, sin fallback (por defecto PRIMARY del .env).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    kit = PromptKit(ROOT / "prompts")

    try:
        versions = args.versions or kit.versions(args.name)
        for version in versions:
            kit.get(args.name, version)  # Falla antes de gastar llamadas si algo está mal.
        overrides = {"primary": args.provider} if args.provider else {}
        settings = Settings(fallbacks=[], max_retries=BATCH_MAX_RETRIES, **overrides)
        client = LLMClient.from_settings(settings)
    except (PromptNotFoundError, ValueError, NotImplementedError) as e:
        if isinstance(e, ValidationError):
            e = "; ".join(err["msg"].removeprefix("Value error, ") for err in e.errors())
        print(f"Error: {e}", file=sys.stderr)
        return 1

    tickets = load_jsonl(args.dataset)
    if not args.no_attack:
        tickets += load_jsonl(args.attack)

    results = {}
    for version in versions:
        print(f"{args.name} v{version}: {len(tickets)} tickets ", end="", file=sys.stderr)
        results[version] = run_version(client, kit, args.name, version, tickets)

    print()
    print(format_table([summarize(v, r) for v, r in results.items()]))
    failed = failures(results)
    if failed:
        print("\nTickets que alguna versión falló:")
        for line in failed:
            print(f"  {line}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
