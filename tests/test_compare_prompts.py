from __future__ import annotations

import json
from collections import Counter
from difflib import SequenceMatcher
from itertools import combinations
from pathlib import Path

import pytest

from prompting import CATEGORIES
from scripts.compare_prompts import CallResult, failures, format_table, normalize, summarize

ROOT = Path(__file__).resolve().parents[1]
LABELED = ROOT / "data" / "tickets_etiquetados.jsonl"
ATTACKS = ROOT / "data" / "tickets_ataque.jsonl"
EXAMPLES = ROOT / "prompts" / "ticket_classifier" / "examples.jsonl"
TICKETS_DB = ROOT / "data" / "tickets_db.json"


def load(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Facturación.", "facturacion"),
        ("  FALLA\n", "falla"),
        ('"proveedor"', "proveedor"),
        ("¿otro?", "otro"),
        ("Categoría: falla", "categoria: falla"),
    ],
)
def test_normalize(raw, expected):
    assert normalize(raw) == expected


def test_invalid_prediction_is_out_of_format_and_wrong():
    result = CallResult("T-1", "facturacion", "urgente")
    assert not result.in_format
    assert not result.correct
    assert CallResult("T-1", "facturacion", "Facturación").correct


def test_summarize_aggregates_hits_tokens_cost_and_attacks():
    results = [
        CallResult("T-1", "falla", "falla", tokens_in=100, tokens_out=2, cost_usd=0.0001),
        CallResult("T-2", "otro", "solicitud", tokens_in=110, tokens_out=2, cost_usd=0.0001),
        CallResult("T-3", "otro", "no sé", tokens_in=90, tokens_out=8, cost_usd=0.0004),
        CallResult("T-9", "facturacion", "urgente", tokens_in=500, attack=True),
        CallResult("T-8", "falla", "falla", tokens_in=500, attack=True),
    ]
    s = summarize(2, results)
    assert (s.correct, s.total, s.out_of_format) == (1, 3, 1)
    assert s.avg_tokens == pytest.approx(104)  # (102 + 112 + 98) / 3; ataques no cuentan
    assert s.cost_per_1k == pytest.approx(0.2)  # 0.0006 / 3 * 1000
    assert (s.attacks_resisted, s.attacks_total) == (1, 2)
    table = format_table([s])
    assert "1/3" in table and "1/2" in table and "costo/1k tickets" in table


def test_failures_lists_tickets_missed_by_any_version():
    results = {
        1: [CallResult("T-1", "falla", "falla"), CallResult("T-2", "otro", "falla")],
        2: [CallResult("T-1", "falla", "falla"), CallResult("T-2", "otro", "otro")],
    }
    lines = failures(results)
    assert len(lines) == 1
    assert lines[0].startswith("T-2") and "v1: falla" in lines[0] and "v2: otro" in lines[0]


def test_labeled_dataset_has_four_per_category():
    tickets = load(LABELED)
    assert len(tickets) == 20
    assert Counter(t["categoria"] for t in tickets) == {c: 4 for c in CATEGORIES}
    assert [t["id"] for t in tickets] == [f"T-{n}" for n in range(2001, 2021)]


def test_all_labels_are_valid_categories():
    for path in (LABELED, ATTACKS):
        assert {t["categoria"] for t in load(path)} <= set(CATEGORIES)
    assert {e["categoria"] for e in load(EXAMPLES)} <= set(CATEGORIES)
    assert all(t["ataque"] for t in load(ATTACKS))
    assert "T-1099" in {t["id"] for t in load(ATTACKS)}


def test_no_evaluation_leak_between_examples_and_datasets():
    """Los ejemplos few-shot no pueden repetir ni parafrasear tickets de evaluación."""
    texts = {
        "examples": [e["ticket"] for e in load(EXAMPLES)],
        "etiquetados": [t["descripcion"] for t in load(LABELED)],
        "ataque": [t["descripcion"] for t in load(ATTACKS)],
        "tickets_db": [t["descripcion"] for t in json.loads(TICKETS_DB.read_text("utf-8"))],
    }
    for (name_a, group_a), (name_b, group_b) in combinations(texts.items(), 2):
        for a in group_a:
            for b in group_b:
                assert normalize(a) != normalize(b), f"Texto repetido en {name_a} y {name_b}: {a}"
                ratio = SequenceMatcher(None, normalize(a), normalize(b)).ratio()
                assert ratio < 0.7, f"Texto casi igual en {name_a} y {name_b}: {a!r} ~ {b!r}"


def test_provider_error_is_a_failure_but_not_out_of_format():
    results = [
        CallResult("T-1", "falla", "falla"),
        CallResult("T-2", "otro", "", error="HTTP 429"),
        CallResult("T-3", "otro", "no sé"),
    ]
    s = summarize(1, results)
    assert (s.correct, s.out_of_format, s.errors) == (1, 1, 1)
    assert "error del proveedor" in format_table([s])
    assert "HTTP 429" in failures({1: results})[0]
