from __future__ import annotations

import json
from typing import get_args

import pytest

from contracts import StructuredOutputError, TicketClassification, generate_structured
from contracts.structured_output import sanitize_schema
from core.errors import AllProvidersFailedError, PermanentProviderError
from core.llm_client import LLMClient
from prompting import CATEGORIES
from tests.conftest import FakeProvider, make_response, read_events

MESSAGES = [
    {"role": "system", "content": "Clasifica tickets."},
    {"role": "user", "content": "<ticket>El portal no carga.</ticket>"},
]
VALID = {
    "razon": "El portal está caído y bloquea a los proveedores.",
    "categoria": "falla",
    "prioridad": "alta",
    "resumen": "Portal de proveedores caído desde las 9:00.",
    "requiere_humano": True,
}


def respond(payload) -> object:
    text = payload if isinstance(payload, str) else json.dumps(payload, ensure_ascii=False)
    return make_response("gemini", text=text)


def make_client(script, logger, fake_sleep, *, json_schema=True, json_object=True):
    provider = FakeProvider("gemini", script)
    provider.supports_json_schema = json_schema
    provider.supports_json_object = json_object
    return provider, LLMClient([provider], logger, sleep=fake_sleep)


def test_valid_on_first_try(logger, fake_sleep):
    provider, client = make_client([respond(VALID)], logger, fake_sleep)

    result = generate_structured(client, MESSAGES, TicketClassification)

    assert result == TicketClassification(**VALID)
    assert len(provider.calls) == 1


def test_invalid_json_is_repaired(logger, fake_sleep):
    provider, client = make_client(
        [respond('{"razon": "incompleto"'), respond(VALID)], logger, fake_sleep
    )

    result = generate_structured(client, MESSAGES, TicketClassification)

    assert result.categoria == "falla"
    second_history = provider.calls[1]["messages"]
    assert second_history[-2] == {"role": "assistant", "content": '{"razon": "incompleto"'}
    assert second_history[-1]["role"] == "user"
    assert "no cumple el formato" in second_history[-1]["content"]


def test_semantic_validator_triggers_repair(logger, fake_sleep):
    wrong = {**VALID, "prioridad": "media"}  # "caído" con prioridad media
    provider, client = make_client([respond(wrong), respond(VALID)], logger, fake_sleep)

    result = generate_structured(client, MESSAGES, TicketClassification)

    assert result.prioridad == "alta"
    assert "Una caída debe ser prioridad alta" in provider.calls[1]["messages"][-1]["content"]


def test_repairs_exhausted_raise(logger, fake_sleep):
    bad = respond({**VALID, "categoria": "urgente"})
    provider, client = make_client([bad, bad, bad], logger, fake_sleep)

    with pytest.raises(StructuredOutputError) as exc_info:
        generate_structured(client, MESSAGES, TicketClassification, max_repairs=2)

    assert exc_info.value.attempts == 3
    assert "categoria" in exc_info.value.last_error
    assert len(provider.calls) == 3


def test_permanent_provider_error_is_not_repaired(logger, fake_sleep):
    error = PermanentProviderError("gemini", "schema no soportado", 400)
    provider, client = make_client([error], logger, fake_sleep)

    with pytest.raises(AllProvidersFailedError) as exc_info:
        generate_structured(client, MESSAGES, TicketClassification)

    assert exc_info.value.errors == [error]
    assert len(provider.calls) == 1


def test_strategy_json_schema(logger, fake_sleep):
    provider, client = make_client([respond(VALID)], logger, fake_sleep, json_schema=True)

    generate_structured(client, MESSAGES, TicketClassification)

    params = provider.calls[0]["params"]
    assert params["response_format"]["type"] == "json_schema"
    assert params["response_format"]["json_schema"]["strict"] is True
    assert provider.calls[0]["messages"] == MESSAGES  # sin instrucción extra


def test_strategy_json_object_adds_schema_to_system(logger, fake_sleep):
    provider, client = make_client(
        [respond(VALID)], logger, fake_sleep, json_schema=False, json_object=True
    )

    generate_structured(client, MESSAGES, TicketClassification)

    call = provider.calls[0]
    assert call["params"]["response_format"] == {"type": "json_object"}
    assert "JSON Schema" in call["messages"][0]["content"]
    assert MESSAGES[0]["content"] == "Clasifica tickets."  # el original no cambia


def test_strategy_instruction_only(logger, fake_sleep):
    fenced = respond("```json\n" + json.dumps(VALID) + "\n```")
    provider, client = make_client(
        [fenced], logger, fake_sleep, json_schema=False, json_object=False
    )

    result = generate_structured(client, MESSAGES, TicketClassification)

    assert result.categoria == "falla"
    assert "response_format" not in provider.calls[0]["params"]
    assert "JSON Schema" in provider.calls[0]["messages"][0]["content"]


def test_repair_attempt_and_schema_in_log(logger, log_path, fake_sleep):
    _, client = make_client([respond("no es json"), respond(VALID)], logger, fake_sleep)

    generate_structured(client, MESSAGES, TicketClassification, log_extra={"prompt_version": 1})

    events = read_events(log_path)
    assert [e["repair_attempt"] for e in events] == [0, 1]
    assert all(e["structured_schema"] == "TicketClassification" for e in events)
    assert all(e["prompt_version"] == 1 for e in events)


def test_literal_matches_categories():
    literal = TicketClassification.model_fields["categoria"].annotation
    assert get_args(literal) == CATEGORIES


def test_razon_is_first_field():
    assert next(iter(TicketClassification.model_fields)) == "razon"


def test_sanitize_keeps_local_rules_out_of_sent_copy():
    original = TicketClassification.model_json_schema()
    sent = sanitize_schema(original)

    assert "maxLength" not in json.dumps(sent)
    assert sent["additionalProperties"] is False
    assert sent["required"] == list(TicketClassification.model_fields)
    assert original["properties"]["resumen"]["maxLength"] == 280  # el original no cambia
    with pytest.raises(ValueError):
        TicketClassification(**{**VALID, "resumen": "x" * 281})
