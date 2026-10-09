from __future__ import annotations

import copy
import json
import logging

import pytest
from pydantic import BaseModel

from contracts import BUSCAR_TICKET, MaxIterationsError, ToolSpec, execute_tool, run_tool_loop
from core.llm_client import LLMClient, ToolCall
from tests.conftest import FakeProvider, make_response, read_events

MESSAGES = [
    {"role": "system", "content": "Eres un asistente de operaciones."},
    {"role": "user", "content": "¿Cómo va el T-1042?"},
]
TOOLS = {BUSCAR_TICKET.name: BUSCAR_TICKET}


def with_calls(*calls: ToolCall):
    response = make_response("gemini", text="")
    response.tool_calls.extend(calls)
    return response


def call(call_id: str, name: str = "buscar_ticket", **args) -> ToolCall:
    return ToolCall(call_id, name, json.dumps(args or {"ticket_id": "T-1042"}))


def content(message: dict) -> dict:
    return json.loads(message["content"])


def client_for(script, logger, fake_sleep):
    provider = FakeProvider("gemini", script)
    return provider, LLMClient([provider], logger, sleep=fake_sleep)


def test_no_tool_calls_returns_text(logger, fake_sleep):
    provider, client = client_for([make_response("gemini", "Hola")], logger, fake_sleep)
    assert run_tool_loop(client, MESSAGES, [BUSCAR_TICKET]) == "Hola"
    assert provider.calls[0]["params"]["tools"] == [BUSCAR_TICKET.to_openai()]


def test_one_tool_call_then_text(logger, fake_sleep):
    script = [with_calls(call("c1")), make_response("gemini", "Está en revisión.")]
    provider, client = client_for(script, logger, fake_sleep)
    events = []

    answer = run_tool_loop(client, MESSAGES, [BUSCAR_TICKET], on_event=lambda *e: events.append(e))

    assert answer == "Está en revisión."
    second = provider.calls[1]["messages"]
    assert second[-2]["role"] == "assistant"
    assert second[-2]["tool_calls"][0]["id"] == "c1"
    assert second[-1]["role"] == "tool" and second[-1]["tool_call_id"] == "c1"
    assert content(second[-1])["estado"] == "en revisión"
    assert [kind for kind, *_ in events] == ["call", "result"]


def test_two_tool_calls_in_one_turn(logger, fake_sleep):
    script = [
        with_calls(call("c1"), call("c2", ticket_id="T-1043")),
        make_response("gemini", "Listo."),
    ]
    provider, client = client_for(script, logger, fake_sleep)

    run_tool_loop(client, MESSAGES, [BUSCAR_TICKET])

    results = [m for m in provider.calls[1]["messages"] if m["role"] == "tool"]
    assert [m["tool_call_id"] for m in results] == ["c1", "c2"]
    assert [content(m)["id"] for m in results] == ["T-1042", "T-1043"]


def test_unknown_tool_returns_error():
    result = execute_tool(call("c1", name="borrar_todo"), TOOLS)
    assert result["role"] == "tool" and result["tool_call_id"] == "c1"
    assert content(result) == {"error": "La herramienta borrar_todo no existe"}


@pytest.mark.parametrize("arguments", ['{"ticket_id": "1042"}', "{no es json", "{}"])
def test_invalid_arguments_return_error(arguments):
    result = execute_tool(ToolCall("c1", "buscar_ticket", arguments), TOOLS)
    assert content(result)["error"].startswith("Argumentos inválidos:")


class _Args(BaseModel):
    x: int


def _explode(args: _Args) -> dict:
    raise RuntimeError("secreto interno: /etc/passwd línea 42")


def test_handler_exception_returns_error_without_traceback(caplog):
    tools = {"falla": ToolSpec("falla", "Siempre falla.", _Args, _explode)}
    with caplog.at_level(logging.WARNING):
        result = execute_tool(ToolCall("c1", "falla", '{"x": 1}'), tools)

    assert content(result) == {"error": "La herramienta falla falló"}
    assert "secreto interno" not in result["content"]
    assert "Traceback" not in result["content"]
    assert "falla falló" in caplog.text  # el detalle se queda en el log local


def test_max_iterations(logger, fake_sleep):
    script = [with_calls(call(f"c{i}")) for i in range(3)]
    _, client = client_for(script, logger, fake_sleep)

    with pytest.raises(MaxIterationsError) as exc_info:
        run_tool_loop(client, MESSAGES, [BUSCAR_TICKET], max_iters=3)

    assert exc_info.value.max_iters == 3


def test_original_messages_not_modified(logger, fake_sleep):
    original = copy.deepcopy(MESSAGES)
    script = [with_calls(call("c1")), make_response("gemini", "ok")]
    _, client = client_for(script, logger, fake_sleep)

    run_tool_loop(client, MESSAGES, [BUSCAR_TICKET])

    assert original == MESSAGES


def test_tool_iteration_and_names_in_log(logger, log_path, fake_sleep):
    script = [with_calls(call("c1"), call("c2", ticket_id="T-1043")), make_response("gemini")]
    _, client = client_for(script, logger, fake_sleep)

    run_tool_loop(client, MESSAGES, [BUSCAR_TICKET], log_extra={"prompt_name": "operator_tools"})

    first, second = read_events(log_path)
    assert first["tool_iteration"] == 0 and second["tool_iteration"] == 1
    assert first["tool_calls"] == ["buscar_ticket", "buscar_ticket"]
    assert "tool_calls" not in second
    assert first["prompt_name"] == "operator_tools"
