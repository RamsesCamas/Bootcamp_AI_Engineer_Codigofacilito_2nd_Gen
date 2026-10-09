from __future__ import annotations

import httpx
import pytest
from pydantic import ValidationError

from core.config import Settings
from core.errors import AllProvidersFailedError, PermanentProviderError, TransientProviderError
from core.llm_client import LLMClient, ToolCall
from core.pricing import cost_usd
from core.providers import GeminiProvider, OllamaProvider
from tests.conftest import FakeProvider, make_response, read_events

MESSAGES = [{"role": "user", "content": "hola"}]


def test_primary_success_without_fallback(logger, log_path, fake_sleep):
    primary = FakeProvider("gemini", [make_response("gemini")])
    backup = FakeProvider("groq", [make_response("groq")])
    client = LLMClient([primary, backup], logger, sleep=fake_sleep)

    response = client.generate(MESSAGES)

    assert response.provider == "gemini"
    assert backup.calls == []
    assert fake_sleep.waits == []
    [event] = read_events(log_path)
    assert event["success"] is True
    assert event["fallback"] is False
    assert event["ttft_ms"] is None


def test_retries_429_with_exponential_backoff(logger, log_path, fake_sleep):
    rate_limited = TransientProviderError("gemini", "rate limit", 429)
    primary = FakeProvider("gemini", [rate_limited, rate_limited, make_response("gemini")])
    client = LLMClient([primary], logger, sleep=fake_sleep)

    response = client.generate(MESSAGES)

    assert response.provider == "gemini"
    assert len(fake_sleep.waits) == 2
    assert 1 <= fake_sleep.waits[0] <= 1.25
    assert 2 <= fake_sleep.waits[1] <= 2.25
    events = read_events(log_path)
    assert [e["success"] for e in events] == [False, False, True]
    assert events[0]["error_type"] == "TransientProviderError"


def test_persistent_503_falls_back_to_second_provider(logger, log_path, fake_sleep):
    unavailable = TransientProviderError("gemini", "unavailable", 503)
    primary = FakeProvider("gemini", [unavailable] * 4)
    backup = FakeProvider("groq", [make_response("groq")])
    client = LLMClient([primary, backup], logger, max_retries=3, sleep=fake_sleep)

    response = client.generate(MESSAGES)

    assert response.provider == "groq"
    assert len(primary.calls) == 4
    assert len(fake_sleep.waits) == 3
    events = read_events(log_path)
    assert len(events) == 5
    assert events[-1]["provider"] == "groq"
    assert events[-1]["fallback"] is True
    assert events[-1]["success"] is True


def test_401_is_not_retried(logger, log_path, fake_sleep):
    unauthorized = PermanentProviderError("gemini", "invalid key", 401)
    primary = FakeProvider("gemini", [unauthorized])
    backup = FakeProvider("groq", [make_response("groq")])
    client = LLMClient([primary, backup], logger, sleep=fake_sleep)

    response = client.generate(MESSAGES)

    assert response.provider == "groq"
    assert len(primary.calls) == 1
    assert fake_sleep.waits == []


def test_all_providers_fail(logger, fake_sleep):
    primary = FakeProvider("gemini", [PermanentProviderError("gemini", "bad", 400)])
    backup = FakeProvider("groq", [PermanentProviderError("groq", "bad", 401)])
    client = LLMClient([primary, backup], logger, sleep=fake_sleep)

    with pytest.raises(AllProvidersFailedError) as exc_info:
        client.generate(MESSAGES)

    assert [e.provider for e in exc_info.value.errors] == ["gemini", "groq"]


def test_params_are_passed_to_provider(logger, fake_sleep):
    primary = FakeProvider("gemini", [make_response("gemini")])
    client = LLMClient([primary], logger, sleep=fake_sleep)

    client.generate(MESSAGES, temperature=0.2)

    assert primary.calls[0]["params"] == {"temperature": 0.2}


def test_cost_matches_slide_24_example():
    # 1,500 tokens de entrada a $0.30/M + 400 de salida a $2.50/M = $0.00145
    assert cost_usd("gemini-2.5-flash", 1_500, 400) == pytest.approx(0.00145)


def test_ollama_is_free_and_unknown_model_warns_once():
    assert cost_usd("llama3.2", 1_000, 1_000, provider="ollama") == 0.0
    with pytest.warns(UserWarning):
        assert cost_usd("modelo-inventado", 100, 100) == 0.0


def test_settings_fail_without_primary_key():
    with pytest.raises(ValidationError, match="GEMINI_API_KEY"):
        Settings(_env_file=None, primary="gemini", fallbacks=[])


def test_settings_fail_without_fallback_key():
    with pytest.raises(ValidationError, match="GROQ_API_KEY"):
        Settings(_env_file=None, gemini_api_key="x", primary="gemini", fallbacks=["groq"])


def test_settings_ollama_needs_no_key():
    settings = Settings(_env_file=None, primary="ollama", fallbacks=[])
    assert settings.ollama_base_url == "http://localhost:11434/v1"


@pytest.mark.parametrize(
    ("status", "expected"),
    [(429, TransientProviderError), (503, TransientProviderError), (401, PermanentProviderError)],
)
def test_provider_maps_http_errors(monkeypatch, status, expected):
    request = httpx.Request("POST", "https://example.test/chat/completions")
    monkeypatch.setattr(httpx, "post", lambda *a, **kw: httpx.Response(status, request=request))
    provider = GeminiProvider("gemini-3.8-flash", "llave-falsa")

    with pytest.raises(expected):
        provider.generate(MESSAGES)


def test_provider_builds_response_with_usage_and_cost(monkeypatch):
    captured = {}

    def fake_post(url, json, headers, timeout):
        captured.update(url=url, json=json)
        body = {
            "model": "gemini-2.5-flash",
            "choices": [{"message": {"role": "assistant", "content": "Hola"}}],
            "usage": {"prompt_tokens": 1_500, "completion_tokens": 400},
        }
        return httpx.Response(200, json=body, request=httpx.Request("POST", url))

    monkeypatch.setattr(httpx, "post", fake_post)
    provider = GeminiProvider("gemini-2.5-flash", "llave-falsa")

    response = provider.generate(MESSAGES, temperature=0.3)

    assert captured["url"] == (
        "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"
    )
    assert captured["json"]["temperature"] == 0.3
    assert (response.text, response.tokens_in, response.tokens_out) == ("Hola", 1_500, 400)
    assert response.cost_usd == pytest.approx(0.00145)


def test_log_extra_is_in_every_attempt(logger, log_path, fake_sleep):
    rate_limited = TransientProviderError("gemini", "rate limit", 429)
    primary = FakeProvider("gemini", [rate_limited, rate_limited, make_response("gemini")])
    client = LLMClient([primary], logger, sleep=fake_sleep)
    extra = {"prompt_name": "ticket_classifier", "prompt_version": 2}

    client.generate(MESSAGES, log_extra=extra, temperature=0)

    events = read_events(log_path)
    assert len(events) == 3
    assert all(e["prompt_name"] == "ticket_classifier" for e in events)
    assert all(e["prompt_version"] == 2 for e in events)
    assert primary.calls[0]["params"] == {"temperature": 0}


def test_provider_parses_tool_calls_and_keeps_extra_content(monkeypatch):
    captured = {}

    def fake_post(url, json, headers, timeout):
        captured.update(json)
        body = {
            "model": "gemini-3.8-flash",
            "choices": [
                {
                    "finish_reason": "tool_calls",
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {
                                    "name": "buscar_ticket",
                                    "arguments": '{"ticket_id": "T-1042"}',
                                },
                                "extra_content": {"google": {"thought_signature": "sig"}},
                            }
                        ],
                    },
                }
            ],
            "usage": {"prompt_tokens": 50, "completion_tokens": 10},
        }
        return httpx.Response(200, json=body, request=httpx.Request("POST", url))

    monkeypatch.setattr(httpx, "post", fake_post)
    tools = [{"type": "function", "function": {"name": "buscar_ticket", "parameters": {}}}]
    response = GeminiProvider("gemini-3.8-flash", "llave-falsa").generate(MESSAGES, tools=tools)

    assert captured["tools"] == tools
    assert response.text == ""
    assert response.finish_reason == "tool_calls"
    [call] = response.tool_calls
    assert (call.id, call.name, call.arguments) == (
        "call_1",
        "buscar_ticket",
        '{"ticket_id": "T-1042"}',
    )
    message = response.as_message()
    assert message["role"] == "assistant" and message["content"] is None
    assert message["tool_calls"][0]["extra_content"] == {"google": {"thought_signature": "sig"}}
    assert message["tool_calls"][0]["function"]["name"] == "buscar_ticket"


def test_client_sends_optional_params_only_when_given(logger, log_path, fake_sleep):
    call = ToolCall("call_1", "buscar_ticket", "{}")
    response = make_response("gemini")
    response.tool_calls.append(call)
    primary = FakeProvider("gemini", [make_response("gemini"), response])
    client = LLMClient([primary], logger, sleep=fake_sleep)

    client.generate(MESSAGES)
    client.generate(MESSAGES, tools=[{"type": "function"}], response_format={"type": "json_object"})

    assert primary.calls[0]["params"] == {}
    assert set(primary.calls[1]["params"]) == {"tools", "response_format"}
    first, second = read_events(log_path)
    assert "tool_calls" not in first
    assert second["tool_calls"] == ["buscar_ticket"]


def test_capabilities_by_provider():
    assert GeminiProvider.supports_tools and GeminiProvider.supports_json_schema
    assert OllamaProvider.supports_tools and not OllamaProvider.supports_json_schema
    assert OllamaProvider.supports_json_object
