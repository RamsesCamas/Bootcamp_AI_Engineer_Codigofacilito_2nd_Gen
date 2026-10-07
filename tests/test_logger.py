from __future__ import annotations

import re

import pytest

from core.llm_client import LLMClient
from tests.conftest import FakeProvider, make_response, read_events

FAKE_SECRET = "sk-secreto-falso-123456"


def test_logger_writes_expected_fields(logger, log_path):
    logger.log(
        provider="gemini",
        model="gemini-3.8-flash",
        tokens_in=10,
        tokens_out=5,
        latency_ms=123.456,
        cost_usd=0.0001,
        fallback=False,
        success=True,
        error_type=None,
        prompt_chars=42,
    )

    [event] = read_events(log_path)
    assert set(event) == {
        "request_id",
        "timestamp",
        "provider",
        "model",
        "tokens_in",
        "tokens_out",
        "latency_ms",
        "ttft_ms",
        "cost_usd",
        "fallback",
        "success",
        "error_type",
        "prompt_chars",
    }
    assert re.fullmatch(r"[0-9a-f]{8}", event["request_id"])
    assert event["timestamp"].endswith("+00:00")
    assert event["ttft_ms"] is None


def test_logger_never_writes_secrets_or_prompt_text(logger, log_path, fake_sleep):
    messages = [
        {"role": "system", "content": "Eres un asistente."},
        {"role": "user", "content": f"Mi llave es {FAKE_SECRET}, no la compartas."},
    ]
    response = make_response("gemini", text=f"Tu llave es {FAKE_SECRET}")
    client = LLMClient([FakeProvider("gemini", [response])], logger, sleep=fake_sleep)

    client.generate(messages)

    content = log_path.read_text(encoding="utf-8")
    assert FAKE_SECRET not in content
    assert "Mi llave es" not in content
    assert "Tu llave es" not in content
    [event] = read_events(log_path)
    assert event["prompt_chars"] == sum(len(m["content"]) for m in messages)


def test_logger_with_log_extra_never_writes_prompt_text(logger, log_path, fake_sleep):
    messages = [{"role": "user", "content": f"<ticket>Mi llave es {FAKE_SECRET}</ticket>"}]
    response = make_response("gemini", text=f"Tu llave es {FAKE_SECRET}")
    client = LLMClient([FakeProvider("gemini", [response])], logger, sleep=fake_sleep)
    extra = {
        "prompt_name": "ticket_classifier",
        "prompt_version": 2,
        "context_tokens": {"system": 120, "user": 30},
        "context_truncated": ["history"],
    }

    client.generate(messages, log_extra=extra)

    content = log_path.read_text(encoding="utf-8")
    assert FAKE_SECRET not in content
    assert "<ticket>" not in content
    [event] = read_events(log_path)
    assert {key: event[key] for key in extra} == extra


def test_logger_rejects_unknown_extra_fields(logger):
    with pytest.raises(TypeError):
        logger.log(
            provider="gemini",
            model="m",
            tokens_in=0,
            tokens_out=0,
            latency_ms=0,
            cost_usd=0,
            fallback=False,
            success=True,
            error_type=None,
            prompt_chars=0,
            prompt_text="esto no debe llegar al log",
        )
