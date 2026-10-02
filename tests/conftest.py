"""Utilidades compartidas por los tests. Ningún test usa la red."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from core.llm_client import LLMResponse
from core.logger import CallLogger


class FakeProvider:
    """Proveedor configurable: devuelve (o lanza) los elementos de `script` en orden."""

    def __init__(self, name: str, script: list, model: str = "fake-model") -> None:
        self.name = name
        self.model = model
        self._script = list(script)
        self.calls: list[dict] = []

    def generate(self, messages: list[dict], **params) -> LLMResponse:
        self.calls.append({"messages": messages, "params": params})
        item = self._script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


class FakeSleep:
    """Reemplazo de time.sleep que solo registra las esperas."""

    def __init__(self) -> None:
        self.waits: list[float] = []

    def __call__(self, seconds: float) -> None:
        self.waits.append(seconds)


def make_response(provider: str = "fake", text: str = "ok") -> LLMResponse:
    return LLMResponse(
        text=text,
        provider=provider,
        model="fake-model",
        tokens_in=10,
        tokens_out=5,
        latency_ms=12.0,
        cost_usd=0.0001,
    )


def read_events(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


@pytest.fixture
def log_path(tmp_path: Path) -> Path:
    return tmp_path / "llm_calls.jsonl"


@pytest.fixture
def logger(log_path: Path) -> CallLogger:
    return CallLogger(log_path)


@pytest.fixture
def fake_sleep() -> FakeSleep:
    return FakeSleep()


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Evita que las variables del entorno del alumno afecten los tests."""
    for var in ("GEMINI_API_KEY", "GROQ_API_KEY", "PRIMARY", "FALLBACKS"):
        monkeypatch.delenv(var, raising=False)
