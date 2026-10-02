"""Configuración del Agente Operador, leída de variables de entorno y del archivo `.env`.

Si falta una variable, el programa falla al inicio, no a mitad de una request.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import SecretStr, model_validator
from pydantic_settings import BaseSettings

KNOWN_PROVIDERS = ("gemini", "groq", "ollama")

# Proveedor -> variable de entorno con su llave. Ollama corre local y no necesita llave.
_KEY_ENV_VARS = {"gemini": "GEMINI_API_KEY", "groq": "GROQ_API_KEY"}


class Settings(BaseSettings):
    model_config = {"env_file": ".env"}

    gemini_api_key: SecretStr | None = None
    groq_api_key: SecretStr | None = None
    gemini_model: str = "gemini-3.8-flash"
    groq_model: str = "openai/gpt-oss-20b"
    ollama_model: str = "llama3.2"
    ollama_base_url: str = "http://localhost:11434/v1"
    primary: str = "gemini"
    fallbacks: list[str] = ["groq"]
    timeout_s: float = 30
    max_retries: int = 3
    log_path: Path = Path("logs/llm_calls.jsonl")

    @model_validator(mode="after")
    def _check_providers(self) -> Settings:
        for name in [self.primary, *self.fallbacks]:
            if name not in KNOWN_PROVIDERS:
                raise ValueError(
                    f"Proveedor desconocido: {name!r}. Usa uno de: {', '.join(KNOWN_PROVIDERS)}."
                )
            if name == "ollama":
                continue
            if getattr(self, f"{name}_api_key") is None:
                env_var = _KEY_ENV_VARS[name]
                raise ValueError(
                    f"Falta la llave de {name}. Agrega {env_var}=... a tu archivo .env "
                    f"o quita {name!r} de PRIMARY/FALLBACKS."
                )
        return self


@lru_cache
def get_settings() -> Settings:
    """Devuelve la configuración, creada la primera vez que se pide (no al importar)."""
    return Settings()
