"""Registry de prompts versionados en YAML.

Cada prompt vive en `prompts/<nombre>/v<N>.yaml`. Exactamente una versión por prompt
tiene `active: true`; esa es la que se usa si no pides una versión concreta.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import yaml
from jinja2 import Environment, StrictUndefined, UndefinedError

# Categorías válidas del clasificador de tickets. Única fuente de verdad en todo el repo.
CATEGORIES: tuple[str, ...] = ("falla", "solicitud", "proveedor", "facturacion", "otro")

REQUIRED_KEYS = ("name", "version", "active", "description", "model_params", "system", "user")
_VERSION_FILE = re.compile(r"^v(\d+)\.yaml$")

# Las plantillas se cargan desde string: sin loader, la plantilla no puede leer archivos.
_ENV = Environment(autoescape=False, keep_trailing_newline=False, undefined=StrictUndefined)


class PromptNotFoundError(LookupError):
    """El prompt o la versión pedida no existe."""


@dataclass(frozen=True)
class Prompt:
    name: str
    version: int
    description: str
    model_params: dict
    system_template: str
    user_template: str
    examples: list[dict]

    def render(self, **variables) -> list[dict]:
        """Devuelve los mensajes `system` y `user` listos para `LLMClient.generate()`.

        `examples` y `categories` se inyectan solos; el llamador solo pasa sus variables.
        """
        context = {"examples": self.examples, "categories": list(CATEGORIES), **variables}
        try:
            system = _ENV.from_string(self.system_template).render(context)
            user = _ENV.from_string(self.user_template).render(context)
        except UndefinedError as e:
            raise ValueError(
                f"Falta una variable al renderizar {self.name} v{self.version}: {e.message}"
            ) from e
        return [
            {"role": "system", "content": system.strip()},
            {"role": "user", "content": user.strip()},
        ]


class PromptKit:
    def __init__(self, root: str | Path = "prompts") -> None:
        self.root = Path(root)

    def versions(self, name: str) -> list[int]:
        """Versiones disponibles de un prompt, de menor a mayor."""
        return sorted(self._files(name))

    def get(self, name: str, version: int | None = None) -> Prompt:
        """Devuelve la versión pedida o, si no se indica, la que tiene `active: true`."""
        files = self._files(name)
        if version is not None:
            if version not in files:
                raise PromptNotFoundError(
                    f"No existe {name} v{version}. Versiones disponibles: {sorted(files)}."
                )
            return self._build(name, version, self._load(files[version], name, version))

        loaded = {v: self._load(path, name, v) for v, path in files.items()}
        active = [v for v, data in loaded.items() if data["active"] is True]
        if len(active) != 1:
            paths = ", ".join(str(files[v]) for v in (active or sorted(files)))
            problem = "ninguna versión activa" if not active else "más de una versión activa"
            error = PromptNotFoundError if not active else ValueError
            raise error(
                f"{name} tiene {problem}: debe haber exactamente una con `active: true`. "
                f"Revisa: {paths}"
            )
        return self._build(name, active[0], loaded[active[0]])

    def _files(self, name: str) -> dict[int, Path]:
        folder = self.root / name
        if not folder.is_dir():
            raise PromptNotFoundError(f"No existe el prompt {name!r} (se buscó en {folder}).")
        files = {}
        for path in folder.glob("v*.yaml"):
            match = _VERSION_FILE.match(path.name)
            if match:
                files[int(match.group(1))] = path
        if not files:
            raise PromptNotFoundError(f"{folder} no tiene archivos v<N>.yaml.")
        return files

    @staticmethod
    def _load(path: Path, name: str, version: int) -> dict:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError(f"{path} no contiene un mapa YAML.")
        missing = [key for key in REQUIRED_KEYS if key not in data]
        if missing:
            raise ValueError(f"{path} no tiene los campos: {', '.join(missing)}.")
        if data["name"] != name:
            raise ValueError(f"{path}: name es {data['name']!r} pero la carpeta es {name!r}.")
        if data["version"] != version:
            raise ValueError(
                f"{path}: version es {data['version']!r} pero el archivo es v{version}.yaml."
            )
        return data

    def _build(self, name: str, version: int, data: dict) -> Prompt:
        return Prompt(
            name=name,
            version=version,
            description=data["description"],
            model_params=dict(data["model_params"] or {}),
            system_template=data["system"],
            user_template=data["user"],
            examples=self._examples(name),
        )

    def _examples(self, name: str) -> list[dict]:
        path = self.root / name / "examples.jsonl"
        if not path.exists():
            return []
        lines = path.read_text(encoding="utf-8").splitlines()
        return [json.loads(line) for line in lines if line.strip()]
