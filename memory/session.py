"""Memoria de corto plazo: la conversación de una sesión, con un resumen de lo más viejo.

La CLI responde una pregunta por invocación. Con `--sesion ID`, cada turno se guarda en
var/sesiones/ID.json y el siguiente lo recibe como historial. Cuando el historial pasa de su
presupuesto, el LLM resume los turnos viejos junto con el resumen anterior (resumen
recursivo, como en MemGPT). Los turnos resumidos salen del prompt, no del archivo.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from context.manager import estimate_tokens
from prompting import PromptKit

if TYPE_CHECKING:
    from core.llm_client import LLMClient

SESIONES_PATH = Path("var") / "sesiones"
PROMPTS_DIR = Path(__file__).resolve().parents[1] / "prompts"
COMPANY = "Acme Operaciones"
PRESUPUESTO = 300  # tokens del historial antes de resumir
RECIENTES = 4  # los últimos turnos nunca se resumen: van completos
_ID = re.compile(r"^[\w-]{1,40}$")  # el id es un nombre de archivo: sin / ni ..

Rol = Literal["usuario", "agente"]


@dataclass
class Turno:
    rol: Rol
    texto: str


@dataclass
class Sesion:
    id: str
    turnos: list[Turno] = field(default_factory=list)
    resumen: str = ""
    resumidos: int = 0  # los primeros `resumidos` turnos ya están dentro del resumen

    def __post_init__(self) -> None:
        if not _ID.match(self.id):
            raise ValueError(f"id de sesión inválido: {self.id!r}; usa letras, números, - o _.")

    def agregar(self, rol: Rol, texto: str) -> None:
        self.turnos.append(Turno(rol, texto))

    @property
    def vigentes(self) -> list[Turno]:
        """Los turnos que van completos en el historial."""
        return self.turnos[self.resumidos :]

    def historial(self) -> str:
        """El texto del bloque `<historial>`: el resumen y los turnos vigentes."""
        lineas = [f"Resumen de lo anterior: {self.resumen}"] if self.resumen else []
        lineas += [f"{turno.rol}: {turno.texto}" for turno in self.vigentes]
        return "\n".join(lineas)

    def compactar(self, client: LLMClient, presupuesto: int = PRESUPUESTO) -> int:
        """Si el historial pasa del presupuesto, resume todo menos los RECIENTES.

        Regresa cuántos turnos entraron al resumen (0 si no hizo falta resumir).
        """
        if len(self.vigentes) <= RECIENTES or estimate_tokens(self.historial()) <= presupuesto:
            return 0
        hasta = len(self.turnos) - RECIENTES
        viejos = self.turnos[self.resumidos : hasta]
        prompt = PromptKit(PROMPTS_DIR).get("resumen_sesion")
        mensajes = prompt.render(
            company=COMPANY,
            resumen=self.resumen or "(ninguno)",
            turnos="\n".join(f"{turno.rol}: {turno.texto}" for turno in viejos),
        )
        log_extra = {"prompt_name": prompt.name, "prompt_version": prompt.version}
        respuesta = client.generate(mensajes, log_extra=log_extra, **prompt.model_params)
        self.resumen = respuesta.text.strip()
        self.resumidos = hasta
        return len(viejos)

    def guardar(self, carpeta: Path = SESIONES_PATH) -> Path:
        carpeta.mkdir(parents=True, exist_ok=True)
        path = carpeta / f"{self.id}.json"
        path.write_text(json.dumps(asdict(self), ensure_ascii=False, indent=2), encoding="utf-8")
        return path

    @classmethod
    def cargar(cls, sesion_id: str, carpeta: Path = SESIONES_PATH) -> Sesion:
        """La sesión guardada, o una nueva si no existe."""
        nueva = cls(sesion_id)  # valida el id antes de armar la ruta
        path = carpeta / f"{nueva.id}.json"
        if not path.exists():
            return nueva
        datos = json.loads(path.read_text(encoding="utf-8"))
        datos["turnos"] = [Turno(**turno) for turno in datos["turnos"]]
        return cls(**datos)
