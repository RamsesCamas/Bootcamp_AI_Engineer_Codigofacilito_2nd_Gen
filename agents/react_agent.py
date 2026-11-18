"""Agente ReAct: piensa, actúa con una herramienta, observa y repite hasta responder."""

from __future__ import annotations

import json
import time
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, model_validator

from agents.budget import AgentBudget, StopReason
from agents.trace import AgentResult, TraceStep
from contracts.structured_output import StructuredOutputError, generate_structured, sanitize_schema
from contracts.tool_loop import execute_tool
from core.errors import AllProvidersFailedError
from core.llm_client import ToolCall
from prompting import PromptKit

if TYPE_CHECKING:
    from agents.tools import ToolRunner
    from contracts.tool_schemas import ToolSpec
    from core.llm_client import LLMClient
    from core.usage import UsageMeter
    from prompting import Prompt

PROMPTS_DIR = Path(__file__).resolve().parents[1] / "prompts"
COMPANY = "Acme Operaciones"
OBSERVATION_LIMIT = 4_000  # Clase 7: caben 3 fragmentos de documentos, no solo un ticket.
TRUNCATION_MARK = "[…recortado]"

# on_event(tipo, datos): "paso" {n, paso}, "observacion" {n, observacion},
# "fin" {resultado}. El agente no imprime; la CLI decide qué mostrar.
EventHandler = Callable[[str, dict[str, Any]], None]


class AgentStep(BaseModel):
    pensamiento: str
    accion: Literal["usar_herramienta", "responder"]
    herramienta: str | None
    argumentos_json: str | None
    respuesta: str | None

    @model_validator(mode="after")
    def _accion_completa(self) -> AgentStep:
        if self.accion == "usar_herramienta":
            if not self.herramienta:
                raise ValueError(
                    'Con accion "usar_herramienta" debes indicar "herramienta" con su nombre.'
                )
            if self.argumentos_json:
                try:
                    json.loads(self.argumentos_json)
                except json.JSONDecodeError as e:
                    raise ValueError(
                        f'"argumentos_json" debe ser un objeto JSON válido en texto: {e.msg}.'
                    ) from e
        elif not self.respuesta:
            raise ValueError('Con accion "responder" debes escribir la "respuesta" final.')
        return self

    def as_tool_call(self, step_id: str) -> ToolCall:
        return ToolCall(step_id, self.herramienta or "", self.argumentos_json or "{}")


def describe_tools(tools: list[ToolSpec]) -> str:
    """Nombre, descripción y schema de argumentos de cada herramienta, para el prompt."""
    if not tools:
        return "(ninguna: responde con lo que ya sabes)"
    lines = []
    for tool in tools:
        schema = sanitize_schema(tool.args_model.model_json_schema(), strict=False)
        lines.append(f"- {tool.name}: {tool.description}")
        lines.append(f"  argumentos_json: {json.dumps(schema, ensure_ascii=False)}")
    return "\n".join(lines)


def format_observation(observacion: dict | None) -> str:
    text = json.dumps(observacion, ensure_ascii=False)
    if len(text) > OBSERVATION_LIMIT:
        text = text[:OBSERVATION_LIMIT] + TRUNCATION_MARK
    return text


def format_trace(traza: list[TraceStep]) -> str:
    lines = []
    for n, step in enumerate(traza, start=1):
        paso = step.paso
        lines.append(f"Paso {n} · Pensamiento: {paso.pensamiento}")
        lines.append(f"Paso {n} · Acción: {paso.herramienta} {paso.argumentos_json or '{}'}")
        lines.append(f"Paso {n} · Observación: {format_observation(step.observacion)}")
    return "\n".join(lines)


class ReActAgent:
    def __init__(
        self,
        client: LLMClient,
        tools: list[ToolSpec],
        *,
        budget: AgentBudget | None = None,
        prompt: Prompt | None = None,
        clock: Callable[[], float] = time.monotonic,
        on_event: EventHandler | None = None,
        agent_kind: str = "react",
        run_id: str | None = None,
        runner: ToolRunner | None = None,
    ) -> None:
        self.client = client
        self.tools = list(tools)
        self.tools_by_name = {tool.name: tool for tool in self.tools}
        self.budget = budget or AgentBudget()
        self.prompt = prompt or PromptKit(PROMPTS_DIR).get("react_agent")
        self.clock = clock
        self.on_event = on_event
        # PlanAndExecute usa ReActAgent como executor: comparte su run_id y su agent_kind,
        # y es él quien anota el motivo de parada en el log.
        self.agent_kind = agent_kind
        self._shared_run_id = run_id
        # Clase 5: con runner, las herramientas pasan por su política (permiso, tope,
        # timeout, reintentos). Sin runner, se ejecutan directo como en la Clase 4.
        self.runner = runner

    def _emit(self, kind: str, **data: Any) -> None:
        if self.on_event:
            self.on_event(kind, data)

    def _messages(self, pregunta: str, traza: list[TraceStep], historial: str) -> list[dict]:
        return self.prompt.render(
            company=COMPANY,
            tools=describe_tools(self.tools),
            pregunta=pregunta,
            traza=format_trace(traza),
            historial=historial,
        )

    def run(self, pregunta: str, historial: str = "") -> AgentResult:
        """Recorre el ciclo ReAct. `historial` es la conversación previa (Clase 9)."""
        run_id = self._shared_run_id or uuid.uuid4().hex[:8]
        start = self.clock()
        traza: list[TraceStep] = []

        with self.client.track() as total:
            for n in range(1, self.budget.max_pasos + 2):
                motivo = self.budget.exhausted(traza, self.clock() - start)
                if motivo:
                    return self._finish(None, traza, motivo, total, run_id)

                log_extra = {
                    "prompt_name": self.prompt.name,
                    "prompt_version": self.prompt.version,
                    "agent_kind": self.agent_kind,
                    "agent_run_id": run_id,
                    "agent_step": n,
                }
                with self.client.track() as step_usage:
                    try:
                        paso = generate_structured(
                            self.client,
                            self._messages(pregunta, traza, historial),
                            AgentStep,
                            log_extra=log_extra,
                            **self.prompt.model_params,
                        )
                    except (StructuredOutputError, AllProvidersFailedError):
                        return self._finish(None, traza, "error", total, run_id)
                self._emit("paso", n=n, paso=paso)

                if paso.accion == "responder":
                    return self._finish(paso.respuesta, traza, "respuesta", total, run_id)

                call = paso.as_tool_call(f"paso-{n}")
                if self.runner and call.name in self.tools_by_name:
                    message = self.runner.execute(call)
                else:
                    message = execute_tool(call, self.tools_by_name)
                observacion = json.loads(message["content"])
                traza.append(
                    TraceStep(
                        paso=paso,
                        observacion=observacion,
                        tokens=step_usage.tokens,
                        cost_usd=step_usage.cost_usd,
                        latency_ms=step_usage.latency_ms,
                    )
                )
                self._emit("observacion", n=n, observacion=observacion)

            return self._finish(None, traza, "max_pasos", total, run_id)  # pragma: no cover

    def _finish(
        self,
        respuesta: str | None,
        traza: list[TraceStep],
        motivo: StopReason,
        usage: UsageMeter,
        run_id: str,
    ) -> AgentResult:
        result = AgentResult(
            respuesta=respuesta,
            traza=traza,
            motivo=motivo,
            llamadas=usage.calls,
            tokens=usage.tokens,
            cost_usd=usage.cost_usd,
            latency_ms=usage.latency_ms,
        )
        if self._shared_run_id is None and usage.calls and self.client.last_request_id:
            self.client.logger.annotate(self.client.last_request_id, stop_reason=motivo)
        self._emit("fin", resultado=result)
        return result
