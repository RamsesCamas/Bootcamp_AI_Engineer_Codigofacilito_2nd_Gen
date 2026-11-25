"""El agente ReAct como grafo de LangGraph (Clase 11).

El mismo ciclo de la Clase 4, escrito como grafo: un estado, dos nodos y una arista
condicional. `decidir` revisa el presupuesto y le pide al LLM el siguiente paso; `actuar`
ejecuta la herramienta que decidió. Cada nodo regresa solo lo que cambia del estado.

    START → decidir ─┬→ actuar → decidir
                     └→ END (respondió o se acabó el presupuesto)

Con un checkpointer, LangGraph guarda una foto del estado al terminar cada nodo. El
`thread_id` (aquí, el id de la ejecución) separa una ejecución de otra.
"""

from __future__ import annotations

import json
import operator
import time
import uuid
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph

from agents.budget import AgentBudget
from agents.checkpoint import dict_a_paso
from agents.react_agent import (
    COMPANY,
    PROMPTS_DIR,
    AgentStep,
    EventHandler,
    describe_tools,
    format_trace,
)
from agents.trace import AgentResult
from contracts.structured_output import StructuredOutputError, generate_structured
from contracts.tool_loop import execute_tool
from core.errors import AllProvidersFailedError
from prompting import PromptKit

if TYPE_CHECKING:
    from langgraph.checkpoint.base import BaseCheckpointSaver
    from langgraph.checkpoint.sqlite import SqliteSaver

    from agents.tools import ToolRunner
    from contracts.tool_schemas import ToolSpec
    from core.llm_client import LLMClient

CHECKPOINTS_PATH = Path("var") / "checkpoints.db"


class Estado(TypedDict):
    pregunta: str
    historial: str
    pasos: Annotated[list[dict], operator.add]  # cada nodo agrega pasos; no reemplaza la lista
    pendiente: dict | None  # la acción que decidió el LLM y que `actuar` todavía no ejecuta
    respuesta: str | None
    motivo: str | None


@contextmanager
def abrir_checkpointer(path: Path = CHECKPOINTS_PATH) -> Iterator[SqliteSaver]:
    """Checkpoints en SQLite: sobreviven al proceso, a diferencia de InMemorySaver."""
    from langgraph.checkpoint.sqlite import SqliteSaver

    path.parent.mkdir(parents=True, exist_ok=True)
    with SqliteSaver.from_conn_string(str(path)) as saver:
        yield saver


class AgenteGrafo:
    def __init__(
        self,
        client: LLMClient,
        tools: list[ToolSpec],
        *,
        budget: AgentBudget | None = None,
        runner: ToolRunner | None = None,
        checkpointer: BaseCheckpointSaver | None = None,
        clock: Callable[[], float] = time.monotonic,
        on_event: EventHandler | None = None,
    ) -> None:
        self.client = client
        self.tools = list(tools)
        self.tools_by_name = {tool.name: tool for tool in self.tools}
        self.budget = budget or AgentBudget()
        self.runner = runner
        self.clock = clock
        self.on_event = on_event
        self.prompt = PromptKit(PROMPTS_DIR).get("react_agent")
        # Sin checkpointer, uno en memoria: el estado de cada paso vive mientras viva el proceso.
        self.grafo = self._construir().compile(checkpointer=checkpointer or InMemorySaver())
        self.ejecucion = ""
        self._inicio = 0.0

    def _construir(self) -> StateGraph:
        grafo = StateGraph(Estado)
        grafo.add_node("decidir", self._decidir)
        grafo.add_node("actuar", self._actuar)
        grafo.add_edge(START, "decidir")
        grafo.add_conditional_edges("decidir", self._siguiente, ["actuar", END])
        grafo.add_edge("actuar", "decidir")
        return grafo

    def _emit(self, kind: str, **data: Any) -> None:
        if self.on_event:
            self.on_event(kind, data)

    def _decidir(self, estado: Estado) -> dict:
        """Revisa el presupuesto y le pide al LLM el siguiente paso."""
        pasos = estado["pasos"]
        motivo = self.budget.exhausted_totals(
            len(pasos),
            sum(p["tokens"] for p in pasos),
            sum(p["cost_usd"] for p in pasos),
            self.clock() - self._inicio,
        )
        if motivo:
            return {"motivo": motivo}
        n = len(pasos) + 1
        mensajes = self.prompt.render(
            company=COMPANY,
            tools=describe_tools(self.tools),
            pregunta=estado["pregunta"],
            traza=format_trace([dict_a_paso(p) for p in pasos]),
            historial=estado["historial"],
        )
        log_extra = {
            "prompt_name": self.prompt.name,
            "prompt_version": self.prompt.version,
            "agent_kind": "grafo",
            "agent_run_id": self.ejecucion,
            "agent_step": n,
        }
        with self.client.track() as uso:
            paso = generate_structured(
                self.client, mensajes, AgentStep, log_extra=log_extra, **self.prompt.model_params
            )
        self._emit("paso", n=n, paso=paso)
        if paso.accion == "responder":
            return {"respuesta": paso.respuesta, "motivo": "respuesta"}
        pendiente = {
            "paso": paso.model_dump(),
            "tokens": uso.tokens,
            "cost_usd": uso.cost_usd,
            "latency_ms": uso.latency_ms,
        }
        return {"pendiente": pendiente}

    def _actuar(self, estado: Estado) -> dict:
        """Ejecuta la herramienta que eligió `decidir` y agrega el paso a la traza."""
        pendiente = estado["pendiente"]
        n = len(estado["pasos"]) + 1
        call = AgentStep.model_validate(pendiente["paso"]).as_tool_call(f"paso-{n}")
        if self.runner and call.name in self.tools_by_name:
            message = self.runner.execute(call)
        else:
            message = execute_tool(call, self.tools_by_name)
        observacion = json.loads(message["content"])
        self._emit("observacion", n=n, observacion=observacion)
        return {"pasos": [{**pendiente, "observacion": observacion}], "pendiente": None}

    def _siguiente(self, estado: Estado) -> str:
        """La arista condicional: si hay una acción pendiente, a `actuar`; si no, se termina."""
        return "actuar" if estado["pendiente"] else END

    def _config(self) -> dict:
        return {"configurable": {"thread_id": self.ejecucion}}

    def run(
        self, pregunta: str, historial: str = "", *, ejecucion: str | None = None
    ) -> AgentResult:
        """Una ejecución nueva. `ejecucion` es su thread_id en el checkpointer."""
        self.ejecucion = ejecucion or uuid.uuid4().hex[:8]
        if self.grafo.get_state(self._config()).values:
            raise ValueError(
                f"ya existe la ejecución {self.ejecucion}; usa --reanudar {self.ejecucion} o "
                "elige otro id."
            )
        entrada = {
            "pregunta": pregunta,
            "historial": historial,
            "pasos": [],
            "pendiente": None,
            "respuesta": None,
            "motivo": None,
        }
        return self._invocar(entrada)

    def _invocar(self, entrada: dict | None) -> AgentResult:
        self._inicio = self.clock()
        with self.client.track() as total:
            try:
                self.grafo.invoke(entrada, self._config(), durability="sync")
            except (StructuredOutputError, AllProvidersFailedError):
                pass  # Como en el ReAct: motivo "error", con los pasos que alcanzaron a correr.
        valores = self.grafo.get_state(self._config()).values
        result = AgentResult(
            respuesta=valores["respuesta"],
            traza=[dict_a_paso(p) for p in valores["pasos"]],
            motivo=valores["motivo"] or "error",
            llamadas=total.calls,
            tokens=total.tokens,
            cost_usd=total.cost_usd,
            latency_ms=total.latency_ms,
        )
        if total.calls and self.client.last_request_id:
            self.client.logger.annotate(self.client.last_request_id, stop_reason=result.motivo)
        self._emit("fin", resultado=result)
        return result


def dibujar() -> str:
    """El grafo en Mermaid, sin llamar al LLM (se puede pegar en mermaid.live)."""
    return AgenteGrafo(None, []).grafo.get_graph().draw_mermaid()
