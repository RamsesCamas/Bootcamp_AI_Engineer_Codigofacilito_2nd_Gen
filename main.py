"""CLI del Agente Operador.

Uso:
    uv run main.py "Resume este ticket: el portal de proveedores no carga desde las 9:00"
    uv run main.py --provider groq "hola"
    uv run main.py --demo
    uv run main.py --prompt ticket_classifier --ticket T-1099
    uv run main.py --prompt ticket_classifier --prompt-version 1 "No abre el ERP"
    uv run main.py --structured --ticket T-1042
    uv run main.py --tools "¿Cómo va el T-1042?"
    uv run main.py --agent react "Compara el T-1042 y el T-1014"
    uv run main.py --agent plan "Compara el T-1042 y el T-1014"
    uv run main.py --agent react --mcp "¿Cómo va el T-1042?"
    uv run main.py --agent react --rol lectura "Marca el T-1042 como resuelto"
    uv run main.py --agent react "Crea un ticket: el portal de pagos rechaza tarjetas"
    uv run main.py --rag "¿Cuál es el tiempo de respuesta de PagosRed según su contrato?"
    uv run main.py --rag "¿Cuándo escalo un ticket de prioridad alta?" --k 3 --embedder fake
    uv run main.py --rag "¿Qué pasa si PagosRed no cumple?" --tipo contrato --proveedor PagosRed
    uv run main.py --rag "¿Qué pasó con el T-1016?" --modo hibrida
    uv run main.py --rag "¿Cuánto tardan los de pagos en contestar?" --modo hibrida --rerank
    uv run main.py --agent react --docs "El T-1042 lleva 6 horas sin respuesta: ¿escalo?"
    uv run main.py --rag "¿Qué penalización tiene el proveedor del T-1042?" --modo grafo
    uv run main.py --agent react --docs --grafo "El T-1043 lleva 3 horas: ¿ya incumplió?"
    uv run main.py --agent react --sesion demo "¿Cómo va el T-1042?"
    uv run main.py --rag "¿A quién escalé el T-1043?" --modo auto
    uv run main.py --agent react --memoria "¿Qué quedé de revisar con PagosRed?"
    uv run main.py --agent react --ejecucion r1 --falla-en 3 "Compara el T-1042 y el T-1043"
    uv run main.py --agent react --reanudar r1
    uv run main.py --agent grafo --verbose "Compara el T-1042 y el T-1043"
    uv run main.py --agent grafo --ejecucion g1 --falla-en 3 "Compara el T-1042 y el T-1043"
    uv run main.py --agent grafo --reanudar g1
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import ValidationError

from contracts import (
    BUSCAR_TICKET,
    CONSULTAR_PROVEEDOR,
    MaxIterationsError,
    StructuredOutputError,
    TicketClassification,
    generate_structured,
    run_tool_loop,
)
from core.config import KNOWN_PROVIDERS, Settings, get_settings
from core.errors import AllProvidersFailedError, ProviderError
from core.llm_client import LLMClient, ToolCall
from prompting import PromptKit, PromptNotFoundError

if TYPE_CHECKING:
    from knowledge.graph import KnowledgeGraph
    from knowledge.store import VectorStore

# --demo: solo Gemini, sin fallbacks. Funciona aunque GroqProvider siga con su TODO.
DEMO_TICKET_ID = "T-1043"
TICKETS_PATH = Path(__file__).parent / "data" / "tickets_ejemplo.jsonl"

DATA_DIR = Path(__file__).parent / "data"
PROMPTS_DIR = Path(__file__).parent / "prompts"
# Variables fijas por prompt; `ticket` sale de --ticket o del texto de la línea de comandos.
PROMPT_VARIABLES = {
    "ticket_classifier": {"company": "Acme Operaciones"},
    "ticket_triage": {"company": "Acme Operaciones"},
    "operator_tools": {"company": "Acme Operaciones"},
}
TICKETS_DB_PATH = DATA_DIR / "tickets_db.json"
# Clase 5: nombre, permiso, resultado e intentos de cada herramienta (nunca argumentos).
TOOL_LOG_PATH = Path(__file__).parent / "logs" / "tool_calls.jsonl"
# Clase 6: RAG sobre data/docs, indexados con scripts/ingest_docs.py.
RAG_K = 5
# Clase 7: los modos de knowledge/retrieval.py (aquí sin importarlo: carga ChromaDB).
RAG_MODOS = ("vector", "bm25", "hibrida", "grafo", "hibrida+grafo", "auto")

# Herramientas disponibles en el modo --tools.
TOOLS = [BUSCAR_TICKET, CONSULTAR_PROVEEDOR]
TOOL_RESULT_PREVIEW = 120  # Caracteres que se muestran de cada resultado en pantalla.

GRAY = "\033[90m"
RESET = "\033[0m"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Agente Operador: pregunta algo al LLM.")
    parser.add_argument(
        "prompt",
        nargs="?",
        help=f"Texto a enviar al modelo (con --demo, por defecto el ticket {DEMO_TICKET_ID}).",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--provider",
        choices=KNOWN_PROVIDERS,
        help="Fuerza un proveedor y desactiva los fallbacks.",
    )
    mode.add_argument(
        "--demo",
        action="store_true",
        help="Demo de la clase: usa solo Gemini, sin fallbacks (no necesita Groq).",
    )
    parser.add_argument(
        "--prompt",
        dest="prompt_name",
        metavar="NOMBRE",
        help="Usa un prompt versionado de prompts/ (versión activa por defecto).",
    )
    parser.add_argument(
        "--prompt-version", type=int, metavar="N", help="Versión del prompt (requiere --prompt)."
    )
    parser.add_argument(
        "--ticket",
        metavar="ID",
        help="Usa la descripción de un ticket de data/*.jsonl o data/tickets_db.json.",
    )
    parser.add_argument(
        "--structured",
        action="store_true",
        help="Clasifica el ticket y regresa un TicketClassification en JSON (Clase 3).",
    )
    parser.add_argument(
        "--tools",
        action="store_true",
        help="Responde la pregunta con herramientas, como buscar_ticket (Clase 3).",
    )
    parser.add_argument(
        "--agent",
        choices=("react", "plan", "grafo"),
        help="Responde con un agente ReAct o plan-and-execute (Clase 4), o con el ReAct como "
        "grafo de LangGraph (Clase 11).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Con --agent, muestra pensamientos y observaciones completas.",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Con --agent, verifica la respuesta contra la traza y la corrige una vez.",
    )
    parser.add_argument(
        "--mcp",
        action="store_true",
        help="Con --agent, toma las herramientas del server MCP por stdio (Clase 5).",
    )
    parser.add_argument(
        "--rol",
        choices=("lectura", "operador"),
        default="operador",
        help="Con --agent, rol para el ToolRunner: lectura u operador (Clase 5).",
    )
    parser.add_argument(
        "--rag",
        metavar="PREGUNTA",
        help="Responde con los documentos de operaciones y cita las fuentes (Clase 6).",
    )
    parser.add_argument(
        "--k", type=int, help=f"Con --rag, cuántos chunks recuperar (por defecto {RAG_K})."
    )
    parser.add_argument(
        "--embedder",
        choices=("gemini", "fake"),
        help="Con --rag, embedder del índice (por defecto gemini; fake no usa red ni cuota).",
    )
    parser.add_argument(
        "--tipo",
        choices=("runbook", "politica", "contrato", "faq"),
        help="Con --rag, busca solo en documentos de este tipo.",
    )
    parser.add_argument(
        "--proveedor", help='Con --rag, busca solo en documentos de este proveedor ("PagosRed").'
    )
    parser.add_argument(
        "--modo",
        choices=RAG_MODOS,
        default="vector",
        help="Con --rag, cómo buscar: vector (Clase 6), bm25 o hibrida (Clase 7), grafo "
        "(Clase 8) o auto: el router elige búsqueda, grafo o memoria (Clase 9).",
    )
    parser.add_argument(
        "--reescribir",
        action="store_true",
        help="Con --rag, el LLM convierte la pregunta en varias consultas (Clase 7).",
    )
    parser.add_argument(
        "--rerank",
        action="store_true",
        help="Con --rag, el LLM califica los candidatos y deja solo los relevantes (Clase 7).",
    )
    parser.add_argument(
        "--docs",
        action="store_true",
        help="Con --agent, agrega buscar_documentos: búsqueda híbrida en data/docs (Clase 7).",
    )
    parser.add_argument(
        "--grafo",
        action="store_true",
        help="Con --agent, agrega consultar_grafo: relaciones de var/grafo.json (Clase 8).",
    )
    parser.add_argument(
        "--memoria",
        action="store_true",
        help="Con --agent, agrega consultar_memoria: los recuerdos de var/memoria.json (Clase 9).",
    )
    parser.add_argument(
        "--sesion",
        metavar="ID",
        help="Con --agent react o grafo, guarda la conversación en var/sesiones/ID.json y la "
        "usa como historial del siguiente turno (Clase 9).",
    )
    parser.add_argument(
        "--ejecucion",
        metavar="ID",
        help="Con --agent react, guarda el estado de cada paso en var/ejecuciones/ID.json "
        "(reto de la Clase 9); con --agent grafo, es el thread_id en var/checkpoints.db "
        "(Clase 11).",
    )
    parser.add_argument(
        "--reanudar",
        metavar="ID",
        help="Con --agent react o grafo, retoma la ejecución ID desde su último paso "
        "guardado, con la misma pregunta.",
    )
    parser.add_argument(
        "--falla-en",
        type=int,
        metavar="N",
        help="Con --agent, simula que el proveedor se cae desde la llamada N (para practicar "
        "--reanudar).",
    )
    args = parser.parse_args(argv)
    if args.rag is not None:
        others = (
            *(args.prompt, args.ticket, args.agent, args.prompt_name, args.demo, args.sesion),
            *(args.ejecucion, args.reanudar, args.falla_en),
        )
        if any(others) or args.structured or args.tools or args.docs or args.grafo or args.memoria:
            parser.error("--rag recibe la pregunta y no se combina con otros modos.")
        if args.modo == "auto" and (args.reescribir or args.rerank):
            parser.error("--modo auto no se combina con --reescribir ni --rerank.")
        return args
    if args.k is not None or (args.embedder is not None and not (args.docs or args.memoria)):
        parser.error("--k requiere --rag; --embedder requiere --rag, --docs o --memoria.")
    if args.tipo is not None or args.proveedor is not None:
        parser.error("--tipo y --proveedor requieren --rag.")
    if args.modo != "vector" or args.reescribir or args.rerank:
        parser.error("--modo, --reescribir y --rerank requieren --rag.")
    if (args.docs or args.grafo or args.memoria) and (not args.agent or args.mcp):
        parser.error("--docs, --grafo y --memoria requieren --agent y no se combinan con --mcp.")
    if args.verify and not args.agent:
        parser.error("--verify requiere --agent.")
    if args.sesion and args.agent not in ("react", "grafo"):
        parser.error("--sesion requiere --agent react o grafo.")
    if args.ejecucion and (args.agent not in ("react", "grafo") or args.mcp):
        parser.error("--ejecucion requiere --agent react o grafo, sin --mcp.")
    if args.reanudar and (args.agent not in ("react", "grafo") or args.mcp):
        parser.error("--reanudar requiere --agent react o grafo, sin --mcp.")
    if args.ejecucion and args.reanudar:
        parser.error("--ejecucion y --reanudar no se usan juntos.")
    if args.reanudar and (args.prompt or args.sesion):
        parser.error("--reanudar retoma la pregunta guardada: no escribas otra ni uses --sesion.")
    if args.falla_en is not None and not args.agent:
        parser.error("--falla-en requiere --agent.")
    if args.mcp and not args.agent:
        parser.error("--mcp requiere --agent.")
    if args.agent and (args.structured or args.tools or args.prompt_name or args.demo):
        parser.error("--agent no se combina con --structured, --tools, --prompt ni --demo.")
    if args.agent and not args.reanudar and (args.prompt is None or args.ticket):
        parser.error('--agent necesita la pregunta como texto, por ejemplo "¿Cómo va el T-1042?".')
    if args.prompt is None and not (args.demo or args.ticket or args.reanudar):
        parser.error("falta el texto a enviar (o usa --demo o --ticket).")
    if args.prompt_version is not None and not args.prompt_name:
        parser.error("--prompt-version requiere --prompt.")
    if args.structured and args.tools:
        parser.error("--structured y --tools no se pueden usar juntos.")
    if (args.structured or args.tools) and (args.prompt_name or args.demo):
        parser.error("--structured y --tools usan su propio prompt: quita --prompt o --demo.")
    if args.tools and (args.prompt is None or args.ticket):
        parser.error('--tools necesita la pregunta como texto, por ejemplo "¿Cómo va el T-1042?".')
    return args


def find_ticket(ticket_id: str) -> dict:
    """Busca un ticket por `id` en todos los data/*.jsonl y en data/tickets_db.json."""
    for path in sorted(DATA_DIR.glob("*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip() and (ticket := json.loads(line)).get("id") == ticket_id:
                return ticket
    for ticket in json.loads(TICKETS_DB_PATH.read_text(encoding="utf-8")):
        if ticket["id"] == ticket_id:
            return ticket
    raise LookupError(f"No encontré el ticket {ticket_id} en {DATA_DIR}.")


def print_tool_event(kind: str, call: ToolCall, result: dict | None) -> None:
    """Muestra el ciclo de herramientas: → lo que pide el modelo, ← lo que regresa."""
    if kind == "call":
        print(f"{GRAY}→ {call.name} {call.arguments}{RESET}")
        return
    content = result["content"] if result else ""
    if len(content) > TOOL_RESULT_PREVIEW:
        content = content[: TOOL_RESULT_PREVIEW - 1] + "…"
    print(f"{GRAY}← {content}{RESET}")


def run_structured(client: LLMClient, text: str) -> int:
    prompt = PromptKit(PROMPTS_DIR).get("ticket_triage")
    messages = prompt.render(**PROMPT_VARIABLES[prompt.name], ticket=text)
    log_extra = {"prompt_name": prompt.name, "prompt_version": prompt.version}
    result = generate_structured(
        client, messages, TicketClassification, log_extra=log_extra, **prompt.model_params
    )
    print(result.model_dump_json(indent=2))
    return 0


def run_tools(client: LLMClient, question: str) -> int:
    prompt = PromptKit(PROMPTS_DIR).get("operator_tools")
    messages = prompt.render(**PROMPT_VARIABLES[prompt.name], message=question)
    log_extra = {"prompt_name": prompt.name, "prompt_version": prompt.version}
    answer = run_tool_loop(
        client,
        messages,
        TOOLS,
        log_extra=log_extra,
        on_event=print_tool_event,
        **prompt.model_params,
    )
    print(answer.strip())
    return 0


STOP_MESSAGES = {
    "max_pasos": "el agente llegó al máximo de pasos sin responder.",
    "presupuesto": "el agente agotó su presupuesto de tokens o de costo.",
    "sin_progreso": "el agente repitió la misma acción sin avanzar.",
    "timeout": "el agente se quedó sin tiempo.",
    "error": "el modelo no produjo un paso válido o el proveedor falló.",
}


def describe_action(paso) -> str:
    """`buscar_ticket T-1042` o `responder`, para la línea de pasos."""
    if paso.accion == "responder":
        return "responder"
    try:
        values = json.loads(paso.argumentos_json or "{}").values()
    except (ValueError, AttributeError):
        values = [paso.argumentos_json]
    return " ".join([paso.herramienta or "?", *(str(v) for v in values)]).strip()


def run_agent(
    client: LLMClient,
    kind: str,
    question: str,
    verbose: bool,
    verify: bool = False,
    *,
    mcp: bool = False,
    rol: str = "operador",
    docs: bool = False,
    embedder: str | None = None,
    grafo: bool = False,
    sesion: str | None = None,
    memoria: bool = False,
    ejecucion: str | None = None,
    reanudar: str | None = None,
) -> int:
    from agents import PlanAndExecute, ReActAgent, verify_and_revise
    from agents.tools import ToolRunner
    from contracts.tool_schemas import (
        ACTUALIZAR_ESTADO,
        CREAR_TICKET,
        LISTAR_TICKETS_POR_PROVEEDOR,
    )

    actions: list[str] = []

    def on_event(event: str, data: dict) -> None:
        if event == "plan":
            plan = data["plan"]
            label = "replan" if data["replan"] else "plan"
            print(f"{label}: {len(plan.pasos)} pasos")
            for i, paso in enumerate(plan.pasos, start=1):
                tool = f" ({paso.herramienta})" if paso.herramienta else ""
                print(f"  {i}. {paso.objetivo}{tool}")
        elif event == "paso":
            paso = data["paso"]
            actions.append(f"[{len(actions) + 1}] {describe_action(paso)}")
            if verbose:
                print(f"{GRAY}{actions[-1]} · {paso.pensamiento}{RESET}")
        elif event == "observacion" and verbose:
            print(f"{GRAY}    ← {json.dumps(data['observacion'], ensure_ascii=False)}{RESET}")
        elif event == "reanudada":
            print(f"{GRAY}reanudada: {data['pasos']} pasos guardados{RESET}")
        elif event == "verificacion":
            if data["corregida"]:
                problemas = "; ".join(data["verificacion"]["problemas"])
                print(f"{GRAY}verificación: no aprobada ({problemas}) → respuesta corregida{RESET}")
            else:
                print(f"{GRAY}verificación: aprobada{RESET}")

    agent_cls = ReActAgent if kind == "react" else PlanAndExecute

    def correr(tools, runner=None):
        """Corre el agente elegido. El grafo de la Clase 11 recibe lo mismo que el ReAct."""
        nonlocal hilo
        if kind != "grafo":
            agente = agent_cls(client, tools, on_event=on_event, runner=runner)
            return agente.run(question, **run_kwargs)
        from agents.grafo_react import CHECKPOINTS_PATH, AgenteGrafo, abrir_checkpointer

        with abrir_checkpointer() as saver:
            agente = AgenteGrafo(
                client,
                tools,
                on_event=on_event,
                runner=runner,
                checkpointer=saver,
                verificar=verify,  # práctica guiada: el grafo verifica en su propio nodo
            )
            try:
                if reanudar:
                    resultado = agente.reanudar(reanudar)
                else:
                    historial = run_kwargs.get("historial", "")
                    resultado = agente.run(question, historial, ejecucion=ejecucion)
            except (ValueError, LookupError) as e:
                fail(str(e))
                return None
        hilo = agente.ejecucion
        print(f"{GRAY}ejecución {hilo} · checkpoints en {CHECKPOINTS_PATH}{RESET}")
        return resultado

    run_kwargs: dict = {}
    if sesion:
        # Clase 9: la conversación previa entra al prompt; lo más viejo, resumido.
        from memory.session import Sesion

        try:
            conversacion = Sesion.cargar(sesion)
        except ValueError as e:
            return fail(str(e))
        try:
            resumidos = conversacion.compactar(client)
        except AllProvidersFailedError:
            resumidos = 0
            print(f"{GRAY}Aviso: no se pudo resumir la sesión; va completa{RESET}", file=sys.stderr)
        nota = f" · {resumidos} turnos pasaron al resumen" if resumidos else ""
        print(f"{GRAY}sesión {sesion}: {len(conversacion.turnos)} turnos{nota}{RESET}")
        run_kwargs["historial"] = conversacion.historial()
    hilo = ejecucion or reanudar  # el id de la ejecución, para --reanudar
    guardada = None
    if hilo and kind == "react":
        # Reto de la Clase 9: el estado después de cada paso, en var/ejecuciones/ID.json.
        from agents.checkpoint import Ejecucion

        try:
            if reanudar:
                guardada = Ejecucion.cargar(reanudar)
            else:
                guardada = Ejecucion.crear(ejecucion, question, run_kwargs.get("historial", ""))
        except (ValueError, LookupError) as e:
            return fail(str(e))
        if not guardada.reanudable:
            return fail(f"la ejecución {reanudar} ya terminó ({guardada.motivo}).")
        question = guardada.pregunta
        run_kwargs.update(
            historial=guardada.historial, traza=guardada.traza(), al_paso=guardada.registrar
        )
        if reanudar:
            print(f"{GRAY}reanudada: {len(guardada.pasos)} pasos guardados{RESET}")
    if mcp:
        # El server MCP aplica su propio ToolRunner con OPERADOR_ROL.
        from operador_mcp.client import MCPToolset

        with MCPToolset(rol=rol) as toolset:
            print(f"{GRAY}herramientas: MCP por stdio · rol {rol}{RESET}")
            result = correr(toolset.tools())
    else:
        # El agente puede proponer actualizar_estado; el ToolRunner decide según el rol.
        extras = [ACTUALIZAR_ESTADO, LISTAR_TICKETS_POR_PROVEEDOR, CREAR_TICKET]
        tools = [*TOOLS, *(tool for tool in extras if tool not in TOOLS)]
        if docs:
            # Clase 7: la búsqueda en documentos es una herramienta más del agente.
            from knowledge.retrieval import Retriever
            from knowledge.store import EmbeddingMismatch
            from tools.documentos import buscar_documentos_tool

            try:
                store = open_docs_store(embedder)
            except (LookupError, EmbeddingMismatch) as e:
                return fail(str(e))
            tools.append(buscar_documentos_tool(Retriever(store, modo="hibrida")))
        if grafo:
            # Clase 8: relaciones del grafo de conocimiento.
            from tools.grafo import consultar_grafo_tool

            try:
                tools.append(consultar_grafo_tool(open_graph()))
            except LookupError as e:
                return fail(str(e))
        if memoria:
            # Clase 9: lo que el usuario contó en otras sesiones.
            from knowledge.embeddings import make_embedder
            from memory.recuerdos import MEMORIA_PATH, Memoria
            from tools.memoria import consultar_memoria_tool, guardar_recuerdo_tool

            if not MEMORIA_PATH.exists():
                return fail(
                    f"no hay memoria en {MEMORIA_PATH}; corre `uv run scripts/memoria.py "
                    "importar data/memorias_demo.jsonl` o consolida una sesión."
                )
            recuerdos = Memoria(make_embedder(embedder or "gemini"))
            tools += [consultar_memoria_tool(recuerdos), guardar_recuerdo_tool(recuerdos)]
        # Al reanudar, el mismo request_id: una escritura repetida no se duplica (Clase 5).
        request_id = guardada.request_id if guardada else None
        runner = ToolRunner(tools, rol=rol, log_path=TOOL_LOG_PATH, request_id=request_id)
        extras = (" + documentos" if docs else "") + (" + grafo" if grafo else "")
        extras += " + memoria" if memoria else ""
        print(f"{GRAY}herramientas: locales{extras} · rol {rol}{RESET}")
        result = correr(tools, runner)
    if result is None:
        return 1
    if guardada:
        guardada.terminar(result)
    respuesta, checked = result.respuesta, None
    llamadas, tokens, cost_usd = result.llamadas, result.tokens, result.cost_usd
    if verify and respuesta and kind != "grafo":
        with client.track() as extra:
            checked = verify_and_revise(client, question, respuesta, result.traza)
        respuesta = checked.respuesta
        llamadas += extra.calls
        tokens += extra.tokens
        cost_usd += extra.cost_usd

    if sesion and respuesta:
        conversacion.agregar("usuario", question)
        conversacion.agregar("agente", respuesta.strip())
        conversacion.guardar()

    if actions:
        print("   ".join(actions))
    if checked:
        if checked.corregida:
            problems = "; ".join(checked.verificacion.problemas)
            print(f"{GRAY}verificación: no aprobada ({problems}) → respuesta corregida{RESET}")
        else:
            print(f"{GRAY}verificación: aprobada{RESET}")
    if respuesta:
        print(respuesta.strip())
    print(
        f"{GRAY}motivo: {result.motivo} · {llamadas} llamadas · "
        f"{tokens:,} tokens · ${cost_usd:.6f}{RESET}"
    )
    if result.motivo != "respuesta":
        print(f"Aviso: {STOP_MESSAGES[result.motivo]}", file=sys.stderr)
        if hilo and result.motivo == "error":
            pista = f"uv run main.py --agent {kind} --reanudar {hilo}"
            print(f"Reanuda con: {pista}", file=sys.stderr)
        return 2
    return 0


def demo_prompt() -> str:
    """Arma el prompt de la demo a partir del ticket de ejemplo."""
    for line in TICKETS_PATH.read_text(encoding="utf-8").splitlines():
        ticket = json.loads(line)
        if ticket["id"] == DEMO_TICKET_ID:
            return f"Resume este ticket: {ticket['asunto']}. {ticket['descripcion']}"
    raise LookupError(f"No encontré el ticket {DEMO_TICKET_ID} en {TICKETS_PATH}")


def fail(message: str) -> int:
    print(f"Error: {message}", file=sys.stderr)
    return 1


def config_error(e: ValidationError) -> str:
    details = "; ".join(err["msg"].removeprefix("Value error, ") for err in e.errors())
    return f"configuración inválida. {details}"


def build_client(args: argparse.Namespace) -> LLMClient:
    """Arma el cliente. Con --prompt o --ticket, si la cadena de fallbacks no se puede
    construir (por ejemplo, GroqProvider pendiente), sigue solo con el primario."""
    if args.demo:
        return LLMClient.from_settings(Settings(primary="gemini", fallbacks=[]))
    if args.provider:
        return LLMClient.from_settings(Settings(primary=args.provider, fallbacks=[]))
    if not (args.prompt_name or args.ticket):
        return LLMClient.from_settings(get_settings())
    try:
        return LLMClient.from_settings(get_settings())
    except (ValidationError, NotImplementedError) as e:
        reason = config_error(e) if isinstance(e, ValidationError) else str(e)
        settings = Settings(fallbacks=[])
        print(
            f"{GRAY}Aviso: se usa solo {settings.primary}, sin fallback ({reason}){RESET}",
            file=sys.stderr,
        )
        return LLMClient.from_settings(settings)


def open_docs_store(embedder: str | None) -> VectorStore:
    """Abre el índice de data/docs. Lanza `LookupError` si no existe o está vacío."""
    from knowledge.embeddings import make_embedder
    from knowledge.store import STORE_PATH, open_store

    ingest_hint = "corre primero `uv run scripts/ingest_docs.py data/docs`"
    if not Path(STORE_PATH).exists():
        raise LookupError(f"no hay índice de documentos en {STORE_PATH}; {ingest_hint}.")
    store = open_store(make_embedder(embedder or "gemini"))
    if store.count() == 0:
        raise LookupError(f"el índice de documentos está vacío; {ingest_hint}.")
    return store


def open_graph() -> KnowledgeGraph:
    """Carga var/grafo.json. Lanza `LookupError` si no se ha construido."""
    from knowledge.graph import GRAPH_PATH, KnowledgeGraph

    if not GRAPH_PATH.exists():
        raise LookupError(f"no hay grafo en {GRAPH_PATH}; corre `uv run scripts/build_graph.py`.")
    return KnowledgeGraph.load(GRAPH_PATH)


def open_router(store: VectorStore, client: LLMClient):
    """Clase 9: el router con las fuentes que existan. Los documentos siempre; el grafo y la
    memoria, si ya se crearon."""
    from knowledge.graph import GRAPH_PATH
    from knowledge.retrieval import Retriever
    from knowledge.router import Router
    from memory.recuerdos import MEMORIA_PATH, Memoria

    fuentes = {"busqueda": Retriever(store, modo="hibrida")}
    if GRAPH_PATH.exists():
        fuentes["grafo"] = Retriever(store, modo="grafo", grafo=open_graph())
    if MEMORIA_PATH.exists():
        fuentes["memoria"] = Memoria(store.embedder)
    return Router(fuentes, client)


def run_rag(args: argparse.Namespace) -> int:
    """`--rag`: busca en el índice, responde con citas e imprime las fuentes (S43)."""
    from knowledge.rag import UMBRAL, answer
    from knowledge.retrieval import Retriever
    from knowledge.store import EmbeddingMismatch, build_where

    base, *pasos = args.modo.split("+")
    try:
        store = open_docs_store(args.embedder)
        grafo = open_graph() if "grafo" in (base, *pasos) else None
    except (LookupError, EmbeddingMismatch) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 2

    retriever = router = None
    try:
        client = build_client(args)
        where = build_where(args.tipo, args.proveedor)
        if args.modo == "auto":
            # Clase 9: el router elige la fuente y el RAG responde con lo que trajo.
            router = open_router(store, client)
            result = answer(client, router, args.rag, k=args.k or RAG_K, where=where, umbral=None)
        elif args.modo == "vector" and not (args.reescribir or args.rerank):
            # Clase 6: búsqueda vectorial y el umbral de distancia decide qué pasa.
            result = answer(client, store, args.rag, k=args.k or RAG_K, where=where, umbral=UMBRAL)
        else:
            # Clase 7: sin umbral de distancia; deciden el reranker o el modelo.
            retriever = Retriever(
                store,
                modo=base,
                client=client,
                reescribir=args.reescribir,
                rerank=args.rerank,
                grafo=grafo,
            )
            result = answer(
                client, retriever, args.rag, k=args.k or RAG_K, where=where, umbral=None
            )
    except ValidationError as e:
        return fail(config_error(e))
    except StructuredOutputError as e:
        return fail(f"el modelo no devolvió una respuesta válida. {e}")
    except AllProvidersFailedError as e:
        lines = "\n".join(f"  - {type(err).__name__}: {err}" for err in e.errors)
        return fail(f"ningún proveedor pudo responder.\n{lines}")
    except ProviderError as e:
        return fail(f"no se pudo calcular el embedding de la pregunta. {e}")

    print(result.respuesta.strip())
    if result.encontrado:
        print("\nFuentes:")
        for fuente in result.fuentes:
            print(f"  [{fuente['n']}] {fuente['etiqueta']}")
    if retriever:
        pasos = [args.modo, *(["reescritura"] if args.reescribir else [])]
        pasos += ["rerank"] if args.rerank else []
        consultas = " · ".join(f"«{c}»" for c in retriever.consultas)
        print(f"{GRAY}búsqueda: {' + '.join(pasos)} · consultas: {consultas}{RESET}")
    if router and router.ruta:
        fuentes = ", ".join(router.fuentes)
        print(
            f"{GRAY}ruta: {router.ruta.fuente} (decidió {router.decidio}) · "
            f"{router.ruta.motivo} · fuentes: {fuentes}{RESET}"
        )
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.rag is not None:
        return run_rag(args)

    try:
        text = find_ticket(args.ticket)["descripcion"] if args.ticket else args.prompt
    except LookupError as e:
        return fail(str(e))

    params: dict = {}
    log_extra: dict = {}
    if args.prompt_name:
        try:
            prompt = PromptKit(PROMPTS_DIR).get(args.prompt_name, args.prompt_version)
            variables = PROMPT_VARIABLES.get(prompt.name, {})
            messages = prompt.render(**variables, ticket=text)
        except (PromptNotFoundError, ValueError) as e:
            return fail(str(e))
        cli_params: dict = {}  # Flags de la CLI (si existen) tienen prioridad.
        params = {**prompt.model_params, **cli_params}
        log_extra = {"prompt_name": prompt.name, "prompt_version": prompt.version}
    else:
        try:
            prompt = PromptKit(PROMPTS_DIR).get("operator_assistant")
            messages = prompt.render(question=text or demo_prompt())
        except (PromptNotFoundError, ValueError) as e:
            return fail(str(e))
        params = dict(prompt.model_params)
        log_extra = {"prompt_name": prompt.name, "prompt_version": prompt.version}

    try:
        client = build_client(args)
    except ValidationError as e:
        return fail(config_error(e))
    except NotImplementedError as e:
        return fail(str(e))
    if args.falla_en:
        from core.fallas import ClienteConFalla

        client = ClienteConFalla(client, args.falla_en)

    if args.agent:
        return run_agent(
            client,
            args.agent,
            text,
            args.verbose,
            args.verify,
            mcp=args.mcp,
            rol=args.rol,
            docs=args.docs,
            embedder=args.embedder,
            grafo=args.grafo,
            sesion=args.sesion,
            memoria=args.memoria,
            ejecucion=args.ejecucion,
            reanudar=args.reanudar,
        )

    if args.structured or args.tools:
        try:
            return run_structured(client, text) if args.structured else run_tools(client, text)
        except StructuredOutputError as e:
            return fail(f"el modelo no devolvió un ticket válido. {e}")
        except MaxIterationsError as e:
            return fail(f"no hubo respuesta final. {e}")
        except AllProvidersFailedError as e:
            lines = "\n".join(f"  - {type(err).__name__}: {err}" for err in e.errors)
            return fail(f"ningún proveedor pudo responder.\n{lines}")

    try:
        response = client.generate(messages, log_extra=log_extra, **params)
    except AllProvidersFailedError as e:
        lines = "\n".join(f"  - {type(err).__name__}: {err}" for err in e.errors)
        return fail(f"ningún proveedor pudo responder.\n{lines}")

    print(response.text.strip())
    print(
        f"{GRAY}{response.provider} · {response.model} · "
        f"{response.tokens_in}/{response.tokens_out} tokens · "
        f"{response.latency_ms:.0f} ms · ${response.cost_usd:.6f}{RESET}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
