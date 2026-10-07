# Agente Operador

Proyecto integrador del **Bootcamp AI Engineer con Python — 2ª Edición: Agentes de IA en Producción** (Código Facilito).

El Agente Operador es un asistente para el equipo de operaciones de una empresa mediana: lee tickets (fallas, solicitudes, proveedores, facturación), los resume, los clasifica y, con el avance del bootcamp, actúa sobre ellos con herramientas, conocimiento interno y controles de producción.

En la Clase 1 construimos el **núcleo** (`core/`): un cliente LLM que habla con varios proveedores mediante el patrón adaptador, con reintentos, fallback, costo por llamada y logging estructurado.

## Qué se construye en cada bloque

| Bloque | Clases | Qué agrega al Agente Operador |
|---|---|---|
| Fundamentos | 1–3 | Cliente LLM multi-proveedor, prompting, manejo de contexto y contratos de salida |
| Agentes | 4–5 | Ciclo de agente con herramientas para actuar sobre los tickets |
| Conocimiento | 6–9 | Búsqueda sobre documentación interna (RAG) y memoria |
| Orquestación | 10–11 | Flujos de varios pasos y coordinación entre agentes |
| Evaluación | 12–13 | Conjuntos de prueba y métricas de calidad |
| Producción | 14–17 | Streaming, observabilidad, costos y despliegue |

## Requisitos e instalación

- Python 3.11 o superior
- [uv](https://docs.astral.sh/uv/)

```bash
uv sync
cp .env.example .env
```

Llena tu `.env` con las llaves:

| Variable | Dónde obtenerla |
|---|---|
| `GEMINI_API_KEY` | Google AI Studio: <https://aistudio.google.com/apikey> (tiene free tier) |
| `GROQ_API_KEY` | Groq Console: <https://console.groq.com/keys> (tiene free tier) |
| — | Ollama corre local y no necesita llave: <https://ollama.com/download>, luego `ollama pull llama3.2` |

Si falta la llave de algún proveedor que está en `PRIMARY` o `FALLBACKS`, el programa falla al arrancar y te dice qué variable agregar.

## Cómo correr

```bash
# Pregunta al LLM (usa PRIMARY y, si falla, FALLBACKS)
uv run main.py "Resume este ticket: el portal de proveedores no carga desde las 9:00"

# Demo de la clase: solo Gemini, sin fallbacks, con el ticket T-1043.
# Funciona aunque GroqProvider siga con su TODO.
uv run main.py --demo

# Fuerza un proveedor (sin fallbacks)
uv run main.py --provider ollama "hola"

# Tests (no usan red)
uv run pytest -q

# Calidad de código
uv run ruff check .
uv run ruff format --check .

# Reporte de las llamadas registradas en logs/llm_calls.jsonl
uv run scripts/report.py
```

Cada llamada imprime la respuesta y una línea gris con `proveedor · modelo · tokens in/out · latencia · costo`, y agrega una línea JSON por intento a `logs/llm_calls.jsonl`.

Los tickets de ejemplo están en `data/tickets_ejemplo.jsonl`.

## Arquitectura de `core/`

```mermaid
flowchart LR
    main["main.py (CLI)"] --> client["LLMClient<br/>reintentos + fallback"]
    client --> gemini["GeminiProvider"]
    client --> groq["GroqProvider"]
    client --> ollama["OllamaProvider"]
    gemini & groq & ollama --> base["OpenAICompatibleProvider<br/>httpx → /chat/completions"]
    client --> logger["CallLogger"]
    logger --> jsonl[("logs/llm_calls.jsonl")]
    base -.-> pricing["pricing.cost_usd"]
    config["config.Settings (.env)"] -.-> client
```

| Archivo | Responsabilidad |
|---|---|
| `core/config.py` | `Settings` leída de `.env`; valida llaves al arrancar |
| `core/errors.py` | Errores transitorios (se reintentan) y permanentes (fallback directo) |
| `core/providers.py` | `Provider` (Protocol), adaptador base compatible con OpenAI y uno por proveedor |
| `core/llm_client.py` | `LLMResponse` y `LLMClient`: backoff 1 s → 2 s → 4 s + jitter, fallback en orden |
| `core/logger.py` | Una línea JSON por intento, sin llaves ni texto del prompt |
| `core/pricing.py` | Precio por millón de tokens y costo por llamada |

## Práctica de la Clase 1

| Paso | Tiempo | Qué hacer |
|---|---|---|
| 1 | 5 min | Clona el repo y corre `uv sync` |
| 2 | 5 min | Copia `.env.example` a `.env` y agrega tus llaves |
| 3 | 15 min | Implementa `GroqProvider` en `core/providers.py` (busca `TODO(clase-1)`). Prueba con `uv run main.py --provider groq "hola"` |
| 4 | 5 min | Fuerza el fallback: `GEMINI_API_KEY=invalida uv run main.py "hola"`. Revisa en `logs/llm_calls.jsonl` el intento fallido y el `fallback: true` |
| 5 | 20 min | Completa `scripts/report.py`: costo total, latencia p50/p95, % de fallback y llamadas por proveedor |
| Bonus | — | Agrega `--temperature` (0–2) a `main.py` y pásalo al proveedor |

## Clase 2: prompt engineering y context engineering

### Qué se agrega y por qué

- **Prompts versionados** (`prompting/`, `prompts/`): los prompts salen del código y viven en YAML con versión. Así se comparan versiones con datos, se registra en el log qué versión respondió y se cambia de versión sin tocar Python.
- **Contexto con presupuesto** (`context/`): el `ContextManager` reparte tokens por sección (system, ejemplos, historial, documentos, pregunta y salida reservada). Si algo no cabe, descarta lo de menor prioridad y más antiguo, y nunca recorta el system.
- **Prompt injection** (`data/tickets_ataque.jsonl`): tickets que intentan darle órdenes al clasificador. Los delimitadores de la v2 reducen el riesgo, pero no lo eliminan.

### Formato de un archivo de prompt

Cada prompt vive en `prompts/<nombre>/v<N>.yaml`:

```yaml
name: ticket_classifier        # igual al nombre de la carpeta
version: 2                     # igual al número del archivo (v2.yaml)
active: true                   # exactamente una versión activa por prompt
description: Clasificador con delimitadores y categoría "otro".
model_params:                  # se pasan tal cual al proveedor
  temperature: 0
  max_tokens: 10
system: |
  Eres el clasificador de tickets del equipo de operaciones de {{ company }}.
  ...
user: |
  <ticket>{{ ticket }}</ticket>
```

- Las plantillas son Jinja2 con `StrictUndefined`: si falta una variable, falla y dice cuál.
- `categories` y `examples` (de `examples.jsonl`, si existe en la carpeta) están disponibles en la plantilla sin pasarlos.
- `PromptKit().get("ticket_classifier")` devuelve la versión con `active: true`; `get("ticket_classifier", 1)` devuelve la v1 aunque no esté activa. Si hay cero o más de una activa, falla y dice qué archivos revisar.
- Categorías válidas: `falla`, `solicitud`, `proveedor`, `facturacion`, `otro`.

### Comandos

```bash
# Compara versiones sobre los 20 tickets etiquetados y los tickets de ataque
uv run scripts/compare_prompts.py ticket_classifier --versions 1 2

# Clasifica un ticket con la versión activa (o una específica)
uv run main.py --prompt ticket_classifier --ticket T-1099
uv run main.py --prompt ticket_classifier --prompt-version 1 --ticket T-1099

# Muestra cómo el ContextManager recorta un historial largo (no llama al LLM)
uv run scripts/context_demo.py
uv run scripts/context_demo.py --budget-history 120
```

`compare_prompts.py` y `main.py --prompt` usan solo el proveedor primario si la cadena de fallbacks no se puede armar (por ejemplo, mientras `GroqProvider` siga pendiente). Cada llamada queda en `logs/llm_calls.jsonl` con `prompt_name` y `prompt_version`.

### Práctica de la Clase 2

1. Migra el system prompt de `main.py` (busca `TODO(clase-2)`) a `prompts/operator_assistant/v1.yaml` y cárgalo con `PromptKit`.
2. Corre `uv run scripts/compare_prompts.py ticket_classifier --versions 1 2` y revisa qué tickets falla cada versión.
3. Escribe `prompts/ticket_classifier/v3.yaml` con few-shot usando `examples.jsonl` (disponible como `examples` en la plantilla) y supera a la v2. Recuerda: solo una versión puede tener `active: true`.
4. Comparte tus aciertos, tokens promedio, costo por 1,000 tickets y si tu v3 resiste `T-1099`.

> Los resultados varían por modelo y entre corridas. Con 20 tickets, cada ticket vale 5 puntos porcentuales: no saques conclusiones de una diferencia de uno o dos tickets.

Para ver la solución: `git switch solucion/clase-2`.

## Ver la solución

```bash
git switch solucion/clase-1
```

Para volver al inicio: `git switch main` (o `git checkout clase-1-inicio`).

## Avisos

- Gemini se consume con su [endpoint oficial compatible con OpenAI](https://ai.google.dev/gemini-api/docs/openai), que Google marca como **beta**. Desde junio de 2026 Google recomienda su API nativa, la [Interactions API](https://ai.google.dev/gemini-api/docs/interactions-overview), para proyectos nuevos. Aquí usamos la compatible con OpenAI para que Gemini, Groq y Ollama compartan un mismo adaptador base. Cambiar a la API nativa solo requiere otro adaptador en `core/providers.py`; el resto del código no se toca.
- Los precios en `core/pricing.py` son **ilustrativos**. Revisa la tabla vigente de cada proveedor antes de tomar decisiones con ellos.
- **Nunca subas tu `.env`** al repositorio. Ya está en `.gitignore`; si alguna vez lo subes por error, rota tus llaves.
