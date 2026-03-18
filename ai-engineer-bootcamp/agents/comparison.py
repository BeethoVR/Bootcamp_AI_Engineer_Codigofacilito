"""
Comparación: Pipeline vs Agente Básico vs Agente ReAct.
Ejecutar: python -m agents.comparison
"""

import logging
import os
import re
import time

from dotenv import load_dotenv
from openai import OpenAI

from agents.basic_agent import BasicAgent
from agents.react_agent import ReactAgent
from agents.tools import search_docs

load_dotenv()

logging.basicConfig(level=logging.WARNING)

# ── Colores ANSI ──────────────────────────────────────────────

BOLD = "\033[1m"
DIM = "\033[2m"
CYAN = "\033[36m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
MAGENTA = "\033[35m"
RED = "\033[31m"
RESET = "\033[0m"

# ── Pipeline RAG estático ─────────────────────────────────────

_pipeline_client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=os.getenv("GROQ_API_KEY"),
    timeout=30.0,
)


def run_pipeline(query: str) -> dict:
    """Pipeline RAG estático: retrieve → generate. Sin loops, sin decisiones."""
    context = search_docs(query)

    try:
        response = _pipeline_client.chat.completions.create(
            model="gpt-oss-120b",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Responde la pregunta basándote SOLO en el contexto proporcionado. "
                        "Si el contexto no contiene la respuesta, indica que no tienes "
                        "información suficiente. Responde en español."
                    ),
                },
                {
                    "role": "user",
                    "content": f"Contexto:\n{context}\n\nPregunta: {query}",
                },
            ],
            temperature=0,
            max_tokens=1024,
        )
        answer = response.choices[0].message.content.strip()
    except Exception as e:
        answer = f"Error: {e}"

    return {"answer": answer, "steps": 1, "approach": "pipeline"}


# ── Preguntas de prueba ───────────────────────────────────────

QUESTIONS = [
    # EASY — respuesta directa
    {
        "question": "¿Cuántos días de vacaciones tienen los empleados en su primer año?",
        "category": "FÁCIL",
        "keywords": ["12", "días", "primer"],
    },
    {
        "question": "¿Cuál es el horario de soporte técnico?",
        "category": "FÁCIL",
        "keywords": ["horario", "soporte", "lunes"],
    },
    # MEDIUM — requiere buscar en el documento correcto
    {
        "question": "¿Qué documentos necesito entregar en mi primer día de trabajo?",
        "category": "MEDIA",
        "keywords": ["primer día", "onboarding", "recepción"],
    },
    {
        "question": "¿Cómo reporto un problema técnico urgente?",
        "category": "MEDIA",
        "keywords": ["P1", "crítico", "soporte", "NovaHub", "ticket"],
    },
    # HARD — combinar info o reformular
    {
        "question": "Si soy nuevo empleado y mi laptop no funciona el primer día, ¿qué debo hacer?",
        "category": "DIFÍCIL",
        "keywords": ["soporte", "equipo", "TI", "ticket"],
    },
    {
        "question": "¿Puedo tomar vacaciones durante mi periodo de prueba?",
        "category": "DIFÍCIL",
        "keywords": ["prueba", "vacaciones", "90", "días"],
    },
    # UNANSWERABLE — no está en los documentos
    {
        "question": "¿Cuál es el salario promedio de los ingenieros?",
        "category": "SIN RESPUESTA",
        "keywords": [],
    },
    {
        "question": "¿La empresa tiene oficina en Barcelona?",
        "category": "SIN RESPUESTA",
        "keywords": [],
    },
]


# ── Evaluación heurística ─────────────────────────────────────


def _check_answer(answer: str | None, keywords: list[str]) -> bool:
    """Evalúa si la respuesta contiene las keywords esperadas."""
    if not answer:
        return False
    if not keywords:
        # Para preguntas sin respuesta, aceptar si dice que no tiene info
        lower = answer.lower()
        return any(
            phrase in lower
            for phrase in [
                "no ",
                "no se encontr",
                "no tengo",
                "no cuento",
                "no hay información",
                "no dispongo",
            ]
        )
    lower = answer.lower()
    return any(kw.lower() in lower for kw in keywords)


def _has_source(answer: str | None) -> bool:
    """Verifica si la respuesta menciona un documento fuente."""
    if not answer:
        return False
    sources = [
        "manual_onboarding",
        "politica_vacaciones",
        "proceso_soporte",
        "onboarding",
        "política",
        "soporte técnico",
    ]
    lower = answer.lower()
    return any(s.lower() in lower for s in sources)


def _count_reformulations(steps: list[dict]) -> int:
    """Cuenta cuántas veces se reformuló la búsqueda."""
    search_actions = []
    for s in steps:
        action = s.get("action", "")
        if "search_docs" in action:
            search_actions.append(action)
    return max(0, len(search_actions) - 1)


# ── Formato de salida ─────────────────────────────────────────


def _print_header(question: str, category: str):
    width = 64
    print()
    print(f"{BOLD}{CYAN}╔{'═' * width}╗{RESET}")
    print(f"{BOLD}{CYAN}║  [{category}] {question[:width - len(category) - 6]}{RESET}")
    print(f"{BOLD}{CYAN}╚{'═' * width}╝{RESET}")


def _print_pipeline(result: dict, elapsed: float):
    answer = result["answer"]
    print(f"\n{GREEN}┌─ PIPELINE ────────────────────────────────────────────────────┐{RESET}")
    for line in _wrap(answer, 60):
        print(f"{GREEN}│{RESET} {line}")
    print(f"{GREEN}│{RESET} {DIM}Pasos: {result['steps']} | Tiempo: {elapsed:.1f}s{RESET}")
    print(f"{GREEN}└──────────────────────────────────────────────────────────────┘{RESET}")


def _print_basic(result: dict, elapsed: float):
    print(f"\n{YELLOW}┌─ AGENTE BÁSICO (Act-Only) ────────────────────────────────────┐{RESET}")
    for s in result["steps"]:
        action = s["action"][:58]
        print(f"{YELLOW}│{RESET} {DIM}Action {s['step']}:{RESET} {action}")
        if s.get("observation") and s["action"] != s.get("observation", ""):
            obs = s["observation"][:55]
            print(f"{YELLOW}│{RESET}   {DIM}Obs {s['step']}: {obs}{RESET}")
    answer = result.get("answer") or "(sin respuesta)"
    print(f"{YELLOW}│{RESET}")
    print(f"{YELLOW}│{RESET} {BOLD}Respuesta:{RESET} ", end="")
    for i, line in enumerate(_wrap(answer, 50)):
        if i == 0:
            print(line)
        else:
            print(f"{YELLOW}│{RESET}            {line}")
    print(
        f"{YELLOW}│{RESET} {DIM}Pasos: {result['total_steps']} | Tiempo: {elapsed:.1f}s{RESET}"
    )
    print(f"{YELLOW}└──────────────────────────────────────────────────────────────┘{RESET}")


def _print_react(result: dict, elapsed: float):
    print(f"\n{MAGENTA}┌─ AGENTE REACT ────────────────────────────────────────────────┐{RESET}")
    for s in result["steps"]:
        thought = s.get("thought", "")[:55]
        action = s["action"][:58]
        print(f"{MAGENTA}│{RESET} {DIM}Thought {s['step']}:{RESET} {thought}")
        print(f"{MAGENTA}│{RESET} {DIM}Action  {s['step']}:{RESET} {action}")
        if s.get("observation") and s["action"] != s.get("observation", ""):
            obs = s["observation"][:55]
            print(f"{MAGENTA}│{RESET}   {DIM}Obs {s['step']}: {obs}{RESET}")
    answer = result.get("answer") or "(sin respuesta)"
    print(f"{MAGENTA}│{RESET}")
    print(f"{MAGENTA}│{RESET} {BOLD}Respuesta:{RESET} ", end="")
    for i, line in enumerate(_wrap(answer, 50)):
        if i == 0:
            print(line)
        else:
            print(f"{MAGENTA}│{RESET}            {line}")
    print(
        f"{MAGENTA}│{RESET} {DIM}Pasos: {result['total_steps']} | Tiempo: {elapsed:.1f}s{RESET}"
    )
    print(f"{MAGENTA}└──────────────────────────────────────────────────────────────┘{RESET}")


def _wrap(text: str, width: int) -> list[str]:
    """Simple line wrapper."""
    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        if len(current) + len(word) + 1 > width:
            lines.append(current)
            current = word
        else:
            current = f"{current} {word}" if current else word
    if current:
        lines.append(current)
    return lines or [""]


# ── Tabla resumen ─────────────────────────────────────────────


def _print_summary(metrics: dict):
    """Imprime la tabla resumen de la comparación."""
    p = metrics["pipeline"]
    b = metrics["basic"]
    r = metrics["react"]
    total = metrics["total"]

    def _winner(p_val, b_val, r_val, lower_is_better=True):
        vals = {"Pipeline": p_val, "Básico": b_val, "ReAct": r_val}
        if lower_is_better:
            return min(vals, key=vals.get)
        return max(vals, key=vals.get)

    rows = [
        (
            "Promedio pasos",
            f"{p['total_steps'] / total:.1f}",
            f"{b['total_steps'] / total:.1f}",
            f"{r['total_steps'] / total:.1f}",
            _winner(
                p["total_steps"] / total,
                b["total_steps"] / total,
                r["total_steps"] / total,
            ),
        ),
        (
            "Promedio tiempo",
            f"{p['total_time'] / total:.1f}s",
            f"{b['total_time'] / total:.1f}s",
            f"{r['total_time'] / total:.1f}s",
            _winner(p["total_time"], b["total_time"], r["total_time"]),
        ),
        (
            "Resp. correctas",
            f"{p['correct']}/{total}",
            f"{b['correct']}/{total}",
            f"{r['correct']}/{total}",
            _winner(p["correct"], b["correct"], r["correct"], lower_is_better=False),
        ),
        (
            "Resp. con fuente",
            f"{p['with_source']}/{total}",
            f"{b['with_source']}/{total}",
            f"{r['with_source']}/{total}",
            _winner(
                p["with_source"],
                b["with_source"],
                r["with_source"],
                lower_is_better=False,
            ),
        ),
        (
            "Trazabilidad",
            "Ninguna",
            "Parcial",
            "Completa",
            "ReAct",
        ),
        (
            "Reformulaciones",
            "0",
            str(b["reformulations"]),
            str(r["reformulations"]),
            _winner(
                0,
                b["reformulations"],
                r["reformulations"],
                lower_is_better=False,
            ),
        ),
    ]

    print(f"\n\n{BOLD}{'═' * 80}{RESET}")
    print(f"{BOLD}  TABLA RESUMEN{RESET}")
    print(f"{BOLD}{'═' * 80}{RESET}")

    hdr = f"{'Métrica':<20} {'Pipeline':>10} {'Ag. Básico':>12} {'Ag. ReAct':>12} {'Ganador':>14}"
    print(f"\n{BOLD}{hdr}{RESET}")
    print("─" * 72)

    for metric, pv, bv, rv, winner in rows:
        print(f"{metric:<20} {pv:>10} {bv:>12} {rv:>12} {BOLD}{winner:>14}{RESET}")

    print("─" * 72)


# ── Análisis ──────────────────────────────────────────────────


def _print_analysis():
    print(f"\n{BOLD}ANÁLISIS DE RESULTADOS{RESET}")
    print("=" * 60)
    print(
        f"""
{GREEN}1. PIPELINE:{RESET} Rápido y predecible, pero sin capacidad de
   adaptación. Si el retrieval falla, la respuesta falla.
   No hay forma de reformular o buscar alternativas.

{YELLOW}2. AGENTE BÁSICO (Act-Only):{RESET} Puede usar herramientas
   iterativamente, pero sin razonamiento explícito tiende
   a repetir búsquedas similares sin reformular. No puede
   explicar POR QUÉ tomó cada decisión.

{MAGENTA}3. AGENTE REACT:{RESET} Más lento pero significativamente más
   preciso. Puede reformular búsquedas fallidas, combinar
   información de múltiples fuentes, y cada decisión es
   trazable a un pensamiento explícito. Ideal para tareas
   complejas donde la precisión importa más que la velocidad.
"""
    )


# ── Main ──────────────────────────────────────────────────────


def main():
    print(f"\n{BOLD}{CYAN}{'═' * 70}{RESET}")
    print(f"{BOLD}{CYAN}  COMPARACIÓN: Pipeline vs Agente Básico vs Agente ReAct{RESET}")
    print(f"{BOLD}{CYAN}{'═' * 70}{RESET}")
    print(f"{DIM}  Modelo: gpt-oss-120b | Docs: data/*.txt | VectorDB: ChromaDB{RESET}")

    basic_agent = BasicAgent()
    react_agent = ReactAgent()

    metrics = {
        "pipeline": {
            "total_steps": 0,
            "total_time": 0.0,
            "correct": 0,
            "with_source": 0,
            "reformulations": 0,
        },
        "basic": {
            "total_steps": 0,
            "total_time": 0.0,
            "correct": 0,
            "with_source": 0,
            "reformulations": 0,
        },
        "react": {
            "total_steps": 0,
            "total_time": 0.0,
            "correct": 0,
            "with_source": 0,
            "reformulations": 0,
        },
        "total": len(QUESTIONS),
    }

    for q in QUESTIONS:
        question = q["question"]
        keywords = q["keywords"]
        category = q["category"]

        _print_header(question, category)

        # ── Pipeline ──
        t0 = time.time()
        p_result = run_pipeline(question)
        p_time = time.time() - t0
        _print_pipeline(p_result, p_time)

        metrics["pipeline"]["total_steps"] += p_result["steps"]
        metrics["pipeline"]["total_time"] += p_time
        if _check_answer(p_result["answer"], keywords):
            metrics["pipeline"]["correct"] += 1
        if _has_source(p_result["answer"]):
            metrics["pipeline"]["with_source"] += 1

        # ── Agente Básico ──
        t0 = time.time()
        b_result = basic_agent.run(question)
        b_time = time.time() - t0
        _print_basic(b_result, b_time)

        metrics["basic"]["total_steps"] += b_result["total_steps"]
        metrics["basic"]["total_time"] += b_time
        if _check_answer(b_result.get("answer"), keywords):
            metrics["basic"]["correct"] += 1
        if _has_source(b_result.get("answer")):
            metrics["basic"]["with_source"] += 1
        metrics["basic"]["reformulations"] += _count_reformulations(b_result["steps"])

        # ── Agente ReAct ──
        t0 = time.time()
        r_result = react_agent.run(question)
        r_time = time.time() - t0
        _print_react(r_result, r_time)

        metrics["react"]["total_steps"] += r_result["total_steps"]
        metrics["react"]["total_time"] += r_time
        if _check_answer(r_result.get("answer"), keywords):
            metrics["react"]["correct"] += 1
        if _has_source(r_result.get("answer")):
            metrics["react"]["with_source"] += 1
        metrics["react"]["reformulations"] += _count_reformulations(r_result["steps"])

    _print_summary(metrics)
    _print_analysis()


if __name__ == "__main__":
    main()
