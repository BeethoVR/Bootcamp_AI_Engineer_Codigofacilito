"""
Agente Básico (Act-Only): decide qué herramienta usar y ejecuta acciones
sin razonamiento explícito. Patrón "Act-Only" de Yao et al. (2022).
"""

import logging
import os
import re

from dotenv import load_dotenv
from openai import OpenAI

from agents.tools import TOOLS_REGISTRY, execute_tool, parse_action

load_dotenv()

logger = logging.getLogger(__name__)

# ── Colores ANSI ──────────────────────────────────────────────
_BOLD = "\033[1m"
_DIM = "\033[2m"
_YELLOW = "\033[33m"       # Action
_MAGENTA = "\033[35m"      # Observation
_GREEN = "\033[32m"        # Respuesta
_RED = "\033[31m"          # Error
_RESET = "\033[0m"

SYSTEM_PROMPT = """\
Eres un agente DocOps que responde preguntas sobre documentos internos de la empresa.

Herramientas disponibles:
- search_docs["query"]: Busca información en los documentos internos.
- lookup["term"]: Busca un término específico dentro del último documento recuperado.
- Finish["respuesta"]: Termina con la respuesta final.

Reglas:
1. Responde SOLO con una acción por turno. No generes pensamientos ni explicaciones.
2. Usa search_docs para encontrar información relevante.
3. Usa lookup para filtrar dentro de resultados largos.
4. Usa Finish SOLO cuando tengas evidencia suficiente de los documentos.
5. Si no encuentras información útil, usa Finish indicando que no encontraste respuesta.
6. Máximo {max_steps} pasos.
7. Responde en español.

Formato estricto:
Action N: tool_name["argumento"]

Ejemplo:
Question: ¿Cuál es la política de vacaciones?
Action 1: search_docs["política vacaciones"]
Observation 1: [1] (politica_vacaciones.txt): Los empleados tienen derecho a 12 días hábiles...
Action 2: Finish["Según la política interna, los empleados tienen derecho a 12 días hábiles de vacaciones en su primer año."]
"""


class BasicAgent:
    def __init__(
        self,
        tools: dict | None = None,
        model: str = "openai/gpt-oss-120b",
        max_steps: int = 5,
    ):
        self.tools = tools or TOOLS_REGISTRY
        self.model = model
        self.max_steps = max_steps
        self.client = OpenAI(
            base_url="https://api.groq.com/openai/v1",
            api_key=os.getenv("GROQ_API_KEY"),
            timeout=30.0,
        )

    def run(self, query: str, verbose: bool = True) -> dict:
        steps: list[dict] = []
        trajectory = f"Question: {query}\n"
        system = SYSTEM_PROMPT.format(max_steps=self.max_steps)

        if verbose:
            print(f"\n{_BOLD}{'─' * 60}{_RESET}")
            print(f"{_BOLD}  AGENTE BÁSICO (Act-Only){_RESET}")
            print(f"{_DIM}  Query: {query}{_RESET}")
            print(f"{_BOLD}{'─' * 60}{_RESET}")

        for step_num in range(1, self.max_steps + 1):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": trajectory},
                    ],
                    temperature=0,
                    max_tokens=1024,
                )
                raw = response.choices[0].message.content.strip()
            except Exception as e:
                logger.error("Step %d - LLM error: %s", step_num, e)
                if verbose:
                    print(f"  {_RED}Error: {e}{_RESET}")
                steps.append(
                    {
                        "step": step_num,
                        "action": "error",
                        "observation": str(e),
                        "success": False,
                    }
                )
                break

            action_line = self._extract_action(raw, step_num)
            logger.info("Step %d - Action: %s", step_num, action_line)

            tool_call = parse_action(action_line)

            # Finish
            if tool_call.tool == "Finish":
                if verbose:
                    print(f"  {_YELLOW}Action {step_num}:{_RESET} {action_line[:70]}")
                    print(f"\n  {_GREEN}{_BOLD}Respuesta:{_RESET} {_GREEN}{tool_call.argument}{_RESET}")
                    print(f"  {_DIM}Total pasos: {step_num}{_RESET}\n")
                steps.append(
                    {
                        "step": step_num,
                        "action": action_line,
                        "observation": tool_call.argument,
                        "success": True,
                    }
                )
                trajectory += f"Action {step_num}: {action_line}\n"
                logger.info("Step %d - Finished: %s", step_num, tool_call.argument)
                return {
                    "answer": tool_call.argument,
                    "steps": steps,
                    "total_steps": step_num,
                }

            # Execute tool
            result = execute_tool(tool_call)
            observation = result.output

            if verbose:
                print(f"  {_YELLOW}Action {step_num}:{_RESET} {action_line[:70]}")
                print(f"  {_MAGENTA}Observation {step_num}:{_RESET} {_MAGENTA}{observation[:120]}{_RESET}")
                print()

            steps.append(
                {
                    "step": step_num,
                    "action": action_line,
                    "observation": observation[:200],
                    "success": result.success,
                }
            )
            trajectory += (
                f"Action {step_num}: {action_line}\n"
                f"Observation {step_num}: {observation}\n"
            )
            logger.info("Step %d - Observation: %s", step_num, observation[:100])

        if verbose:
            print(f"  {_RED}Max pasos alcanzados sin respuesta final.{_RESET}\n")

        return {"answer": None, "steps": steps, "total_steps": len(steps)}

    @staticmethod
    def _extract_action(raw: str, step_num: int) -> str:
        """Extrae la línea de acción de la respuesta del LLM."""
        # Strip markdown bold markers
        cleaned = re.sub(r"\*{1,2}", "", raw).strip()

        # Try exact step number
        match = re.search(rf"Action\s*{step_num}\s*:\s*(.+)", cleaned)
        if match:
            return match.group(1).strip()

        # Try any Action pattern
        match = re.search(r"Action\s*\d*\s*:\s*(.+)", cleaned)
        if match:
            return match.group(1).strip()

        # Fallback: line that looks like a tool call
        for line in cleaned.split("\n"):
            if re.search(r"\w+\s*[\[\(]", line):
                return line.strip()

        return cleaned
