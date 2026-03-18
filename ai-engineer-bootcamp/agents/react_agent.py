"""
Agente ReAct: Reason + Act. Alterna entre Thought, Action y Observation.
Implementa el paradigma ReAct de Yao et al. (ICLR 2023).
"""

import logging
import os
import re

from dotenv import load_dotenv
from openai import OpenAI

from agents.tools import TOOLS_REGISTRY, execute_tool, parse_action

load_dotenv()

logger = logging.getLogger(__name__)

REACT_SYSTEM_PROMPT = """\
Eres un agente DocOps que responde preguntas sobre documentos internos de la empresa.

Resuelve las tareas intercalando Thought, Action y Observation paso a paso.

Herramientas disponibles:
- search_docs["query"]: Busca información en los documentos internos. Usa queries cortos y específicos.
- lookup["term"]: Busca un término específico dentro del último documento recuperado.
- Finish["respuesta"]: Termina con la respuesta final. Usar SOLO cuando tengas evidencia suficiente.

Reglas:
1. SIEMPRE genera un Thought antes de cada Action
2. Usa search_docs para encontrar información relevante
3. Usa lookup para filtrar dentro de resultados largos
4. Usa Finish SOLO cuando tengas evidencia suficiente de los documentos
5. Si una búsqueda no retorna resultados útiles, reformula con otros términos
6. Máximo {max_steps} pasos. Si no encuentras respuesta, usa Finish con lo que tengas.
7. Responde en español.

Formato estricto por paso:
Thought N: [tu razonamiento sobre qué hacer]
Action N: tool_name["argumento"]

Después de cada Action, recibirás:
Observation N: [resultado de la herramienta]

Ejemplo:
Question: ¿Cuál es la política de vacaciones?
Thought 1: Necesito buscar información sobre la política de vacaciones en los documentos internos.
Action 1: search_docs["política vacaciones días"]
Observation 1: [1] (politica_vacaciones.txt): Los empleados tienen derecho a 15 días hábiles de vacaciones al año...
Thought 2: Encontré la información. La política establece 15 días hábiles anuales.
Action 2: Finish["Según la política interna, los empleados tienen derecho a 15 días hábiles de vacaciones al año."]

Ejemplo 2 (búsqueda fallida):
Question: ¿Cómo se solicita equipo de cómputo nuevo?
Thought 1: Necesito buscar el proceso para solicitar equipo de cómputo.
Action 1: search_docs["solicitud equipo cómputo"]
Observation 1: No se encontraron documentos relevantes para: 'solicitud equipo cómputo'. Intenta reformular.
Thought 2: La búsqueda no encontró resultados. Voy a intentar con términos más generales.
Action 2: search_docs["proceso equipamiento hardware"]
Observation 2: [1] (manual_onboarding.txt): Para solicitar equipo nuevo, contacta al área de TI...
Thought 3: Ahora sí encontré información relevante sobre el proceso.
Action 3: Finish["Para solicitar equipo de cómputo nuevo, debes contactar al área de TI según el manual de onboarding."]
"""


class ReactAgent:
    def __init__(
        self,
        tools: dict | None = None,
        model: str = "gpt-oss-120b",
        max_steps: int = 8,
    ):
        self.tools = tools or TOOLS_REGISTRY
        self.model = model
        self.max_steps = max_steps
        self.client = OpenAI(
            base_url="https://api.groq.com/openai/v1",
            api_key=os.getenv("GROQ_API_KEY"),
            timeout=30.0,
        )

    def run(self, query: str) -> dict:
        steps: list[dict] = []
        trajectory = f"Question: {query}\n"
        system = REACT_SYSTEM_PROMPT.format(max_steps=self.max_steps)

        for step_num in range(1, self.max_steps + 1):
            # Detección de loops
            if self._detect_loop(trajectory):
                trajectory += (
                    f"Thought {step_num}: Estoy repitiendo acciones. "
                    "Debo reformular mi estrategia con términos diferentes.\n"
                )
                logger.warning(
                    "Step %d - Loop detected, injecting reformulation thought",
                    step_num,
                )

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
                steps.append(
                    {
                        "step": step_num,
                        "thought": "Error en LLM",
                        "action": "error",
                        "observation": str(e),
                        "success": False,
                    }
                )
                break

            thought, action_line = self._parse_react_output(raw, step_num)
            logger.info("Step %d - Thought: %s", step_num, thought[:100])
            logger.info("Step %d - Action: %s", step_num, action_line)

            tool_call = parse_action(action_line)

            # Finish
            if tool_call.tool == "Finish":
                steps.append(
                    {
                        "step": step_num,
                        "thought": thought,
                        "action": action_line,
                        "observation": tool_call.argument,
                        "success": True,
                    }
                )
                trajectory += (
                    f"Thought {step_num}: {thought}\n"
                    f"Action {step_num}: {action_line}\n"
                )
                logger.info("Step %d - Finished: %s", step_num, tool_call.argument[:100])
                return {
                    "answer": tool_call.argument,
                    "steps": steps,
                    "total_steps": step_num,
                    "trajectory": trajectory,
                }

            # Execute tool
            result = execute_tool(tool_call)
            observation = result.output

            steps.append(
                {
                    "step": step_num,
                    "thought": thought,
                    "action": action_line,
                    "observation": observation[:200],
                    "success": result.success,
                }
            )
            trajectory += (
                f"Thought {step_num}: {thought}\n"
                f"Action {step_num}: {action_line}\n"
                f"Observation {step_num}: {observation}\n"
            )
            logger.info("Step %d - Observation: %s", step_num, observation[:100])

        return {
            "answer": None,
            "steps": steps,
            "total_steps": len(steps),
            "trajectory": trajectory,
        }

    @staticmethod
    def _detect_loop(trajectory: str, window: int = 3) -> bool:
        """Detecta si las últimas `window` acciones son idénticas."""
        actions = re.findall(r"Action\s+\d+:\s*(.+)", trajectory)
        if len(actions) < window:
            return False
        return len(set(actions[-window:])) == 1

    @staticmethod
    def _parse_react_output(raw: str, step_num: int) -> tuple[str, str]:
        """Extrae Thought y Action de la salida del LLM."""
        # Thought
        thought_match = re.search(
            rf"Thought\s*{step_num}?\s*:\s*(.+?)(?=Action|$)", raw, re.DOTALL
        )
        thought = (
            thought_match.group(1).strip()
            if thought_match
            else "Sin razonamiento explícito"
        )

        # Action — try exact step, then any step
        action_match = re.search(rf"Action\s*{step_num}\s*:\s*(.+)", raw)
        if not action_match:
            action_match = re.search(r"Action\s*\d*\s*:\s*(.+)", raw)

        if action_match:
            return thought, action_match.group(1).strip()

        # Fallback: look for tool call pattern
        for line in reversed(raw.split("\n")):
            if re.search(r"\w+\s*[\[\(]", line):
                return thought, line.strip()

        return thought, f'Finish["No pude determinar la acción. Respuesta parcial: {raw[:200]}"]'
