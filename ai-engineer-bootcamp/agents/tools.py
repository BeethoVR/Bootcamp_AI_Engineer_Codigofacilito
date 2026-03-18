"""
Herramientas compartidas para los agentes DocOps.
Búsqueda en documentos (ChromaDB), lookup y parsing de acciones.
"""

import logging
import os
import re
from pathlib import Path

import chromadb
from pydantic import BaseModel

logger = logging.getLogger(__name__)


# ── Modelos Pydantic ──────────────────────────────────────────


class ToolCall(BaseModel):
    tool: str
    argument: str


class ToolResult(BaseModel):
    output: str
    success: bool
    source: str | None = None


# ── Estado interno ────────────────────────────────────────────

_last_search_context: str = ""
_collection_cache: chromadb.Collection | None = None

# ── ChromaDB setup ────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHROMA_DIR = str(PROJECT_ROOT / "chroma_db")
DATA_DIR = str(PROJECT_ROOT / "data")
COLLECTION_NAME = "agents_docs"


def _chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """Divide texto en chunks con solapamiento."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        if chunk.strip():
            chunks.append(chunk.strip())
        start = end - overlap
    return chunks


def _get_collection() -> chromadb.Collection:
    """Obtiene o crea la colección de ChromaDB para los agentes."""
    global _collection_cache
    if _collection_cache is not None:
        return _collection_cache

    client = chromadb.PersistentClient(path=CHROMA_DIR)

    existing_names = [c.name for c in client.list_collections()]
    if COLLECTION_NAME in existing_names:
        _collection_cache = client.get_collection(COLLECTION_NAME)
        logger.info("Loaded existing collection '%s'", COLLECTION_NAME)
        return _collection_cache

    logger.info("Creating new ChromaDB collection '%s'", COLLECTION_NAME)
    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    data_path = Path(DATA_DIR)
    documents: list[str] = []
    metadatas: list[dict] = []
    ids: list[str] = []

    for txt_file in sorted(data_path.glob("*.txt")):
        content = txt_file.read_text(encoding="utf-8")
        chunks = _chunk_text(content, chunk_size=500, overlap=50)
        for i, chunk in enumerate(chunks):
            documents.append(chunk)
            metadatas.append({"source": txt_file.name})
            ids.append(f"{txt_file.stem}_{i}")

    if documents:
        collection.add(documents=documents, metadatas=metadatas, ids=ids)
        logger.info(
            "Indexed %d chunks from %d files",
            len(documents),
            len(list(data_path.glob("*.txt"))),
        )

    _collection_cache = collection
    return _collection_cache


# ── Herramientas ──────────────────────────────────────────────


def search_docs(query: str) -> str:
    """Busca documentos relevantes en ChromaDB."""
    global _last_search_context

    try:
        collection = _get_collection()
        results = collection.query(query_texts=[query], n_results=3)

        if not results["documents"] or not results["documents"][0]:
            return (
                f"No se encontraron documentos relevantes para: '{query}'. "
                "Intenta reformular."
            )

        formatted = []
        context_parts = []
        for i, (doc, meta) in enumerate(
            zip(results["documents"][0], results["metadatas"][0]), 1
        ):
            source = meta.get("source", "desconocido")
            formatted.append(f"[{i}] ({source}): {doc}")
            context_parts.append(doc)

        _last_search_context = "\n".join(context_parts)
        return "\n".join(formatted)

    except Exception as e:
        logger.error("Error en search_docs: %s", e)
        return f"Error al buscar documentos: {e}"


def lookup(term: str) -> str:
    """Busca un término específico en el último resultado de search_docs."""
    if not _last_search_context:
        return "No hay contexto previo. Usa search_docs primero."

    sentences = re.split(r"[.!?\n]+", _last_search_context)
    matches = [s.strip() for s in sentences if term.lower() in s.lower() and s.strip()]

    if matches:
        return " | ".join(matches[:3])
    return f"Término '{term}' no encontrado en el último resultado de búsqueda."


# ── Registro de herramientas ──────────────────────────────────

TOOLS_REGISTRY = {
    "search_docs": {
        "description": (
            "Busca información en los documentos internos de la empresa. "
            "Argumento: query de búsqueda."
        ),
        "function": search_docs,
    },
    "lookup": {
        "description": (
            "Busca un término específico dentro del último documento recuperado. "
            "Argumento: término a buscar."
        ),
        "function": lookup,
    },
    "Finish": {
        "description": (
            "Termina la ejecución con la respuesta final. "
            "Argumento: respuesta completa."
        ),
        "function": None,
    },
}


# ── Parsing y ejecución ──────────────────────────────────────


def parse_action(text: str) -> ToolCall:
    """Parsea texto de acción a un ToolCall."""
    pattern = r'(\w+)\s*[\[\(]\s*["\']?(.*?)["\']?\s*[\]\)]'
    match = re.search(pattern, text, re.DOTALL)

    if match:
        return ToolCall(tool=match.group(1), argument=match.group(2))

    return ToolCall(tool="error", argument=f"No se pudo parsear la acción: {text}")


def execute_tool(action: ToolCall) -> ToolResult:
    """Ejecuta una herramienta y retorna el resultado."""
    if action.tool == "error":
        return ToolResult(output=action.argument, success=False, source="parser")

    if action.tool not in TOOLS_REGISTRY:
        available = list(TOOLS_REGISTRY.keys())
        return ToolResult(
            output=f"Herramienta '{action.tool}' no encontrada. Disponibles: {available}",
            success=False,
            source=action.tool,
        )

    func = TOOLS_REGISTRY[action.tool]["function"]

    if func is None:
        return ToolResult(output=action.argument, success=True, source=action.tool)

    try:
        result = func(action.argument)
        return ToolResult(output=result, success=True, source=action.tool)
    except Exception as e:
        logger.error("Error executing tool '%s': %s", action.tool, e)
        return ToolResult(
            output=f"Error ejecutando {action.tool}: {e}",
            success=False,
            source=action.tool,
        )
