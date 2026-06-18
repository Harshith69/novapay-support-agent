"""Build the NovaPay knowledge-base vector store.

Loads the four knowledge-base text files, splits them into overlapping chunks,
embeds them with a free CPU sentence-transformer, and persists them to a
ChromaDB collection. Idempotent: re-running rebuilds the collection cleanly.

The embedding function is shared with the retriever via
``get_embedding_function`` so indexing and querying always use the same model.

Run:  ``python rag/build_vectorstore.py``
"""
from __future__ import annotations

from pathlib import Path

from common.config import KNOWLEDGE_BASE_DIR, CHROMA_DIR, settings
from common.logging_utils import get_logger

logger = get_logger("build_vectorstore")

_embedding_fn = None


def get_embedding_function():
    """Return a cached Chroma-compatible sentence-transformer embedder."""
    global _embedding_fn
    if _embedding_fn is None:
        from chromadb.utils import embedding_functions

        _embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=settings.embedding_model
        )
    return _embedding_fn


def _get_client():
    import chromadb

    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(CHROMA_DIR))


def _chunk_documents() -> tuple[list[str], list[dict], list[str]]:
    """Read + split KB files. Returns (texts, metadatas, ids)."""
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    # Token-aware-ish splitter using a char≈token heuristic (4 chars/token).
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size * 4,
        chunk_overlap=settings.chunk_overlap * 4,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    texts: list[str] = []
    metadatas: list[dict] = []
    ids: list[str] = []

    kb_files = sorted(KNOWLEDGE_BASE_DIR.glob("*.txt"))
    if not kb_files:
        raise FileNotFoundError(
            f"No knowledge-base files found in {KNOWLEDGE_BASE_DIR}. "
            "Create the .txt policy docs first."
        )

    for path in kb_files:
        content = path.read_text(encoding="utf-8")
        chunks = splitter.split_text(content)
        for i, chunk in enumerate(chunks):
            texts.append(chunk)
            metadatas.append({"source": path.name, "chunk_index": i})
            ids.append(f"{path.stem}-{i}")
        logger.info("%-22s -> %d chunks", path.name, len(chunks))

    return texts, metadatas, ids


def build(reset: bool = True) -> int:
    """(Re)build the collection. Returns total chunks indexed."""
    client = _get_client()

    if reset:
        try:
            client.delete_collection(settings.chroma_collection)
        except Exception:
            pass  # collection may not exist yet

    collection = client.get_or_create_collection(
        name=settings.chroma_collection,
        embedding_function=get_embedding_function(),
        metadata={"hnsw:space": "cosine"},
    )

    texts, metadatas, ids = _chunk_documents()
    collection.add(documents=texts, metadatas=metadatas, ids=ids)

    total = collection.count()
    logger.info("Indexed %d chunks into '%s' at %s", total, settings.chroma_collection, CHROMA_DIR)
    return total


def ensure_built() -> None:
    """Build the store only if it is missing — used at app startup."""
    if not Path(CHROMA_DIR).exists() or not any(Path(CHROMA_DIR).iterdir()):
        logger.info("Vector store not found — building on first run…")
        build(reset=True)
    else:
        logger.info("Vector store already present at %s", CHROMA_DIR)


if __name__ == "__main__":
    build(reset=True)
