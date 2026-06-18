"""Query-time retrieval over the NovaPay knowledge base.

Exposes two functions used by every specialist agent:
- ``retrieve(query, top_k)`` -> ranked chunks with a 0–1 similarity score;
- ``format_context(results)`` -> a prompt-ready context block.

The Chroma collection is opened once and cached, so repeated queries in the
same process do not pay reload cost.
"""
from __future__ import annotations

from functools import lru_cache

from common.config import CHROMA_DIR, settings
from common.logging_utils import get_logger
from rag.build_vectorstore import get_embedding_function

logger = get_logger("retriever")


@lru_cache(maxsize=1)
def _get_collection():
    import chromadb

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return client.get_collection(
        name=settings.chroma_collection,
        embedding_function=get_embedding_function(),
    )


def _distance_to_similarity(distance: float) -> float:
    """Convert Chroma cosine distance (0=identical) to a 0–1 similarity."""
    return max(0.0, min(1.0, 1.0 - distance))


def retrieve(query: str, top_k: int = settings.default_top_k) -> list[dict]:
    """Return the ``top_k`` most relevant KB chunks for ``query``.

    Each result: {text, source, relevance_score}. Returns [] on any failure
    (e.g. store not built) so callers degrade gracefully.
    """
    query = (query or "").strip()
    if not query:
        return []

    try:
        collection = _get_collection()
        res = collection.query(
            query_texts=[query],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
    except Exception as exc:
        logger.error("Retrieval failed: %s", exc)
        return []

    docs = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]
    dists = res.get("distances", [[]])[0]

    results = []
    for text, meta, dist in zip(docs, metas, dists):
        results.append({
            "text": text,
            "source": (meta or {}).get("source", "unknown"),
            "relevance_score": round(_distance_to_similarity(dist), 4),
        })
    return results


def format_context(results: list[dict]) -> str:
    """Format retrieved chunks for injection into an agent prompt."""
    if not results:
        return "(no relevant policy context found)"
    return "\n".join(
        f"SOURCE [{r['source']}]: {r['text'].strip()}" for r in results
    )


if __name__ == "__main__":
    import json

    demo = retrieve("My UPI payment failed but money was debited", top_k=3)
    print(json.dumps(demo, indent=2, ensure_ascii=False))
    print("\n--- formatted ---\n")
    print(format_context(demo))
