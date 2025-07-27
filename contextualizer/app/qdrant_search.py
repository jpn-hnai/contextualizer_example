"""
Light‑weight helper for semantic search against Qdrant.
Call top_k(vector, conversation_id, k) to get the k most‑relevant memories.
"""

from __future__ import annotations
import os
from typing import List, Dict, Any

from qdrant_client import QdrantClient, models

# ---- Env ------------------------------------------------------------------ #

QDRANT_URL  = os.getenv("QDRANT_URL", "http://qdrant:6333")
COLLECTION  = os.getenv("QDRANT_COLLECTION", "memories")

_client: QdrantClient | None = None


def _client_lazy() -> QdrantClient:
    """Lazy‑init so the module imports fast."""
    global _client
    if _client is None:
        _client = QdrantClient(url=QDRANT_URL, prefer_grpc=False)
    return _client

# ---- Public API ----------------------------------------------------------- #

def top_k(
    vector: List[float],
    conversation_id: str,
    k: int = 6
) -> List[Dict[str, Any]]:
    """
    Semantic + boolean search:
      • vector ‑ cosine similarity
      • filter  ‑ exact match on conversation_id
    Returns the payload plus score for the top‑k hits.
    """
    hits = _client_lazy().search(
        collection_name=COLLECTION,
        query_vector=vector,
        query_filter=models.Filter(            # ← CORRECT PARAM NAME
            must=[
                models.FieldCondition(
                    key="conversation_id",
                    match=models.MatchValue(value=conversation_id)
                )
            ]
        ),
        limit=k,
        with_payload=True,
    )

    return [
        {
            "text": h.payload["text"],
            "role": h.payload.get("role", "unknown"),
            "timestamp": h.payload.get("ts"),
            "score": h.score,
        }
        for h in hits
    ]
