"""
Thin wrapper around Qdrant that handles
• boot‑time collection creation & indexing
• embedding delegation to the embed_head
• CRUD helpers for memories
"""

from __future__ import annotations

import os
import time
from typing import List, Dict, Any

import requests
from qdrant_client import QdrantClient, models


# --------------------------------------------------------------------------- #
#  Environment & defaults                                                     #
# --------------------------------------------------------------------------- #

QDRANT_URL        = os.getenv("QDRANT_URL", "http://qdrant:6333")
COLLECTION        = os.getenv("QDRANT_COLLECTION", "memories")
VECTOR_SIZE       = int(os.getenv("VECTOR_SIZE", "768"))
EMBED_HEAD_URL    = os.getenv("EMBED_HEAD_URL", "http://embed_head:8000/embed")
DISTANCE          = models.Distance.COSINE
EMBED_TIMEOUT_SEC = 20
TOP_K_DEFAULT     = 6


# --------------------------------------------------------------------------- #
#  Wrapper class                                                              #
# --------------------------------------------------------------------------- #

class QdrantMemory:
    """High‑level helper used by FastAPI endpoints."""

    def __init__(self) -> None:
        self.client = QdrantClient(url=QDRANT_URL)
        self._ensure_collection()

    # -------------------------  bootstrap helpers  ------------------------- #

    def _ensure_collection(self) -> None:
        """Create the collection & payload indexes if they don’t exist yet."""

        existing = self.client.get_collections().collections
        if not any(c.name == COLLECTION for c in existing):
            self.client.recreate_collection(
                collection_name=COLLECTION,
                vectors_config=models.VectorParams(
                    size=VECTOR_SIZE,
                    distance=DISTANCE
                ),
                on_disk_payload=True
            )
            # index frequently‑queried payload keys
            self.client.create_payload_index(
                COLLECTION, field_name="conversation_id", field_schema="keyword"
            )
            self.client.create_payload_index(
                COLLECTION, field_name="ts", field_schema="integer"
            )

    # -----------------------------  embeddings  ---------------------------- #

    def embed(self, text: str) -> List[float]:
        """Forward to the existing embed_head service and return a vector."""
        resp = requests.post(
            EMBED_HEAD_URL,
            json={"text": text},
            timeout=EMBED_TIMEOUT_SEC,
        )
        resp.raise_for_status()
        return resp.json()["vector"]

    # ------------------------------  writes  ------------------------------- #

    def upsert_memory(
        self,
        text: str,
        conversation_id: str,
        role: str = "user",
        ts: int | None = None,
        extra_payload: Dict[str, Any] | None = None,
    ) -> None:
        """
        Store a chunk of text + its embedding.

        • role  —  "user" | "assistant" | "system" | "summary" …
        • ts    —  Unix epoch seconds; defaults to now
        """
        ts = ts or int(time.time())
        vec = self.embed(text)

        payload: Dict[str, Any] = {
            "text": text,
            "role": role,
            "conversation_id": conversation_id,
            "ts": ts,
        }
        if extra_payload:
            payload.update(extra_payload)

        self.client.upload_points(
            collection_name=COLLECTION,
            points=models.Batch(
                ids=[f"{conversation_id}-{ts}-{role}-{hash(text)}"],
                vectors=[vec],
                payloads=[payload],
            ),
        )

    # ------------------------------  reads  -------------------------------- #

    def search(
        self,
        vector: List[float],
        conversation_id: str,
        limit: int = TOP_K_DEFAULT,
    ) -> List[Dict[str, Any]]:
        """
        Hybrid search: semantic similarity filtered by conversation_id
        and ordered by Qdrant’s native score.

        Returns a list of dicts with (text, ts, score, role).
        """
        hits = self.client.search(
            collection_name=COLLECTION,
            query_vector=vector,
            limit=limit,
            with_payload=True,
            filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="conversation_id",
                        match=models.MatchValue(value=conversation_id)
                    )
                ]
            ),
        )

        return [
            {
                "text": h.payload["text"],
                "role": h.payload["role"],
                "timestamp": h.payload["ts"],
                "score": h.score,
            }
            for h in hits
        ]
