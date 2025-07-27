from __future__ import annotations

import os, time
from uuid import uuid4
from datetime import datetime
from typing import List, Dict, Any, Optional

import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, validator
from qdrant_client import QdrantClient, models

# ----------------- env & clients ----------------- #
QDRANT_URL     = os.getenv("QDRANT_URL", "http://qdrant:6333")
COLLECTION     = os.getenv("QDRANT_COLLECTION", "memories")
VECTOR_SIZE    = int(os.getenv("VECTOR_SIZE", "384"))
EMBED_URL      = os.getenv("EMBED_HEAD_URL", "http://embed_head:8000/embed")

qdrant = QdrantClient(url=QDRANT_URL)

def _ensure_collection():
    if COLLECTION not in {c.name for c in qdrant.get_collections().collections}:
        qdrant.recreate_collection(
            collection_name=COLLECTION,
            vectors_config=models.VectorParams(size=VECTOR_SIZE,
                                               distance=models.Distance.COSINE),
            on_disk_payload=True,
        )
        qdrant.create_payload_index(COLLECTION, "conversation_id", "keyword")
        qdrant.create_payload_index(COLLECTION, "ts", "integer")

_ensure_collection()

# ----------------- models ------------------------ #
class Item(BaseModel):
    text            : str
    conversation_id : str
    role            : str = "user"
    timestamp       : Optional[int] = None
    extra_payload   : Optional[Dict[str, Any]] = None

    @validator("timestamp", pre=True, always=True)
    def _ts(cls, v): return v or int(time.time())

class Batch(BaseModel):
    items: List[Item]

# ----------------- app --------------------------- #
app = FastAPI(title="Genio Ingestor")

@app.get("/ping")
def ping(): return {"status": "ingestor-alive"}

@app.post("/memory", status_code=201)
def ingest_one(item: Item):
    _store([item])
    return {"status": "ok"}

@app.post("/memory/batch", status_code=201)
def ingest_batch(batch: Batch):
    _store(batch.items)
    return {"status": "ok", "inserted": len(batch.items)}

# ----------------- helpers ----------------------- #
def _embed(texts: List[str]) -> List[List[float]]:
    # single‑sentence POST for now; you can batch‑encode later
    return [requests.post(EMBED_URL, json={"text": t}, timeout=30).json()["vector"]
            for t in texts]

def _store(items: List[Item]):
    try:
        vectors = _embed([i.text for i in items])
        points  = []
        for itm, vec in zip(items, vectors):
            payload = {
                "text": itm.text,
                "role": itm.role,
                "conversation_id": itm.conversation_id,
                "ts": itm.timestamp,
            }
            if itm.extra_payload: payload.update(itm.extra_payload)
            points.append(
                models.PointStruct(id=str(uuid4()), vector=vec, payload=payload)
            )
        qdrant.upsert(COLLECTION, points)
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))
