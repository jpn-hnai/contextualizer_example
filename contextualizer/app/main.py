from __future__ import annotations
from datetime import datetime
from typing import List

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .qdrant_search import top_k

app = FastAPI(title="Genio Contextualizer", version="0.2.0")

class ContextReq(BaseModel):
    conversation_id: str = Field(..., description="Well or chat ID")
    query_vector   : List[float] = Field(..., min_items=3)
    k              : int = Field(default=6, gt=0, le=30)

@app.get("/ping")
def ping():
    return {"status": "ctx-alive"}

@app.post("/context")
def context(req: ContextReq):
    try:
        hits = top_k(req.query_vector, req.conversation_id, req.k)
        return {
            "contexts": hits,
            "generated_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
