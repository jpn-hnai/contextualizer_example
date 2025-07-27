from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import torch

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

app = FastAPI(title="Genio Embed Head")

class Req(BaseModel):
    text: str

@app.post("/embed")
def embed(req: Req):
    try:
        vec = model.encode(req.text,
                           device=device,
                           normalize_embeddings=True,
                           convert_to_numpy=True).tolist()
        return {"vector": vec}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/ping")
def ping():
    return {"status": "embed-alive", "device": device}
