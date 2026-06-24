"""
RAG API  —  Retrieval-Augmented Generation service.

Flow:
  POST /ingest  → embed text → store in Qdrant
  POST /query   → embed query → search Qdrant → augment prompt → call LLM
"""

from __future__ import annotations

import os
import uuid
from typing import Any

import httpx
import mlflow
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    VectorParams,
)

# ── Configuration ──────────────────────────────────────────────────────────────
QDRANT_HOST        = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT        = int(os.getenv("QDRANT_PORT", "6333"))
OLLAMA_BASE_URL    = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
LLM_MODEL          = os.getenv("LLM_MODEL", "llama3.2")
EMBEDDING_MODEL    = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")
MLFLOW_TRACKING    = os.getenv("MLFLOW_TRACKING_URI", "")
MEM0_API_URL       = os.getenv("MEM0_API_URL", "http://mem0-api:8001")
COLLECTION_NAME    = "enterprise_docs"
EMBEDDING_DIM      = 768   # nomic-embed-text output dimension

# ── Clients ────────────────────────────────────────────────────────────────────
qdrant = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

if MLFLOW_TRACKING:
    mlflow.set_tracking_uri(MLFLOW_TRACKING)

try:
    qdrant.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
    )
except Exception:
    pass  # collection already exists

# ── FastAPI app ────────────────────────────────────────────────────────────────
app = FastAPI(title="RAG API", version="1.0.0", description="Enterprise Retrieval-Augmented Generation service")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Schemas ────────────────────────────────────────────────────────────────────
class IngestRequest(BaseModel):
    text: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    doc_id: str | None = None


class IngestResponse(BaseModel):
    doc_id: str
    status: str


class QueryRequest(BaseModel):
    query: str
    top_k: int = Field(default=5, ge=1, le=20)
    user_id: str | None = None
    filters: dict[str, str] = Field(default_factory=dict)


class QueryResponse(BaseModel):
    answer: str
    sources: list[dict[str, Any]]
    model_used: str
    tokens_estimate: int


# ── Helpers ────────────────────────────────────────────────────────────────────
async def _embed(text: str) -> list[float]:
    async with httpx.AsyncClient(timeout=60) as client:
        resp = await client.post(
            f"{OLLAMA_BASE_URL}/api/embeddings",
            json={"model": EMBEDDING_MODEL, "prompt": text},
        )
        resp.raise_for_status()
        return resp.json()["embedding"]


async def _generate(prompt: str) -> str:
    async with httpx.AsyncClient(timeout=180) as client:
        resp = await client.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json={"model": LLM_MODEL, "prompt": prompt, "stream": False},
        )
        resp.raise_for_status()
        return resp.json()["response"]


async def _get_user_memories(user_id: str) -> str:
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.post(
                f"{MEM0_API_URL}/memory/search",
                json={"user_id": user_id, "query": "preferences context", "top_k": 3},
            )
            memories = resp.json().get("memories", [])
            if memories:
                return "\n".join(f"- {m['content']}" for m in memories)
    except Exception:
        pass
    return ""


# ── Endpoints ──────────────────────────────────────────────────────────────────
@app.post("/ingest", response_model=IngestResponse)
async def ingest_document(req: IngestRequest):
    """Embed a document chunk and store it in Qdrant."""
    embedding = await _embed(req.text)
    doc_id = req.doc_id or str(uuid.uuid4())
    qdrant.upsert(
        collection_name=COLLECTION_NAME,
        points=[
            PointStruct(
                id=doc_id,
                vector=embedding,
                payload={"text": req.text, **req.metadata},
            )
        ],
    )
    return IngestResponse(doc_id=doc_id, status="ingested")


@app.post("/query", response_model=QueryResponse)
async def query(req: QueryRequest):
    """RAG query: retrieve relevant chunks, augment prompt, generate answer."""
    embedding = await _embed(req.query)

    # Build optional payload filter
    qdrant_filter = None
    if req.filters:
        conditions = [
            FieldCondition(key=k, match=MatchValue(value=v))
            for k, v in req.filters.items()
        ]
        qdrant_filter = Filter(must=conditions)

    results = qdrant.search(
        collection_name=COLLECTION_NAME,
        query_vector=embedding,
        query_filter=qdrant_filter,
        limit=req.top_k,
        with_payload=True,
    )

    context = (
        "\n\n---\n\n".join(r.payload.get("text", "") for r in results)
        if results
        else "No relevant documents found in the knowledge base."
    )
    sources = [{"score": round(r.score, 4), **r.payload} for r in results]

    # Enrich with user memories when user_id is present
    memory_section = ""
    if req.user_id:
        memories = await _get_user_memories(req.user_id)
        if memories:
            memory_section = f"\n\nUser context/preferences:\n{memories}"

    prompt = (
        "You are an enterprise AI assistant. Answer the question using ONLY the provided "
        "context. If the context is insufficient, say so explicitly. Be concise and precise."
        f"{memory_section}\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {req.query}\n\nAnswer:"
    )

    if MLFLOW_TRACKING:
        with mlflow.start_run(run_name="rag_query", nested=True):
            mlflow.log_param("model", LLM_MODEL)
            mlflow.log_param("top_k", req.top_k)
            mlflow.log_metric("retrieved_docs", len(results))

    answer = await _generate(prompt)
    return QueryResponse(
        answer=answer,
        sources=sources,
        model_used=LLM_MODEL,
        tokens_estimate=len(prompt.split()),
    )


@app.delete("/document/{doc_id}")
async def delete_document(doc_id: str):
    qdrant.delete(collection_name=COLLECTION_NAME, points_selector=[doc_id])
    return {"status": "deleted", "doc_id": doc_id}


@app.get("/collections")
def list_collections():
    return {"collections": [c.name for c in qdrant.get_collections().collections]}


@app.get("/health")
def health():
    return {"status": "ok", "model": LLM_MODEL, "embedding_model": EMBEDDING_MODEL}
