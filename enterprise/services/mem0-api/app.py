"""
Mem0 API  —  Personalized memory service for enterprise AI agents.

Each user has:
  - Semantic long-term memory   → Qdrant (vector search)
  - Recent-activity timeline    → Redis  (fast ordered list, last 50)

Endpoints:
  POST   /memory              → store a new memory
  POST   /memory/search       → semantic search
  GET    /memory/{uid}/recent → last N memories (timeline)
  DELETE /memory/{id}         → remove a memory
"""

from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timezone
from typing import Any

import httpx
import redis.asyncio as aioredis
from fastapi import FastAPI
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
QDRANT_HOST     = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT     = int(os.getenv("QDRANT_PORT", "6333"))
REDIS_URL       = os.getenv("REDIS_URL", "redis://localhost:6379/1")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")
LLM_MODEL       = os.getenv("LLM_MODEL", "llama3.2")
COLLECTION_NAME = "user_memories"
EMBEDDING_DIM   = 768
RECENT_LIMIT    = 50   # items kept in the Redis timeline per user

# ── Clients ────────────────────────────────────────────────────────────────────
qdrant = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

try:
    qdrant.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
    )
except Exception:
    pass

# ── FastAPI ────────────────────────────────────────────────────────────────────
app = FastAPI(title="Mem0 API", version="1.0.0", description="Personalized memory for enterprise AI agents")

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


# ── Schemas ────────────────────────────────────────────────────────────────────
class MemoryAddRequest(BaseModel):
    user_id: str
    content: str
    category: str = "general"          # e.g. preference, fact, context
    metadata: dict[str, Any] = Field(default_factory=dict)


class MemorySearchRequest(BaseModel):
    user_id: str
    query: str
    top_k: int = Field(default=5, ge=1, le=20)
    category: str | None = None


class MemoryItem(BaseModel):
    memory_id: str
    user_id: str
    content: str
    category: str
    created_at: str
    score: float | None = None


class ConsolidateRequest(BaseModel):
    user_id: str


# ── Helpers ────────────────────────────────────────────────────────────────────
async def _embed(text: str) -> list[float]:
    async with httpx.AsyncClient(timeout=60) as client:
        resp = await client.post(
            f"{OLLAMA_BASE_URL}/api/embeddings",
            json={"model": EMBEDDING_MODEL, "prompt": text},
        )
        resp.raise_for_status()
        return resp.json()["embedding"]


async def _get_redis() -> aioredis.Redis:
    return await aioredis.from_url(REDIS_URL, decode_responses=True)


# ── Endpoints ──────────────────────────────────────────────────────────────────
@app.post("/memory", status_code=201)
async def add_memory(req: MemoryAddRequest):
    """Embed and store a new memory for a user."""
    embedding = await _embed(req.content)
    mem_id = str(uuid.uuid4())
    ts = datetime.now(timezone.utc).isoformat()

    qdrant.upsert(
        collection_name=COLLECTION_NAME,
        points=[
            PointStruct(
                id=mem_id,
                vector=embedding,
                payload={
                    "user_id": req.user_id,
                    "content": req.content,
                    "category": req.category,
                    "created_at": ts,
                    **req.metadata,
                },
            )
        ],
    )

    redis = await _get_redis()
    await redis.lpush(
        f"mem:{req.user_id}:recent",
        json.dumps({"id": mem_id, "content": req.content, "category": req.category, "ts": ts}),
    )
    await redis.ltrim(f"mem:{req.user_id}:recent", 0, RECENT_LIMIT - 1)
    await redis.aclose()

    return {"memory_id": mem_id, "status": "stored", "created_at": ts}


@app.post("/memory/search")
async def search_memory(req: MemorySearchRequest):
    """Semantic search over a user's memories."""
    embedding = await _embed(req.query)

    conditions: list[FieldCondition] = [
        FieldCondition(key="user_id", match=MatchValue(value=req.user_id))
    ]
    if req.category:
        conditions.append(FieldCondition(key="category", match=MatchValue(value=req.category)))

    results = qdrant.search(
        collection_name=COLLECTION_NAME,
        query_vector=embedding,
        query_filter=Filter(must=conditions),
        limit=req.top_k,
        with_payload=True,
    )

    memories = [
        MemoryItem(
            memory_id=str(r.id),
            user_id=r.payload["user_id"],
            content=r.payload["content"],
            category=r.payload.get("category", "general"),
            created_at=r.payload.get("created_at", ""),
            score=round(r.score, 4),
        )
        for r in results
    ]
    return {"memories": [m.model_dump() for m in memories]}


@app.get("/memory/{user_id}/recent")
async def get_recent_memories(user_id: str, limit: int = 10):
    """Retrieve the most recent memories from Redis timeline."""
    redis = await _get_redis()
    items = await redis.lrange(f"mem:{user_id}:recent", 0, min(limit, RECENT_LIMIT) - 1)
    await redis.aclose()
    return {"memories": [json.loads(i) for i in items]}


@app.get("/memory/{user_id}/summary")
async def get_user_summary(user_id: str):
    """Return a short LLM-generated summary of the user's stored memories."""
    results = qdrant.scroll(
        collection_name=COLLECTION_NAME,
        scroll_filter=Filter(must=[FieldCondition(key="user_id", match=MatchValue(value=user_id))]),
        limit=20,
        with_payload=True,
    )[0]

    if not results:
        return {"summary": "No memories stored for this user."}

    memories_text = "\n".join(f"- {r.payload['content']}" for r in results)
    prompt = (
        f"Summarize the following user memories in 2-3 sentences to create a user profile:\n\n"
        f"{memories_text}\n\nProfile summary:"
    )

    async with httpx.AsyncClient(timeout=60) as client:
        resp = await client.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json={"model": LLM_MODEL, "prompt": prompt, "stream": False},
        )
        resp.raise_for_status()
        summary = resp.json()["response"]

    return {"user_id": user_id, "memory_count": len(results), "summary": summary}


@app.delete("/memory/{memory_id}")
async def delete_memory(memory_id: str):
    qdrant.delete(collection_name=COLLECTION_NAME, points_selector=[memory_id])
    return {"status": "deleted", "memory_id": memory_id}


@app.get("/health")
def health():
    return {"status": "ok", "embedding_model": EMBEDDING_MODEL}
