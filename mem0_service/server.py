#!/usr/bin/env python3
"""
Semantic cache REST service for RecursiveMAS.

Uses Qdrant + sentence-transformers directly — no LLM needed.
Every store() writes to Qdrant; every search() returns the most
similar past answer above the configured threshold.

Endpoints:
  POST /search  — find semantically similar past answers
  POST /store   — persist a new Q&A pair
  GET  /health  — liveness probe
"""
from __future__ import annotations

import logging
import os
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
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
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO, format="[mem0-service] %(message)s")
log = logging.getLogger(__name__)

# ── Configuration ─────────────────────────────────────────────────────────────

_QDRANT_HOST = os.getenv("QDRANT_HOST", "qdrant")
_QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
_EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
_EMBED_DIM   = int(os.getenv("EMBED_DIM", "384"))
_COLLECTION  = f"recursivemas_cache_{_EMBED_DIM}"

log.info("Qdrant collection: '%s'  embed_dim=%d", _COLLECTION, _EMBED_DIM)

# ── Clients ───────────────────────────────────────────────────────────────────

_qdrant: QdrantClient | None = None
_embedder: SentenceTransformer | None = None


def _get_qdrant() -> QdrantClient:
    global _qdrant
    if _qdrant is None:
        _qdrant = QdrantClient(host=_QDRANT_HOST, port=_QDRANT_PORT)
        _ensure_collection(_qdrant)
    return _qdrant


def _get_embedder() -> SentenceTransformer:
    global _embedder
    if _embedder is None:
        log.info("Loading embedder '%s' …", _EMBED_MODEL)
        _embedder = SentenceTransformer(_EMBED_MODEL)
        log.info("Embedder ready.")
    return _embedder


def _ensure_collection(client: QdrantClient) -> None:
    try:
        client.get_collection(_COLLECTION)
        log.info("Collection '%s' already exists.", _COLLECTION)
    except Exception:
        log.info("Creating collection '%s' …", _COLLECTION)
        client.create_collection(
            collection_name=_COLLECTION,
            vectors_config=VectorParams(size=_EMBED_DIM, distance=Distance.COSINE),
        )
        log.info("Collection '%s' created.", _COLLECTION)


# ── Startup: pre-load models so first request is instant ─────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("Startup: connecting to Qdrant and loading embedder …")
    _get_qdrant()
    _get_embedder()
    log.info("mem0 service ready — all components warm.")
    yield


# ── FastAPI app ───────────────────────────────────────────────────────────────

app = FastAPI(title="RecursiveMAS Semantic Cache", version="2.1.0", lifespan=lifespan)


class SearchRequest(BaseModel):
    query: str
    agent_id: str = "recursivemas"
    limit: int = Field(default=1, ge=1, le=10)


class SearchResult(BaseModel):
    memory: str
    score: float
    metadata: dict = {}


class SearchResponse(BaseModel):
    results: list[SearchResult]
    latency_ms: float


class StoreRequest(BaseModel):
    content: str           # full Q&A text stored in payload (for retrieval)
    query: str = ""        # text to embed for similarity search; falls back to content
    agent_id: str = "recursivemas"
    metadata: dict = {}


class StoreResponse(BaseModel):
    status: str
    latency_ms: float


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/search", response_model=SearchResponse)
def search(req: SearchRequest) -> SearchResponse:
    t0 = time.perf_counter()
    try:
        vector = _get_embedder().encode(req.query).tolist()
        response = _get_qdrant().query_points(
            collection_name=_COLLECTION,
            query=vector,
            query_filter=Filter(
                must=[FieldCondition(key="agent_id", match=MatchValue(value=req.agent_id))]
            ),
            limit=req.limit,
            with_payload=True,
        )
        hits = response.points
    except Exception as exc:
        log.error("search failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))

    results = [
        SearchResult(
            memory=h.payload.get("content", ""),
            score=h.score,
            metadata={k: v for k, v in h.payload.items() if k not in ("content", "agent_id")},
        )
        for h in hits
    ]
    latency_ms = (time.perf_counter() - t0) * 1000
    log.info("search '%s…' → %d results (%.1f ms)", req.query[:60], len(results), latency_ms)
    return SearchResponse(results=results, latency_ms=latency_ms)


@app.post("/store", response_model=StoreResponse)
def store(req: StoreRequest) -> StoreResponse:
    t0 = time.perf_counter()
    try:
        # Embed the question only (req.query), not the full Q&A string.
        # This ensures lookup(question) ≈ 1.0 for the same question regardless
        # of answer length. Falls back to content if query is not provided.
        embed_text = req.query if req.query.strip() else req.content
        vector = _get_embedder().encode(embed_text).tolist()
        point = PointStruct(
            id=str(uuid.uuid4()),
            vector=vector,
            payload={"content": req.content, "agent_id": req.agent_id, **req.metadata},
        )
        _get_qdrant().upsert(collection_name=_COLLECTION, points=[point])
    except Exception as exc:
        log.error("store failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))

    latency_ms = (time.perf_counter() - t0) * 1000
    log.info("stored  agent_id='%s'  embed_on='%s…'  (%.1f ms)",
             req.agent_id, embed_text[:40], latency_ms)
    return StoreResponse(status="ok", latency_ms=latency_ms)

