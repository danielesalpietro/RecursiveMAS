#!/usr/bin/env python3
"""
Lightweight mem0 REST service for RecursiveMAS semantic cache.

Exposes three endpoints:
  POST /search  — find semantically similar past answers
  POST /store   — persist a new Q&A pair
  GET  /health  — liveness probe

Vector store: Qdrant (self-hosted, separate container)
Embedder:     sentence-transformers/all-MiniLM-L6-v2 (local, no API key needed)
LLM:          optional — set OPENAI_API_KEY for mem0 memory extraction;
              without it mem0 stores raw text (still works for caching).
"""
from __future__ import annotations

import logging
import os
import time

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO, format="[mem0-service] %(message)s")
log = logging.getLogger(__name__)

# ── mem0 configuration ────────────────────────────────────────────────────────

_QDRANT_HOST = os.getenv("QDRANT_HOST", "qdrant")
_QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
_EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

_config: dict = {
    "vector_store": {
        "provider": "qdrant",
        "config": {
            "host": _QDRANT_HOST,
            "port": _QDRANT_PORT,
            "collection_name": "recursivemas_cache",
        },
    },
    "embedder": {
        "provider": "huggingface",
        "config": {"model": _EMBED_MODEL},
    },
}

if os.getenv("OPENAI_API_KEY"):
    _config["llm"] = {
        "provider": "openai",
        "config": {
            "api_key": os.getenv("OPENAI_API_KEY"),
            "model": os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        },
    }
    log.info("LLM memory extraction enabled (OpenAI)")
else:
    log.info("No OPENAI_API_KEY — mem0 will store raw text (no LLM extraction)")

from mem0 import Memory  # noqa: E402 — import after config is ready

_mem: Memory | None = None


def _get_mem() -> Memory:
    global _mem
    if _mem is None:
        log.info("Initialising mem0 Memory instance …")
        _mem = Memory.from_config(_config)
        log.info("mem0 ready.")
    return _mem


# ── FastAPI app ───────────────────────────────────────────────────────────────

app = FastAPI(title="RecursiveMAS mem0 Cache Service", version="1.0.0")


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
    content: str
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
        raw = _get_mem().search(req.query, agent_id=req.agent_id, limit=req.limit)
    except Exception as exc:
        log.error("search failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))

    results = []
    for item in raw:
        if isinstance(item, dict):
            results.append(
                SearchResult(
                    memory=item.get("memory", item.get("text", "")),
                    score=float(item.get("score", item.get("similarity", 0.0))),
                    metadata=item.get("metadata", {}),
                )
            )

    latency_ms = (time.perf_counter() - t0) * 1000
    log.info("search '%s…' → %d results (%.1f ms)", req.query[:60], len(results), latency_ms)
    return SearchResponse(results=results, latency_ms=latency_ms)


@app.post("/store", response_model=StoreResponse)
def store(req: StoreRequest) -> StoreResponse:
    t0 = time.perf_counter()
    try:
        _get_mem().add(req.content, agent_id=req.agent_id, metadata=req.metadata)
    except Exception as exc:
        log.error("store failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))

    latency_ms = (time.perf_counter() - t0) * 1000
    log.info("stored entry for agent_id='%s' (%.1f ms)", req.agent_id, latency_ms)
    return StoreResponse(status="ok", latency_ms=latency_ms)
