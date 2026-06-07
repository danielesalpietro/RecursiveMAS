"""
SemanticCache — thin client for the mem0 REST service.

Pre-model:  lookup(question, style, domain)  → answer str or None
Post-model: store(question, answer, style, domain, latency_ms)

The agent_id is scoped to (style, domain) so caches never cross-contaminate
different MAS configurations.

If the mem0 service is unreachable the client degrades silently — RecursiveMAS
continues without caching rather than crashing.
"""
from __future__ import annotations

import logging
import os
import time

import requests

log = logging.getLogger(__name__)

_MEM0_URL = os.getenv("MEM0_URL", "http://localhost:8080")
_THRESHOLD = float(os.getenv("MEM0_THRESHOLD", "0.92"))
_TIMEOUT = float(os.getenv("MEM0_TIMEOUT_S", "3.0"))

# Prefix stored in mem0 so lookup can extract the answer portion cleanly.
_Q_PREFIX = "Q: "
_A_PREFIX = " | A: "


def _agent_id(style: str, domain: str) -> str:
    return f"recursivemas__{style}__{domain}"


def _parse_answer(memory: str) -> str:
    """Extract the answer from a stored 'Q: … | A: …' string."""
    idx = memory.find(_A_PREFIX)
    return memory[idx + len(_A_PREFIX):].strip() if idx != -1 else memory.strip()


def lookup(question: str, style: str, domain: str) -> str | None:
    """
    Search mem0 for a semantically similar past answer.
    Returns the cached answer string on hit, None on miss or service error.
    """
    try:
        t0 = time.perf_counter()
        resp = requests.post(
            f"{_MEM0_URL}/search",
            json={"query": question, "agent_id": _agent_id(style, domain), "limit": 1},
            timeout=_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()
        latency_ms = (time.perf_counter() - t0) * 1000

        results = data.get("results", [])
        if results and results[0]["score"] >= _THRESHOLD:
            answer = _parse_answer(results[0]["memory"])
            log.info(
                "[cache HIT]  score=%.3f  latency=%.1f ms  style=%s  domain=%s",
                results[0]["score"], latency_ms, style, domain,
            )
            return answer

        log.info(
            "[cache MISS] best_score=%.3f  latency=%.1f ms  style=%s  domain=%s",
            results[0]["score"] if results else 0.0, latency_ms, style, domain,
        )
        return None

    except Exception as exc:
        log.warning("[cache] mem0 unreachable, skipping lookup: %s", exc)
        return None


def store(
    question: str,
    answer: str,
    style: str,
    domain: str,
    inference_latency_ms: float = 0.0,
) -> None:
    """
    Persist a Q&A pair in mem0 after a successful inference.
    Fails silently if the service is unreachable.
    """
    content = f"{_Q_PREFIX}{question}{_A_PREFIX}{answer}"
    metadata = {
        "style": style,
        "domain": domain,
        "inference_latency_ms": round(inference_latency_ms, 1),
    }
    try:
        resp = requests.post(
            f"{_MEM0_URL}/store",
            json={"content": content, "agent_id": _agent_id(style, domain), "metadata": metadata},
            timeout=_TIMEOUT,
        )
        resp.raise_for_status()
        log.info("[cache] stored  style=%s  domain=%s", style, domain)
    except Exception as exc:
        log.warning("[cache] mem0 unreachable, skipping store: %s", exc)
