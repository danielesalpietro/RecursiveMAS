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

_MEM0_URL         = os.getenv("MEM0_URL", "http://localhost:8080")
_THRESHOLD        = float(os.getenv("MEM0_THRESHOLD",        "0.92"))
_ENRICH_THRESHOLD = float(os.getenv("MEM0_ENRICH_THRESHOLD", "0.75"))
_ENRICH_LIMIT     = int(os.getenv("MEM0_ENRICH_LIMIT",       "3"))
_TIMEOUT          = float(os.getenv("MEM0_TIMEOUT_S",         "3.0"))

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


def enrich(question: str, style: str, domain: str) -> tuple[str, int]:
    """Retrieve top-N semantically related Q&A pairs for context injection.

    Returns (context_block, hit_count).  context_block is an empty string when
    nothing relevant is found or the service is unreachable (fail-open).

    Only meaningful for text-based pipelines (sequential_text) where each agent
    reads the enriched prompt as plain text.  Threshold MEM0_ENRICH_THRESHOLD
    (default 0.75) is intentionally lower than the full-HIT threshold so that
    partial knowledge is surfaced even when an exact answer cannot be reused.
    """
    try:
        resp = requests.post(
            f"{_MEM0_URL}/search",
            json={"query": question, "agent_id": _agent_id(style, domain), "limit": _ENRICH_LIMIT},
            timeout=_TIMEOUT,
        )
        resp.raise_for_status()
        results = resp.json().get("results", [])
        relevant = [r for r in results if r["score"] >= _ENRICH_THRESHOLD]
        if not relevant:
            log.info("[enrich] no relevant context (style=%s domain=%s)", style, domain)
            return "", 0

        lines = ["[Prior knowledge from past reasoning sessions — use if relevant]"]
        for i, r in enumerate(relevant, 1):
            mem = r["memory"]
            sep = mem.find(_A_PREFIX)
            if sep != -1:
                q_text = mem[len(_Q_PREFIX):sep].strip()
                a_text = mem[sep + len(_A_PREFIX):].strip()
                lines.append(f"{i}. Q: {q_text}")
                lines.append(f"   A: {a_text}")
            else:
                lines.append(f"{i}. {mem.strip()}")
        lines.append("")

        log.info("[enrich] %d context(s) injected (style=%s domain=%s)", len(relevant), style, domain)
        return "\n".join(lines), len(relevant)

    except Exception as exc:
        log.warning("[enrich] mem0 unreachable, skipping enrichment: %s", exc)
        return "", 0
