#!/usr/bin/env python3
"""
Milestone 5 — End-to-end query pipeline.

ask(question) retrieves relevant review chunks, sends them to the LLM for
grounded generation, and returns the answer plus programmatically extracted
source citations.
"""

import sys
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).parent))
from retrieval import retrieve
from llm import generate

TOP_K = 5


def _build_context(chunks: List[Dict[str, Any]]) -> str:
    """
    Number each retrieved chunk so the LLM can see how many sources it has.
    Headers are informational only; citations are extracted from metadata, not
    from the LLM response.
    """
    parts = []
    for i, chunk in enumerate(chunks, start=1):
        m = chunk["metadata"]
        header = (
            f"[Review {i} — {m.get('apartment_name', '')} | "
            f"{m.get('review_id', '')} | {m.get('review_date', '')}]"
        )
        parts.append(f"{header}\n{chunk['text']}")
    return "\n\n---\n\n".join(parts)


def _format_source(metadata: Dict[str, Any]) -> str:
    return (
        f"{metadata.get('apartment_name', 'Unknown')} — "
        f"{metadata.get('review_id', 'N/A')} "
        f"({metadata.get('review_date', 'N/A')})"
    )


def ask(question: str, k: int = TOP_K) -> Dict[str, Any]:
    """
    Full RAG pipeline: retrieve → generate → return answer + sources.

    Sources are extracted from chunk metadata before calling the LLM so
    attribution is guaranteed regardless of what the model produces.
    Split sub-chunks from the same review are deduplicated in the source list.

    Returns:
        {
            "answer":  str,
            "sources": List[str]   # "Apartment Name — REVIEW123 (MM-DD-YYYY)"
        }
    """
    chunks = retrieve(question, k=k)

    context = _build_context(chunks)
    answer = generate(context, question)

    # Programmatic attribution: deduplicate by review_id across split sub-chunks
    seen: set = set()
    sources: List[str] = []
    for chunk in chunks:
        rid = chunk["metadata"].get("review_id", "")
        if rid not in seen:
            seen.add(rid)
            sources.append(_format_source(chunk["metadata"]))

    return {"answer": answer, "sources": sources}
