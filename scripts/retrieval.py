#!/usr/bin/env python3
"""
Milestone 4 — Retrieval.

Provides retrieve(), which embeds a query with all-MiniLM-L6-v2 and returns
the top-k most relevant chunks from the ChromaDB collection.

Requires the index to be built first:
    python scripts/embedding.py
"""

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from sentence_transformers import SentenceTransformer

sys.path.insert(0, str(Path(__file__).parent))
from embedding import EMBED_MODEL, get_collection

_model: Optional[SentenceTransformer] = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBED_MODEL)
    return _model


def retrieve(query: str, k: int = 5) -> List[Dict[str, Any]]:
    """
    Embed query and return the top-k nearest chunks from the collection.

    Each result dict:
        {
            "text":     str,
            "metadata": dict,   # apartment_id, review_id, ratings, …
            "distance": float,  # cosine distance (lower = more similar)
        }
    """
    model = _get_model()
    collection = get_collection()

    query_embedding = model.encode(
        query,
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=k,
        include=["documents", "metadatas", "distances"],
    )

    # results["*"] is a list-of-lists (one list per query); we have one query
    docs = results["documents"][0]
    metas = results["metadatas"][0]
    dists = results["distances"][0]

    return [
        {"text": doc, "metadata": meta, "distance": dist}
        for doc, meta, dist in zip(docs, metas, dists)
    ]
