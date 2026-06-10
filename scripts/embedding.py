#!/usr/bin/env python3
"""
Milestone 4 — Embedding pipeline.

Loads chunks from the Milestone 3 pipeline, generates embeddings with
all-MiniLM-L6-v2, and stores them in a persistent ChromaDB collection.

Run once to build the index:
    python scripts/embedding.py

The index is persisted at ./vectorstore/ and reused by retrieval.py.
Re-run with --force to wipe and rebuild from scratch.
"""

import sys
from pathlib import Path
from typing import Any, Dict, List

import chromadb
from sentence_transformers import SentenceTransformer

sys.path.insert(0, str(Path(__file__).parent))
from chunk_reviews import load_and_chunk

COLLECTION_NAME = "apartment_reviews"
VECTORSTORE_PATH = str(Path(__file__).parent.parent / "vectorstore")
EMBED_MODEL = "all-MiniLM-L6-v2"
# ChromaDB has a max batch size; 512 is safe and keeps memory reasonable
BATCH_SIZE = 512


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _chunk_id(chunk: Dict[str, Any], fallback_idx: int) -> str:
    """Deterministic ID: review_id or review_id_N for split sub-chunks."""
    m = chunk["metadata"]
    rid = m.get("review_id") or f"chunk_{fallback_idx}"
    ci = m.get("chunk_index")
    return f"{rid}_{ci}" if ci is not None else rid


def _sanitize_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
    """
    ChromaDB metadata values must be str, int, float, or bool — not None.
    Replace None ratings with -1 (sentinel for "not rated").
    """
    return {k: (v if v is not None else -1) for k, v in metadata.items()}


def get_client() -> chromadb.PersistentClient:
    return chromadb.PersistentClient(path=VECTORSTORE_PATH)


def get_collection() -> chromadb.Collection:
    """Return the existing collection (must already be built)."""
    client = get_client()
    return client.get_collection(name=COLLECTION_NAME)


# ---------------------------------------------------------------------------
# Index building
# ---------------------------------------------------------------------------

def build_index(force: bool = False) -> chromadb.Collection:
    """
    Load all chunks, embed them, and store in ChromaDB.

    force=True  — delete and rebuild even if the collection exists.
    force=False — skip silently if the collection is already populated.
    """
    client = get_client()

    # Handle existing collection
    existing_names = [c.name for c in client.list_collections()]
    if COLLECTION_NAME in existing_names:
        if not force:
            col = client.get_collection(COLLECTION_NAME)
            n = col.count()
            print(f"Collection '{COLLECTION_NAME}' already has {n:,} entries. "
                  "Use --force to rebuild.")
            return col
        print(f"Deleting existing collection '{COLLECTION_NAME}' …")
        client.delete_collection(COLLECTION_NAME)

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    print("Loading and chunking reviews …")
    chunks = load_and_chunk()
    print(f"  {len(chunks):,} chunks ready for embedding")

    print(f"Loading embedding model '{EMBED_MODEL}' …")
    model = SentenceTransformer(EMBED_MODEL)

    texts = [c["text"] for c in chunks]
    ids = [_chunk_id(c, i) for i, c in enumerate(chunks)]
    metadatas = [_sanitize_metadata(c["metadata"]) for c in chunks]

    max_batch = client.get_max_batch_size()
    batch_size = min(BATCH_SIZE, max_batch)

    print(f"Embedding and storing in batches of {batch_size} …")
    total = len(chunks)
    stored = 0

    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        batch_texts = texts[start:end]
        batch_ids = ids[start:end]
        batch_meta = metadatas[start:end]

        embeddings = model.encode(
            batch_texts,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).tolist()

        collection.add(
            ids=batch_ids,
            embeddings=embeddings,
            documents=batch_texts,
            metadatas=batch_meta,
        )
        stored += len(batch_ids)
        pct = stored / total * 100
        print(f"  {stored:,}/{total:,}  ({pct:.0f}%)", end="\r", flush=True)

    print(f"\nDone. {collection.count():,} vectors in collection '{COLLECTION_NAME}'.")
    print(f"Persisted at: {VECTORSTORE_PATH}")
    return collection


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    force = "--force" in sys.argv
    build_index(force=force)
