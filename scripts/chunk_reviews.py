#!/usr/bin/env python3
"""
Milestone 3 — Document pipeline: load, clean, and chunk apartment reviews.

Reads data/final_data.json and returns a list of chunk objects ready for
embedding:
    {"text": "...", "metadata": {...}}

Strategy: one review = one chunk. Reviews whose formatted text exceeds the
all-MiniLM-L6-v2 limit of 256 tokens are split by splitting only review_text
at sentence boundaries; the apartment header and management response are kept
intact and re-attached to every sub-chunk so each remains self-contained.
"""

import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).parent))
from clean_data import clean_text

DATA_FILE = Path(__file__).parent.parent / "data" / "final_data.json"

# all-MiniLM-L6-v2 hard limit; 2 slots reserved for [CLS] / [SEP]
MAX_TOKENS = 254
OVERLAP_TOKENS = 50

_tokenizer = None


def _get_tokenizer():
    global _tokenizer
    if _tokenizer is None:
        from transformers import AutoTokenizer
        _tokenizer = AutoTokenizer.from_pretrained(
            "sentence-transformers/all-MiniLM-L6-v2",
            clean_up_tokenization_spaces=True,
        )
    return _tokenizer


def _tok(text: str) -> int:
    return len(_get_tokenizer().tokenize(text))


# ---------------------------------------------------------------------------
# Text splitting
# ---------------------------------------------------------------------------

def _split_into_windows(text: str, max_tokens: int, overlap_tokens: int) -> List[str]:
    """
    Split text into overlapping sentence-aligned windows each ≤ max_tokens.
    Sentences that alone exceed max_tokens are further split at word boundaries.
    """
    sentence_re = re.compile(r"(?<=[.!?])\s+")

    # Pre-fragment any sentence that alone exceeds the limit
    sentences: List[str] = []
    for raw in sentence_re.split(text.strip()):
        if _tok(raw) > max_tokens:
            words = raw.split()
            fragment: List[str] = []
            frag_tok = 0
            for word in words:
                wt = _tok(word)
                if frag_tok + wt > max_tokens and fragment:
                    sentences.append(" ".join(fragment))
                    fragment, frag_tok = [word], wt
                else:
                    fragment.append(word)
                    frag_tok += wt
            if fragment:
                sentences.append(" ".join(fragment))
        else:
            sentences.append(raw)

    windows: List[str] = []
    current: List[str] = []
    current_tokens = 0

    for sentence in sentences:
        s_tok = _tok(sentence)
        if current_tokens + s_tok > max_tokens and current:
            windows.append(" ".join(current))
            # Only carry sentences whose combined token count leaves room for
            # the incoming sentence; this prevents the next window from
            # starting already over the limit.
            carry: List[str] = []
            carry_tok = 0
            for s in reversed(current):
                t = _tok(s)
                if carry_tok + t <= overlap_tokens and carry_tok + t + s_tok <= max_tokens:
                    carry.insert(0, s)
                    carry_tok += t
                else:
                    break
            current, current_tokens = carry, carry_tok
        current.append(sentence)
        current_tokens += s_tok

    if current:
        windows.append(" ".join(current))

    return windows


# ---------------------------------------------------------------------------
# Chunk construction
# ---------------------------------------------------------------------------

def _build_chunk_text(
    apartment_name: str,
    review_title: str,
    review_text: str,
    response_text: str,
) -> str:
    return (
        f"Apartment: {apartment_name}\n\n"
        f"Review Title: {review_title}\n\n"
        f"Review:\n{review_text}\n\n"
        f"Management Response:\n{response_text}"
    ).strip()


def _extract_metadata(record: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "apartment_id": record.get("apartment_id", ""),
        "apartment_name": record.get("apartment_name", ""),
        "review_id": record.get("review_id", ""),
        "review_date": record.get("review_date", ""),
        "grounds_rating": record.get("grounds_rating"),
        "noise_rating": record.get("noise_rating"),
        "maintenance_rating": record.get("maintenance_rating"),
        "recommend_rating": record.get("recommend_rating"),
        "safety_rating": record.get("safety_rating"),
        "neighborhood_rating": record.get("neighborhood_rating"),
        "office_staff_rating": record.get("office_staff_rating"),
    }


def chunk_record(record: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Convert a single review record into one or more chunk objects.

    Splitting strategy (only triggered when a review exceeds MAX_TOKENS):
      1. Compute how many tokens are already consumed by the header and
         management response so we know how many remain for review_text.
      2. If the management response alone is so long that < 30 tokens remain
         for review_text, drop the response from sub-chunks rather than
         producing near-empty windows.
      3. Split only review_text; the header and response are re-attached to
         every sub-chunk so each is self-contained.
    """
    apartment_name = record.get("apartment_name", "")
    review_title = clean_text(record.get("review_title", ""))
    review_text = clean_text(record.get("review_text", ""))
    response_text = clean_text(record.get("response_text", ""))

    full_text = _build_chunk_text(apartment_name, review_title, review_text, response_text)
    metadata = _extract_metadata(record)

    if _tok(full_text) <= MAX_TOKENS:
        return [{"text": full_text, "metadata": metadata}]

    # --- splitting needed ---
    header = (
        f"Apartment: {apartment_name}\n\n"
        f"Review Title: {review_title}\n\n"
        f"Review:\n"
    )
    footer = f"\n\nManagement Response:\n{response_text}" if response_text else ""

    review_budget = MAX_TOKENS - _tok(header) - _tok(footer)

    if review_budget < 30:
        # Response is extremely long; omit it from sub-chunks so review_text
        # gets reasonable window sizes. Metadata still carries all ratings.
        footer = ""
        review_budget = MAX_TOKENS - _tok(header)

    review_windows = _split_into_windows(review_text, review_budget, OVERLAP_TOKENS)

    chunks = []
    for i, window in enumerate(review_windows):
        text = (header + window + footer).strip()
        chunks.append(
            {
                "text": text,
                "metadata": {
                    **metadata,
                    "chunk_index": i,
                    "chunk_total": len(review_windows),
                },
            }
        )
    return chunks


def load_and_chunk(data_file: Path = DATA_FILE) -> List[Dict[str, Any]]:
    """Load final_data.json and return all chunks. Importable by later milestones."""
    with open(data_file, "r", encoding="utf-8") as f:
        records = json.load(f)

    chunks: List[Dict[str, Any]] = []
    for record in records:
        chunks.extend(chunk_record(record))
    return chunks


# ---------------------------------------------------------------------------
# CLI: print stats and 5 sample chunks
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print(f"Loading reviews from: {DATA_FILE}")
    all_chunks = load_and_chunk()

    split_chunks = [c for c in all_chunks if "chunk_index" in c["metadata"]]
    split_reviews = len({c["metadata"]["review_id"] for c in split_chunks})

    print(f"\nTotal chunks  : {len(all_chunks):,}")
    print(f"Split reviews : {split_reviews}  ({len(split_chunks)} sub-chunks from long reviews)")

    print("\n" + "=" * 70)
    print("SAMPLE CHUNKS (first 5)")
    print("=" * 70)

    for i, chunk in enumerate(all_chunks[:5]):
        m = chunk["metadata"]
        label = (
            f"sub-chunk {m['chunk_index'] + 1}/{m['chunk_total']}"
            if "chunk_index" in m
            else "single chunk"
        )
        print(f"\n{'─' * 70}")
        print(f"  Chunk #{i + 1}  [{label}]")
        print(f"{'─' * 70}")
        print(f"  Apartment : {m['apartment_name']}")
        print(f"  Review ID : {m['review_id']}")
        print(f"  Date      : {m['review_date']}")
        print(
            f"  Ratings   : grounds={m['grounds_rating']}  "
            f"noise={m['noise_rating']}  "
            f"maint={m['maintenance_rating']}  "
            f"safety={m['safety_rating']}  "
            f"staff={m['office_staff_rating']}"
        )
        print(f"  Tokens    : {_tok(chunk['text'])}")
        print(f"  Text ({len(chunk['text'])} chars):")
        print()
        for line in chunk["text"].splitlines():
            print(f"    {line}")
        print()
