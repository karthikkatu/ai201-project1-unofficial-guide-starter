#!/usr/bin/env python3
"""
Milestone 4 — Retrieval evaluation.

Runs three evaluation queries against the ChromaDB collection and prints
ranked results for manual inspection.

Requires the index to be built first:
    python scripts/embedding.py
Then run:
    python scripts/test_retrieval.py
"""

import sys
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).parent))
from retrieval import retrieve

QUERIES = [
    "What do residents say about maintenance response times?",
    "What do residents say about safety?",
    "What are the most common complaints about living here?",
]

TOP_K = 5
# Max characters of chunk text to print per result
TEXT_PREVIEW = 400


def run_evaluation(queries: List[str], k: int = TOP_K) -> None:
    for query in queries:
        print("\n" + "=" * 72)
        print(f"  QUERY: {query}")
        print("=" * 72)

        results = retrieve(query, k=k)

        for rank, result in enumerate(results, start=1):
            m = result["metadata"]
            text = result["text"]
            preview = text[:TEXT_PREVIEW] + ("…" if len(text) > TEXT_PREVIEW else "")

            print(f"\n  Rank {rank}  |  distance={result['distance']:.4f}")
            print(f"  Apartment : {m.get('apartment_name', 'N/A')}")
            print(f"  Review ID : {m.get('review_id', 'N/A')}  |  Date: {m.get('review_date', 'N/A')}")
            print(f"  Ratings   : "
                  f"maint={m.get('maintenance_rating')}  "
                  f"safety={m.get('safety_rating')}  "
                  f"noise={m.get('noise_rating')}  "
                  f"staff={m.get('office_staff_rating')}")
            print(f"  {'─' * 66}")
            for line in preview.splitlines():
                print(f"    {line}")
            print()


if __name__ == "__main__":
    run_evaluation(QUERIES)
