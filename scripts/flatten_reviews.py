#!/usr/bin/env python3
"""
Read all JSON files matching /data/*/reviews.json, flatten the reviews,
and write one combined JSON array to /data/all_reviews_flattened.json.

Output format:
[
  {
    "apartment_id": "...",
    "apartment_name": "...",
    "review_id": "...",
    "author": "...",
    "review_title": "...",
    "review_text": "...",
    "response_text": "...",
    "grounds_rating": 2,
    "noise_rating": 1,
    "maintenance_rating": 2,
    "recommend_rating": 1,
    "safety_rating": 1,
    "neighborhood_rating": 1,
    "office_staff_rating": 1,
    "review_date": "MM-DD-YYYY"
  }
]
"""

import glob
import json
from pathlib import Path
from typing import Any, Dict, List, Optional


INPUT_PATTERN = "/Users/karthik/workspace/ai201-project1-unofficial-guide-starter/data/*/reviews.json"
OUTPUT_FILE = "/Users/karthik/workspace/ai201-project1-unofficial-guide-starter/data/all_reviews_flattened.json"


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def safe_get(d: Dict[str, Any], key: str, default: Any = "") -> Any:
    value = d.get(key, default)
    return default if value is None else value


def extract_apartment_info(payload: Any) -> Dict[str, Any]:
    """
    Supports a few common input shapes:
    1) {"apartment_id": ..., "apartment_name": ..., "reviews": [...]}
    2) {"reviews": [{"complexId": ..., "complexName": ..., ...}, ...]}
    3) list of review objects
    """
    if isinstance(payload, dict):
        apartment_id = payload.get("apartment_id") or payload.get("complexId") or payload.get("apartmentId") or ""
        apartment_name = payload.get("apartment_name") or payload.get("complexName") or payload.get("apartmentName") or ""
        return {
            "apartment_id": str(apartment_id),
            "apartment_name": str(apartment_name),
        }

    return {"apartment_id": "", "apartment_name": ""}


def normalize_review(review: Dict[str, Any], apartment_id: str, apartment_name: str) -> Dict[str, Any]:
    aspect = review.get("aspect_ratings") or {}

    return {
        "apartment_id": apartment_id,
        "apartment_name": apartment_name,
        "review_id": str(safe_get(review, "review_id", "")),
        "author": str(safe_get(review, "author", "")),
        "review_title": str(safe_get(review, "review_title", "")),
        "review_text": str(safe_get(review, "review_text", "")),
        "response_text": str(safe_get(review, "response_text", "")),
        "grounds_rating": aspect.get("grounds"),
        "noise_rating": aspect.get("noise"),
        "maintenance_rating": aspect.get("maintenance"),
        "recommend_rating": aspect.get("recommend"),
        "safety_rating": aspect.get("safety"),
        "neighborhood_rating": aspect.get("neighborhood"),
        "office_staff_rating": aspect.get("office_staff"),
        "review_date": str(safe_get(review, "review_date", "")),
    }


def iter_reviews(payload: Any) -> List[Dict[str, Any]]:
    """
    Returns a list of review dicts from the input payload.
    """
    if isinstance(payload, dict) and isinstance(payload.get("reviews"), list):
        return payload["reviews"]

    if isinstance(payload, list):
        return payload

    return []


def main() -> None:
    files = sorted(glob.glob(INPUT_PATTERN))
    if not files:
        raise SystemExit(f"No files found matching: {INPUT_PATTERN}")

    all_reviews: List[Dict[str, Any]] = []

    for file_path in files:
        try:
            payload = load_json(file_path)
        except Exception as e:
            print(f"Skipping unreadable file {file_path}: {e}")
            continue

        payload = payload[0]

        apt_info = extract_apartment_info(payload)
        apartment_id = apt_info["apartment_id"]
        apartment_name = apt_info["apartment_name"]

        reviews = iter_reviews(payload)

        # If the top-level object is a single apartment review file and apartment metadata is
        # missing at the top level, try to infer it from each review object.
        for review in reviews:
            if not apartment_id:
                apartment_id = str(review.get("apartment_id") or review.get("complexId") or "")
            if not apartment_name:
                apartment_name = str(review.get("apartment_name") or review.get("complexName") or "")

            if not apartment_id:
                # Skip reviews that cannot be tied to an apartment.
                continue

            all_reviews.append(normalize_review(review, apartment_id, apartment_name))

    # Optional de-duplication by review_id
    deduped: List[Dict[str, Any]] = []
    seen = set()
    for r in all_reviews:
        rid = r.get("review_id", "")
        if rid and rid in seen:
            continue
        if rid:
            seen.add(rid)
        deduped.append(r)

    Path(OUTPUT_FILE).parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(deduped, f, indent=2, ensure_ascii=False)

    print(f"Wrote {len(deduped)} reviews to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()