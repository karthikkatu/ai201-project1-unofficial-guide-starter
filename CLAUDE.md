# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A RAG pipeline for apartment resident reviews sourced from ApartmentRatings.com. Users ask natural-language questions and receive grounded answers backed by real resident reviews. The LLM for generation is Groq; the vector store is ChromaDB; the embedding model is `all-MiniLM-L6-v2` from sentence-transformers.

## Environment Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # then fill in GROQ_API_KEY
```

## Data Pipeline (run in order)

```bash
# 1. Flatten all per-apartment review files into one array
python scripts/flatten_reviews.py
# → data/all_reviews_flattened.json

# 2. Clean HTML entities, normalize whitespace, deduplicate sentences
python scripts/clean_data.py
# → data/final_data.json  (this is the canonical input for embedding)
```

`data/final_data.json` is the cleaned, flattened source used for all downstream steps. Never re-run the pipeline against already-cleaned data.

## Data Schema

Each record in `data/final_data.json` (see `data/sample_reviews.json` for a concrete example):

```json
{
  "apartment_id": "480820018885282",
  "apartment_name": "Avana Tempe Apartments",
  "review_id": "REVIEW112828351",
  "author": "Current Resident 858529",
  "review_title": "",
  "review_text": "...",
  "response_text": "...",
  "grounds_rating": 3,
  "noise_rating": 4,
  "maintenance_rating": 5,
  "recommend_rating": 2,
  "safety_rating": 4,
  "neighborhood_rating": 4,
  "office_staff_rating": 5,
  "review_date": "12-31-2020"
}
```

## Chunking Strategy

One review = one chunk. Do not split reviews or merge multiple reviews. The chunk `text` field must be formatted as:

```
Apartment: {apartment_name}

Review Title: {review_title}

Review:
{review_text}

Management Response:
{response_text}
```

The chunk `metadata` must carry: `apartment_id`, `apartment_name`, `review_id`, `review_date`, and all six `*_rating` fields. Chunk objects are `{"text": "...", "metadata": {...}}`.

## RAG Architecture

```
data/*/reviews.json
        ↓ scripts/flatten_reviews.py
data/all_reviews_flattened.json
        ↓ scripts/clean_data.py
data/final_data.json
        ↓ ingestion + chunking (one review = one chunk)
        ↓ all-MiniLM-L6-v2 embeddings (sentence-transformers)
        ↓ ChromaDB (persisted at ./vectorstore/)
        ↓ top-k=5 similarity retrieval
        ↓ Groq LLM (grounded generation)
Answer
```

## Key Constraints

- **Do not read `data/final_data.json` or `data/*/reviews.json` in full** — these are large files. Use `data/sample_reviews.json` to understand the schema.
- The `sample_reviews.json` filename has a trailing space in the filesystem — reference it carefully.
- ChromaDB persists to `./vectorstore/`; embedding must be re-run if the collection is deleted.
- Answers must be grounded: the LLM system prompt must instruct the model to answer only from retrieved review chunks and cite them.
- Groq API key is required for generation; embedding runs locally with no API key.
