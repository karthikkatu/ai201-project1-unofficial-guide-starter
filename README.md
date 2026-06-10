# The Unofficial Guide — Project 1

## Project Overview

A retrieval-augmented generation (RAG) system that answers natural-language questions about resident experiences at apartment communities in the Tempe/Scottsdale area. Users ask questions like "What do residents say about maintenance?" and receive answers grounded exclusively in real resident reviews, with sources cited programmatically.

**Stack:** sentence-transformers (`all-MiniLM-L6-v2`) · ChromaDB · Groq (`llama-3.3-70b-versatile`) · Gradio

---

## Domain

Apartment resident experiences and reviews, collected from ApartmentRatings.com for ten communities in the Tempe and Scottsdale, Arizona area.

This knowledge is valuable because official apartment websites present only marketing content. Real resident experiences — covering maintenance reliability, safety, noise levels, management responsiveness, pest control, and rent increases — are scattered across hundreds of individual reviews that no prospective renter has time to read manually. A RAG system makes this collective knowledge searchable through natural language.

---

## Document Sources

All reviews were collected from [ApartmentRatings.com](https://www.apartmentratings.com). Per-property page URLs were not retained during data collection; each property can be found by searching the apartment name on that site. The local file paths below are the authoritative source references for this project.

| # | Source | Site | Local file |
|---|--------|------|------------|
| 1 | Avana Tempe Apartments | ApartmentRatings.com | `./data/avana/reviews.json` |
| 2 | IMT Desert Palm Village | ApartmentRatings.com | `./data/dpv/reviews.json` |
| 3 | Elliot's Crossing | ApartmentRatings.com | `./data/elliot/reviews.json` |
| 4 | Finisterra On Grove | ApartmentRatings.com | `./data/finisterra/reviews.json` |
| 5 | Galleria Palms | ApartmentRatings.com | `./data/galleria/reviews.json` |
| 6 | MAA Fountainhead | ApartmentRatings.com | `./data/maa/reviews.json` |
| 7 | Onnix | ApartmentRatings.com | `./data/onnix/reviews.json` |
| 8 | Sentry Tempe | ApartmentRatings.com | `./data/sentry/reviews.json` |
| 9 | Scottsdale Gateway Apartments | ApartmentRatings.com | `./data/sga/reviews.json` |
| 10 | Studio 710 | ApartmentRatings.com | `./data/studio/reviews.json` |

---

## Architecture

```
data/*/reviews.json
        ↓ scripts/flatten_reviews.py
data/all_reviews_flattened.json
        ↓ scripts/clean_data.py
data/final_data.json
        ↓ scripts/chunk_reviews.py   (one review = one chunk)
        ↓ all-MiniLM-L6-v2 embeddings (sentence-transformers)
        ↓ scripts/embedding.py → ChromaDB (persisted at ./vectorstore/)
        ↓ top-k=5 similarity retrieval   scripts/retrieval.py
        ↓ scripts/llm.py → Groq llama-3.3-70b-versatile (grounded generation)
Answer + Sources   ←   scripts/query.py → app.py (Gradio UI)
```

**Script responsibilities:**

| Script | Stage |
|--------|-------|
| `scripts/flatten_reviews.py` | Merge all per-apartment JSON files into one flat array |
| `scripts/clean_data.py` | Decode HTML entities, normalize whitespace, deduplicate sentences |
| `scripts/chunk_reviews.py` | Format reviews into chunk objects; split reviews exceeding 254 tokens |
| `scripts/embedding.py` | Generate embeddings with all-MiniLM-L6-v2; persist to ChromaDB |
| `scripts/retrieval.py` | Embed query; return top-k nearest chunks with metadata and distance |
| `scripts/llm.py` | Call Groq with a strict grounding system prompt |
| `scripts/query.py` | End-to-end `ask()`: retrieve → generate → return answer + sources |
| `app.py` | Gradio web interface |

---

## Chunking Strategy

**Chunk size:** One review = one chunk. Reviews are not split by token count unless they exceed the embedding model's 256-token hard limit.

**Overlap:** Sentence-level overlap of 50 tokens is carried forward between sub-chunks only for reviews that require splitting (approximately 11% of the dataset).

**Why this fits the domain:** Each apartment review is already a focused, self-contained account of a single resident's experience. Splitting a review at an arbitrary character boundary would sever the semantic thread between a complaint and its context (e.g., "roach infestation" and the follow-up "pest control came three times with no result" would end up in different chunks, weakening retrieval relevance for both fragments).

**Chunk text format:**
```
Apartment: {apartment_name}

Review Title: {review_title}

Review:
{review_text}

Management Response:
{response_text}
```

**Metadata stored per chunk:** `apartment_id`, `apartment_name`, `review_id`, `review_date`, `grounds_rating`, `noise_rating`, `maintenance_rating`, `recommend_rating`, `safety_rating`, `neighborhood_rating`, `office_staff_rating`

**Final chunk count:** 3,136 chunks from 2,356 unique reviews (349 reviews required splitting into 1,129 sub-chunks; the remaining 2,007 reviews each produced exactly one chunk).

---

## Retrieval Approach

**Embedding model:** `all-MiniLM-L6-v2` (sentence-transformers) — a lightweight, fast model well-suited for semantic similarity on short-to-medium texts. Embeddings are 384-dimensional and normalized to unit length before storage and query.

**Vector store:** ChromaDB with `hnsw:space=cosine`. Cosine distance is used because normalized vectors make cosine similarity equivalent to dot-product similarity, and it is robust to differences in review length.

**Top-k:** 5. Five chunks gives the LLM enough cross-review context to synthesize a multi-perspective answer while remaining well within the model's usable context window — each review chunk averages ~160 tokens, so five chunks occupy roughly 800 of the available tokens before the question is even added.

**Distance scores:** Returned with every result. Lower cosine distance = more semantically similar. Typical retrieved chunks fall in the 0.35–0.55 range for specific topical queries.

**Production tradeoff reflection:** For a real deployment, the primary improvement would be upgrading the embedding model to one with stronger domain understanding of informal language (slang, typos, mixed sentiment). `all-MiniLM-L6-v2` performs well on clean text but can miss semantic nuance in colloquial resident writing. A re-ranker applied after the initial top-k retrieval would also help reduce the off-target results observed in evaluation (see Failure Analysis below).

---

## Grounded Generation

**System prompt grounding instruction:**

> You are an apartment review assistant.
>
> Answer using ONLY the provided review context.
>
> If the answer cannot be found in the provided reviews, respond exactly:
> 'I don't have enough information on that.'
>
> Do not use outside knowledge. Do not speculate. Do not infer facts that are not explicitly supported by the retrieved reviews.

**How grounding is enforced:** Temperature is set to `0.0`, which reduces output variation and makes the model less likely to deviate from prompt instructions. The context passed to the LLM is built exclusively from retrieved ChromaDB chunks — no external data reaches the model.

**How source attribution is surfaced:** Sources are extracted from chunk metadata *before* `generate()` is called. The LLM never controls citations; it cannot fabricate or omit one. Split sub-chunks from the same review are deduplicated by `review_id` so each review appears in the source list only once.

---

## Evaluation Results

Five questions from `planning.md` were run through the complete pipeline (`ask()` in `scripts/query.py`). Responses are summarized below. Four of five questions produced partially accurate answers — in each case because retrieval returned a non-representative sample of reviews, skewing either positive or negative depending on the vocabulary overlap between the query and review text rather than the topical balance in the full dataset.

| # | Question | Expected Answer | System Response (summarized) | Retrieval Quality | Response Accuracy |
|---|----------|-----------------|------------------------------|-------------------|-------------------|
| 1 | What do residents say about maintenance response times at Avana Tempe Apartments? | Mixed — quick and responsive in some reviews; delays, unresolved issues, and repeated follow-ups in others. | Reports slow maintenance: requests can take up to 3 months, promises are not kept, and work is often not completed on the first visit. Does not surface positive maintenance reviews. | Partially relevant — retrieved only negative maintenance reviews; positive maintenance reviews exist in the dataset but were not returned. | Partially accurate |
| 2 | What do residents say about noise levels at Avana Tempe Apartments? | Mixed — quiet and peaceful in some reviews; freeway noise, loud neighbors, and parties in others. | Correctly identifies mixed opinions: most reviews describe the complex as quiet, while one resident says it is "loud 24/7." Mentions highway noise as a specific source. | Relevant | Accurate |
| 3 | What do residents say about safety at Avana Tempe Apartments? | Mixed — some feel safe; others mention crime, unsafe neighbors, police activity, feeling unsafe at night. | Skews positive — four of five retrieved reviews describe feeling safe; one review mentions management not caring about resident safety. Crime and police activity concerns not surfaced. | Partially relevant — missing reviews that describe crime, break-ins, or police activity. | Partially accurate |
| 4 | How do residents describe the office staff at Avana Tempe Apartments? | Mixed — praised as friendly and professional by some; criticized for poor communication and rudeness by others. | Entirely positive — describes staff as personable, professional, attentive, and going above and beyond. No critical reviews retrieved. | Partially relevant — retrieval returned only positive staff reviews; critical reviews exist in the dataset. | Partially accurate |
| 5 | What are the most common complaints about living at Avana Tempe Apartments? | Maintenance, pest issues, hot water/AC problems, poor communication, noise, and safety concerns. | Identifies slow maintenance, unknowledgeable staff, lack of communication, and distant parking. Notes that most retrieved reviews are positive. Misses pest issues, hot water/AC problems, and safety concerns. | Partially relevant — retrieved a mix of positive and negative reviews instead of complaint-focused ones. | Partially accurate |

**Retrieval quality:** Relevant / Partially relevant / Off-target
**Response accuracy:** Accurate / Partially accurate / Inaccurate

---

## Failure Analysis

**Question that failed:** *How do residents describe the office staff at Avana Tempe Apartments?*

**Expected behavior:** The system should retrieve a representative sample of reviews discussing office staff — including both praise and criticism — and generate a balanced answer reflecting that residents have mixed opinions.

**Actual behavior:** All five retrieved chunks described the office staff positively (personable, professional, attentive). The generated answer contained no critical perspectives, even though the dataset includes reviews specifically criticizing staff for poor communication, rudeness, and unresolved issues.

**Root cause — retrieval stage:** The five retrieved chunks all had `office_staff_rating` of 4 or 5, with cosine distances of 0.19–0.25 — a tight cluster of high-similarity, uniformly positive reviews. The query "How do residents describe the office staff?" is semantically close to text that *describes* staff in expressive detail. Positive reviews tend to do exactly that (specific adjectives, named staff members, concrete praise). Critical reviews, by contrast, describe staff indirectly through complaints ("they never called back," "I had to go in person three times") — language that is semantically farther from the query phrase even though it is topically relevant.

This is not a grounding or generation failure. The LLM answered faithfully from the context it received. The failure is entirely upstream: the retrieval stage returned a non-representative sample because cosine similarity of "describe office staff" is structurally higher with descriptive positive reviews than with complaint-pattern negative ones.

**Potential improvement:** Adding a re-ranking step after the initial top-k retrieval (e.g., using a cross-encoder) would re-score candidates based on relevance to the full query intent rather than surface-level embedding similarity. Alternatively, querying with multiple phrasings ("complaints about office staff," "problems with management") and merging the result sets would produce a more representative sample. A metadata filter on `office_staff_rating <= 2` could also be used to explicitly surface low-rated staff reviews when the query is about criticism.

---

## Spec Reflection

**One way the planning spec helped guide implementation:**

The planning spec established the exact chunk text format before any code was written:

```
Apartment: {apartment_name}

Review Title: {review_title}

Review:
{review_text}

Management Response:
{response_text}
```

Having this format locked down in `planning.md` prevented a common mistake in RAG pipelines: stripping the metadata context that makes a retrieved chunk interpretable in isolation. When a chunk is surfaced during retrieval, the LLM needs to know which apartment the review is about — embedding that in the text itself (rather than relying solely on metadata fields) made every sub-chunk self-contained and grounded the generation correctly from day one.

**One way the final implementation diverged from the spec, and why:**

The spec listed `Chunk size: 500` and `Overlap: 50` as fixed parameters. The final implementation does not apply a fixed character or token split to every review. Instead, one review always becomes one chunk; the 500/50 window logic is invoked only as a fallback when a review exceeds the embedding model's 254-token limit (approximately 11% of reviews). The change was made because most apartment reviews are 100–300 words — well within the model's context window — and splitting a coherent review at an arbitrary boundary would fragment the semantic meaning of individual complaints. A resident describing a roach infestation followed by failed pest control attempts carries meaning as a single unit; splitting it at token 500 would produce two chunks that each lose the cause-and-effect relationship.

---

## AI Usage

**Instance 1 — Redesigning the chunking split to preserve structure**

- *What I gave the AI:* The chunk format from `planning.md`, the requirement that one review = one chunk with splitting only for oversized reviews, and the observed output from an early version of `chunk_reviews.py` showing that sub-chunks had malformed structure (the "Management Response:" section header appeared inline with the review body instead of on its own line).
- *What it produced:* A redesigned `chunk_record()` function that splits only `review_text` rather than the full formatted chunk string. The header and management response are re-attached to every sub-chunk as fixed prefix/suffix, so each window retains the structured format regardless of where the split falls.
- *What I changed or overrode:* The initial fix also exposed a token-budget overflow bug: the carry-forward overlap logic could produce a window with `carry_tokens + sentence_tokens > max_tokens` before the next check ran, resulting in a 295-token chunk that exceeded the model limit. The carry condition was tightened to `carry_tok + t + s_tok <= max_tokens` so the incoming sentence is factored into the overlap budget before any sentence is carried.

**Instance 2 — Improving context structure in `query.py`**

- *What I gave the AI:* The milestone 5 spec requiring grounded generation, and the initial version of `query.py` where `_build_context()` concatenated chunk texts with plain `---` separators and no identifying labels.
- *What it produced:* A revised `_build_context()` that prepends a numbered header to each chunk — `[Review 1 — Avana Tempe Apartments | REVIEW112398248 | 12-31-2020]` — before the chunk text. The intent was to let the LLM reference reviews by number in its answer.
- *What I changed or overrode:* The AI's initial header format embedded the full `apartment_id` (a 15-digit numeric string) in the header, making it noisy and token-wasteful. I stripped the `apartment_id` from the header and kept only `apartment_name`, `review_id`, and `review_date`, since those are the human-readable fields the LLM would actually use in a citation. The `apartment_id` is still available in the metadata for programmatic lookup but plays no role in the generated answer.

---

## Setup Instructions

```bash
# 1. Clone the repository and create a virtual environment
# Requires Python 3.9 or higher
python3 -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure your Groq API key
cp .env.example .env
# Edit .env and set: GROQ_API_KEY=your_key_here
# Free key at: https://console.groq.com
```

---

## Run Instructions

```bash
# Activate the virtual environment first
source .venv/bin/activate

# --- Data pipeline (run once, outputs already committed) ---

# Step 1: Flatten per-apartment review files
python scripts/flatten_reviews.py
# → data/all_reviews_flattened.json

# Step 2: Clean HTML, normalize whitespace
python scripts/clean_data.py
# → data/final_data.json

# --- Index building (run once, or after data changes) ---

python scripts/embedding.py
# → ./vectorstore/   (3,136 vectors)
# Use --force to wipe and rebuild: python scripts/embedding.py --force

# --- Launch the web interface ---

python app.py
# → http://localhost:7860

# --- Retrieval smoke test (no UI required) ---

python scripts/test_retrieval.py
```
