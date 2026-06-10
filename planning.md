# Project 1 Planning: The Unofficial Guide

## Domain

I chose the domain of apartment resident experiences and reviews, using data collected from ApartmentRatings.com. This domain contains firsthand accounts from current and former residents about apartment communities, including topics such as maintenance quality, safety, noise levels, management responsiveness, amenities, parking, rent increases, pest issues, and overall satisfaction.

This knowledge is valuable because official apartment websites and leasing offices typically highlight only positive features and marketing information, while real resident experiences are scattered across thousands of individual reviews. Prospective renters often struggle to find reliable information about recurring issues such as maintenance delays, safety concerns, hidden fees, noise problems, or management quality without manually reading hundreds of reviews. A retrieval-augmented system can make this collective resident knowledge searchable and accessible through natural language questions.

---

## Documents

| # | Source | Description | File location |
|---|--------|-------------|-----------------|
| 1 | Avana Tempe Apartments | Reviews of Avana Tempe Apartments from www.apartmentratings.com | ./data/avana/reviews.json |
| 2 | IMT Desert Palm Village | Reviews of IMT Desert Palm Village from www.apartmentratings.com | ./data/dpv/reviews.json |
| 3 | Elliot's Crossing | Reviews of Elliot's Crossing from www.apartmentratings.com | ./data/elliot/reviews.json |
| 4 | Finisterra On Grove | Reviews of Finisterra On Grove from www.apartmentratings.com | ./data/finisterra/reviews.json |
| 5 | Galleria Palms | Reviews of Galleria Palms from www.apartmentratings.com | ./data/galleria/reviews.json |
| 6 | MAA Fountainhead | Reviews of MAA Fountainhead from www.apartmentratings.com | ./data/maa/reviews.json |
| 7 | Onnix | Reviews of Onnix from www.apartmentratings.com | ./data/onnix/reviews.json |
| 8 | Sentry Tempe | Reviews of Sentry Tempe from www.apartmentratings.com | ./data/sentry/reviews.json |
| 9 | Scottsdale Gateway Apartments | Reviews of Scottsdale Gateway Apartments from www.apartmentratings.com | ./data/sga/reviews.json |
| 10 | Studio 710 | Reviews of Studio 710 from www.apartmentratings.com | ./data/studio/reviews.json |

---

## Chunking Strategy


- Each review becomes exactly one chunk.
- Combine review_title, review_text, and response_text into a single chunk.
- If it is very long, split it into chunks of about 500 tokens with 100 overlap.
- Do not split reviews by token count.
- Do not merge multiple reviews together.
- Keep apartment_id, apartment_name, review_id, review_date, and all rating fields as metadata.
- Generate chunk text in this format:

Apartment: {apartment_name}

Review Title: {review_title}

Review:
{review_text}

Management Response:
{response_text}

Output should be a list of chunks where each chunk contains:
{
  "text": "...",
  "metadata": {...}
}


**Chunk size:**500

**Overlap:**50

**Reasoning:**

Most chunking strategies exist because documents are too large (articles, PDFs, manuals, books). In this dataset, each review is already a focused piece of information discussing a specific experience:

maintenance problems
noise complaints
safety concerns
management responsiveness
amenities
location
rent increases

If we split a review into smaller chunks, we'll often lose context.

---

## Retrieval Approach

**Embedding model:** all-MiniLM-L6-v2 (sentence-transformers)

**Top-k:** 5

**Production tradeoff reflection:**
If I were deploying this for real users and cost were not a constraint, I would prioritize a stronger embedding model with better domain understanding over all-MiniLM-L6-v2. I would weigh context length, retrieval accuracy on long apartment reviews, and robustness to informal language, since resident reviews often include slang, noise, typos, and mixed sentiment. I would also consider multilingual support, latency, and indexing cost, but for this domain the biggest gain would likely come from a higher-quality model that captures maintenance, safety, noise, and management-related meaning more accurately than a lightweight baseline.

---

## Evaluation Plan

| # | Question | Expected answer |
|---|----------|-----------------|
| 1 | What do residents say about maintenance response times at Avana Tempe Apartments? | Residents say maintenance is often quick and responsive in some reviews, but other reviews mention delays, unresolved issues, and repeated follow-ups. |
| 2 | What do residents say about noise levels at Avana Tempe Apartments? | Opinions are mixed, but several reviews describe the community as quiet or peaceful, while others mention freeway noise, loud neighbors, or parties. |
| 3 | What do residents say about safety at Avana Tempe Apartments? | Some residents feel safe and comfortable, but other reviews mention concerns about crime, unsafe neighbors, police activity, or feeling unsafe at night. |
| 4 | How do residents describe the office staff at Avana Tempe Apartments? | Some reviews praise the staff as friendly, professional, and helpful, while others criticize them for poor communication, rudeness, or not handling issues well. |
| 5 | What are the most common complaints about living at Avana Tempe Apartments? | Common complaints include maintenance problems, pest issues, hot water or AC problems, poor communication, noise, and safety concerns. |

---

## Anticipated Challenges

1. Reviews contain noisy, subjective, and sometimes contradictory information. Different residents may report very different experiences about the same apartment community (e.g., maintenance, safety, or noise), making it difficult for the retrieval system to provide a single definitive answer.

2. Retrieval may return reviews that mention similar keywords but are not relevant to the user's question. For example, a query about maintenance quality could retrieve reviews that only briefly mention maintenance while primarily discussing safety or management, reducing answer quality.
---

## Architecture

+----------------------+
| Document Ingestion   |
|----------------------|
| Apartment Reviews    |
| JSON Files           |
| Python (json)        |
+----------+-----------+
           |
           v
+----------------------+
| Chunking             |
|----------------------|
| One Review =         |
| One Chunk            |
| Custom Python Logic  |
+----------+-----------+
           |
           v
+----------------------+
| Embedding            |
|----------------------|
| all-MiniLM-L6-v2     |
| sentence-transformers|
+----------+-----------+
           |
           v
+----------------------+
| Vector Store         |
|----------------------|
| ChromaDB             |
| Store Embeddings     |
+----------+-----------+
           |
           v
+----------------------+
| Retrieval            |
|----------------------|
| Similarity Search    |
| Top-k = 5            |
+----------+-----------+
           |
           v
+----------------------+
| Generation           |
|----------------------|
| LLM + Retrieved      |
| Reviews              |
| Final Answer         |
+----------------------+

---

## AI Tool Plan

### Milestone 3 — Ingestion and Chunking

I will use Claude Code to implement the ingestion and chunking pipeline. I will provide the apartment review schema, the chunking strategy, and the requirement that each review should become exactly one chunk with the review title, review text, and management response combined. I expect Claude to generate code that reads the cleaned JSON review files, performs any final preprocessing, and produces chunk objects containing text and metadata. I will verify the output by checking that every review generates exactly one chunk, no reviews are dropped, and metadata such as apartment name, review ID, review date, and ratings are preserved.

### Milestone 4 — Embedding and Retrieval

I will use Claude Code to implement the embedding and retrieval pipeline. I will provide the Retrieval Approach section, specifying the `all-MiniLM-L6-v2` embedding model, Chroma DB as the vector store, and a retrieval value of Top-k = 5. I expect Claude to generate code that creates embeddings for each review chunk, stores them in a persistent Chroma collection, and retrieves the most relevant chunks for a user query using semantic similarity search. I will verify the implementation by running the evaluation questions and confirming that the retrieved reviews contain information relevant to the expected answers.

### Milestone 5 — Generation and Interface

I will use Claude Code to implement the answer generation pipeline and user interface. I will provide the pipeline diagram, evaluation questions, and the requirement that generated answers must be based only on retrieved review chunks. I expect Claude to produce the retrieval-augmented generation logic, prompt template, and a simple command-line or web interface that accepts user questions and returns grounded answers. I will verify the system by comparing generated responses against the expected answers in the evaluation plan and ensuring that answers are supported by retrieved reviews rather than unsupported assumptions.
