import json
import html
import re
import unicodedata


def clean_text(text: str) -> str:
    """
    Clean review_title, review_text, and response_text for RAG.

    Steps:
    - decode HTML entities (&rsquo;, &quot;, &#44;, etc.)
    - normalize unicode
    - remove control characters
    - normalize whitespace/newlines
    - remove consecutive duplicate lines
    - remove consecutive duplicate sentences
    """
    if not text:
        return ""

    text = str(text)

    # Decode HTML entities
    text = html.unescape(text)

    # Unicode normalization
    text = unicodedata.normalize("NFKC", text)

    # Normalize line endings
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Remove control characters
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", " ", text)

    # Normalize spaces
    text = re.sub(r"[ \t]+", " ", text)

    # Collapse excessive blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Remove duplicate consecutive lines
    lines = [line.strip() for line in text.split("\n")]
    deduped_lines = []

    prev = None
    for line in lines:
        if line and line == prev:
            continue
        deduped_lines.append(line)
        prev = line

    text = "\n".join(deduped_lines)

    # Remove duplicate consecutive sentences
    paragraphs = text.split("\n\n")
    cleaned_paragraphs = []

    sentence_splitter = re.compile(r"(?<=[.!?])\s+")

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        sentences = sentence_splitter.split(para)

        deduped_sentences = []
        prev_sentence = None

        for sentence in sentences:
            sentence = sentence.strip()

            if not sentence:
                continue

            normalized = sentence.lower()

            if normalized == prev_sentence:
                continue

            deduped_sentences.append(sentence)
            prev_sentence = normalized

        cleaned_paragraphs.append(" ".join(deduped_sentences))

    text = "\n\n".join(cleaned_paragraphs)

    # Final cleanup
    text = text.replace("\n", " ")

    # Collapse multiple spaces again
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n[ \t]+", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[-_=*~]{3,}", " ", text)
    text = re.sub(r"(.)\1{2,}", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text.strip()


def clean_reviews_json(input_file: str, output_file: str) -> None:
    """
    Reads your flattened reviews JSON file and cleans:
      - review_title
      - review_text
      - response_text

    Writes cleaned JSON to output_file.
    """

    with open(input_file, "r", encoding="utf-8") as f:
        reviews = json.load(f)

    for review in reviews:
        review["review_title"] = clean_text(
            review.get("review_title", "")
        )

        review["review_text"] = clean_text(
            review.get("review_text", "")
        )

        review["response_text"] = clean_text(
            review.get("response_text", "")
        )

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(
            reviews,
            f,
            indent=2,
            ensure_ascii=False
        )

    print(
        f"Cleaned {len(reviews):,} reviews -> {output_file}"
    )

if __name__ == "__main__":
    clean_reviews_json(
        "../data/all_reviews_flattened.json",
        "../data/final_data.json"
    )