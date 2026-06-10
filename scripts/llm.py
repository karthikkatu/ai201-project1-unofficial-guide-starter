#!/usr/bin/env python3
"""
Milestone 5 — Groq LLM wrapper.

Loads GROQ_API_KEY from .env and exposes generate(), which sends a
system prompt + user message to llama-3.3-70b-versatile and returns the
response text.
"""

import os
from pathlib import Path

from dotenv import load_dotenv
from groq import Groq
from typing import Optional

load_dotenv(Path(__file__).parent.parent / ".env")

GROQ_MODEL = "llama-3.3-70b-versatile"

SYSTEM_PROMPT = """\
You are an apartment review assistant.

Answer using ONLY the provided review context.

If the answer cannot be found in the provided reviews, respond exactly:

'I don't have enough information on that.'

Do not use outside knowledge.
Do not speculate.
Do not infer facts that are not explicitly supported by the retrieved reviews.\
"""

_client: Optional[Groq] = None


def _get_client() -> Groq:
    global _client
    if _client is None:
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "GROQ_API_KEY is not set. Copy .env.example to .env and add your key."
            )
        _client = Groq(api_key=api_key)
    return _client


def generate(context: str, question: str) -> str:
    """
    Send retrieved context + question to Groq and return the answer text.

    Temperature is 0 for deterministic, instruction-following output.
    """
    user_message = f"Review context:\n\n{context}\n\nQuestion: {question}"

    response = _get_client().chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        temperature=0.0,
        max_tokens=1024,
    )
    return response.choices[0].message.content.strip()
