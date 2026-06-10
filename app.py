#!/usr/bin/env python3
"""
Milestone 5 — Gradio user interface.

Run:
    python app.py
Then open http://localhost:7860 in your browser.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "scripts"))

import gradio as gr
from query import ask


def handle_query(question: str):
    if not question.strip():
        return "", ""
    result = ask(question)
    sources = "\n".join(f"• {s}" for s in result["sources"])
    return result["answer"], sources


with gr.Blocks(title="Apartment Review Assistant") as demo:
    gr.Markdown("## Apartment Review Assistant")
    gr.Markdown(
        "Ask questions about resident experiences at apartments in the dataset. "
        "Answers are grounded exclusively in real resident reviews."
    )

    inp = gr.Textbox(label="Your question", placeholder="e.g. What do residents say about maintenance?")
    btn = gr.Button("Ask")
    answer = gr.Textbox(label="Answer", lines=8)
    sources = gr.Textbox(label="Retrieved from", lines=4)

    btn.click(handle_query, inputs=inp, outputs=[answer, sources])
    inp.submit(handle_query, inputs=inp, outputs=[answer, sources])


if __name__ == "__main__":
    demo.launch()
