from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TypedDict


@dataclass(slots=True)
class Citation:
    index: int
    source: str
    page: int
    snippet: str


class CitationDict(TypedDict):
    index: int
    source: str
    page: int
    snippet: str


def semantic_split_text(text: str, max_chars: int = 1_200) -> list[str]:
    cleaned = re.sub(r"\s+", " ", text).strip()
    if not cleaned:
        return []

    sentences = re.split(r"(?<=[.!?])\s+", cleaned)
    chunks: list[str] = []
    current = ""

    for sentence in sentences:
        if not sentence:
            continue

        candidate = sentence if not current else f"{current} {sentence}"
        if len(candidate) <= max_chars:
            current = candidate
            continue

        if current:
            chunks.append(current)

        if len(sentence) <= max_chars:
            current = sentence
            continue

        start = 0
        while start < len(sentence):
            end = start + max_chars
            chunks.append(sentence[start:end].strip())
            start = end
        current = ""

    if current:
        chunks.append(current)

    return chunks


def build_answer_markdown(
    answer_text: str,
    confidence: float,
    citations: list[CitationDict],
) -> str:
    clamped = max(0.0, min(1.0, confidence))
    confidence_text = f"Confidence: {clamped * 100:.1f}%"

    lines = [answer_text.strip(), "", "References:"]
    for citation in citations:
        index = citation["index"]
        source = citation["source"]
        page = citation["page"]
        snippet = citation["snippet"]
        lines.append(f"[{index}] {source} (p.{page}) — {snippet}")

    lines.extend(["", confidence_text])
    return "\n".join(lines)
