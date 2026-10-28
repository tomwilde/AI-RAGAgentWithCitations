from src.rag_logic import build_answer_markdown, semantic_split_text


def test_semantic_split_text_respects_max_chars() -> None:
    text = "Sentence one. Sentence two is a bit longer. Sentence three wraps up."
    chunks = semantic_split_text(text, max_chars=30)
    assert chunks
    assert all(len(chunk) <= 30 for chunk in chunks)


def test_build_answer_markdown_includes_confidence_and_citations() -> None:
    markdown = build_answer_markdown(
        answer_text="A concise answer.",
        confidence=0.8123,
        citations=[
            {"index": 1, "source": "doc.pdf", "page": 2, "snippet": "Relevant snippet A"},
            {"index": 2, "source": "doc.pdf", "page": 5, "snippet": "Relevant snippet B"},
        ],
    )

    assert "A concise answer." in markdown
    assert "Confidence: 81.2%" in markdown
    assert "[1] doc.pdf (p.2)" in markdown
    assert "[2] doc.pdf (p.5)" in markdown
