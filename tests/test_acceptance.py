from streamlit.testing.v1 import AppTest


def test_acceptance_core_rag_ui_flow() -> None:
    at = AppTest.from_file("../app.py")
    at.session_state["history"] = [
        {"question": "What is in this file?", "answer": "Sample answer [1]\n\nConfidence: 90.0%"}
    ]
    at.run()

    assert len(at.file_uploader) == 1
    assert len(at.text_area) >= 2

    labels = [button.label for button in at.button]
    assert "Generate Embeddings" not in labels
    assert "Flush Embeddings" not in labels
    assert "Query" in labels

    values = [area.value or "" for area in at.text_area]
    assert any("What is in this file?" in value for value in values)
    assert any("Confidence: 90.0%" in value for value in values)
