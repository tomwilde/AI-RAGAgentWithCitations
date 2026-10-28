from streamlit.testing.v1 import AppTest


def test_app_renders_core_actions() -> None:
    at = AppTest.from_file("../app.py").run()

    labels = [btn.label for btn in at.button]
    assert "Generate Embeddings" not in labels
    assert "Query" in labels
    assert "Flush Embeddings" not in labels


def test_question_is_disabled_before_embeddings_ready() -> None:
    at = AppTest.from_file("../app.py").run()
    assert at.text_area(key="Question").disabled


def test_question_is_enabled_after_embeddings_ready() -> None:
    at = AppTest.from_file("../app.py")
    at.session_state["embeddings_ready"] = True
    at.run()
    assert not at.text_area(key="Question").disabled


def test_embedding_progress_feedback_is_rendered() -> None:
    at = AppTest.from_file("../app.py")
    at.session_state["embed_progress"] = 0.5
    at.session_state["embed_message"] = "Embedding chunks 5/10 (50%)"
    at.run()

    assert len(at.progress) == 1
    assert any("Embedding chunks 5/10 (50%)" in info.value for info in at.info)
