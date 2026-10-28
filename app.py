from __future__ import annotations

import hashlib
import os
import tempfile

import streamlit as st

from src.rag_service import create_service_from_env

st.set_page_config(page_title="RAG Agent with Citations")
st.title("RAG Agent with Citations")


def init_state() -> None:
    st.session_state.setdefault("service", None)
    st.session_state.setdefault("embed_progress", 0.0)
    st.session_state.setdefault("embed_message", "")
    st.session_state.setdefault("embed_error", "")
    st.session_state.setdefault("embeddings_ready", False)
    st.session_state.setdefault("last_uploaded_digest", "")
    st.session_state.setdefault("history", [])


def get_service():
    if st.session_state.service is None:
        st.session_state.service = create_service_from_env()
    return st.session_state.service


def generate_embeddings(file_name: str, pdf_path: str) -> None:
    st.session_state.embed_progress = 0.0
    st.session_state.embed_message = "Preparing embeddings... (0%)"
    st.session_state.embed_error = ""
    st.session_state.embeddings_ready = False

    progress_bar = st.progress(st.session_state.embed_progress)
    status_box = st.empty()
    status_box.info(st.session_state.embed_message)

    try:
        service = get_service()
        service.flush()
        st.session_state.history = []

        def progress_cb(done: int, total: int) -> None:
            progress_ratio = done / total
            progress_percent = int(progress_ratio * 100)
            st.session_state.embed_progress = progress_ratio
            st.session_state.embed_message = (
                f"Embedding chunks {done}/{total} ({progress_percent}%)"
            )
            progress_bar.progress(progress_ratio)
            status_box.info(st.session_state.embed_message)

        total = service.ingest_pdf_path(
            pdf_path=pdf_path,
            source_name=file_name,
            progress_cb=progress_cb,
        )

        st.session_state.embed_progress = 1.0
        st.session_state.embed_message = f"Indexed {total} chunks. (100%)"
        st.session_state.embeddings_ready = total > 0
        progress_bar.progress(1.0)
        status_box.success(st.session_state.embed_message)
    except Exception as error:
        st.session_state.embed_error = str(error)
        status_box.error(st.session_state.embed_error)


init_state()

uploaded = st.file_uploader("Upload PDF", type=["pdf"], label_visibility="collapsed")

if uploaded is not None:
    uploaded.seek(0)
    hasher = hashlib.sha256()
    temp_input_path = ""
    try:
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as temp_upload:
            temp_input_path = temp_upload.name
            while True:
                chunk = uploaded.read(1024 * 1024)
                if not chunk:
                    break
                hasher.update(chunk)
                temp_upload.write(chunk)

        uploaded_digest = hasher.hexdigest()

        if uploaded_digest != st.session_state.last_uploaded_digest:
            st.session_state.last_uploaded_digest = uploaded_digest
            generate_embeddings(uploaded.name, temp_input_path)
    finally:
        uploaded.seek(0)
        if temp_input_path and os.path.exists(temp_input_path):
            os.unlink(temp_input_path)

if st.session_state.embed_error:
    st.error(st.session_state.embed_error)

st.subheader("Questions and Answers")

mode = st.toggle(
    "Use Agentic RAG",
    value=True,
    help=(
        "Simple: direct local retrieval + single grounded answer with citations.\n\n"
        "Agentic: applies retrieval score gates and model self-verification; "
        "if local evidence is insufficient, it fails gracefully with a clear message."
    ),
)
mode_label = "Agentic" if mode else "Simple"

query_disabled = not st.session_state.embeddings_ready

question = st.text_area("Question", key="Question", height=180, disabled=query_disabled)
if st.button("Query", disabled=query_disabled):
    if not question.strip():
        st.warning("Enter a question.")
    else:
        try:
            if mode:
                answer = get_service().answer_agentic(question.strip())
            else:
                answer = get_service().answer(question.strip())

            answer = f"Using {mode_label} RAG mode\n\n{answer}"

            st.session_state.history.append(
                {
                    "question": question.strip(),
                    "answer": answer,
                    "mode": mode_label,
                }
            )
        except Exception as error:
            st.error(str(error))

for idx, item in enumerate(st.session_state.history, start=1):
    st.markdown(f"### Q{idx}")
    st.text_area(f"Question {idx}", value=item["question"], height=100, disabled=True)
    st.text_area(f"Answer {idx}", value=item["answer"], height=220, disabled=True)
