from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor, as_completed
import io
import os
import threading
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from openai import OpenAI
from pypdf import PdfReader
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointIdsList, PointStruct, VectorParams

from src.agentic_logic import GenerationStage, gate_retrieval, resolve_final_status
from src.agentic_logic import parse_json_object, should_include_low_confidence_answer
from src.rag_logic import CitationDict, build_answer_markdown, semantic_split_text

COLLECTION_NAME = "rag_chunks"
EMBEDDING_MODEL = "text-embedding-3-small"
CHAT_MODEL = "openai/gpt-4.1-mini"


class IngestionCancelledError(RuntimeError):
    pass


@dataclass(slots=True)
class ChunkRecord:
    point_id: str
    text: str
    source: str
    page: int


class RAGService:
    def __init__(self, qdrant_url: str, openrouter_api_key: str) -> None:
        self.qdrant = QdrantClient(url=qdrant_url)
        self.openai = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=openrouter_api_key)
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        collections = self.qdrant.get_collections().collections
        names = {collection.name for collection in collections}
        if COLLECTION_NAME not in names:
            self.qdrant.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
            )

    def flush(self) -> None:
        self.qdrant.delete_collection(collection_name=COLLECTION_NAME)
        self._ensure_collection()

    def _embed_chunk(self, chunk: ChunkRecord) -> PointStruct:
        embedding_response = self.openai.embeddings.create(
            model=EMBEDDING_MODEL,
            input=chunk.text,
        )
        embedding = embedding_response.data[0].embedding
        return PointStruct(
            id=chunk.point_id,
            vector=embedding,
            payload={
                "text": chunk.text,
                "source": chunk.source,
                "page": chunk.page,
            },
        )

    def _count_chunks(self, reader: PdfReader) -> int:
        total = 0
        for page in reader.pages:
            page_text = page.extract_text() or ""
            total += len(semantic_split_text(page_text))
        return total

    def _ingest_reader(
        self,
        reader: PdfReader,
        source_name: str,
        progress_cb: Callable[[int, int], None],
        cancel_event: threading.Event | None = None,
    ) -> int:
        total = self._count_chunks(reader)
        if total == 0:
            return 0

        inserted_ids: list[str] = []
        completed = 0
        worker_count = min(
            total,
            max(1, int(os.getenv("EMBEDDING_WORKERS", "6"))),
        )

        try:
            for page_idx, page in enumerate(reader.pages, start=1):
                page_text = page.extract_text() or ""
                page_chunks = semantic_split_text(page_text)
                if not page_chunks:
                    continue

                chunk_records = [
                    ChunkRecord(
                        point_id=str(uuid.uuid4()),
                        text=chunk_text,
                        source=source_name,
                        page=page_idx,
                    )
                    for chunk_text in page_chunks
                ]

                page_workers = min(worker_count, len(chunk_records))
                points_batch: list[PointStruct] = []
                with ThreadPoolExecutor(max_workers=page_workers) as executor:
                    futures: dict[Future[PointStruct], str] = {
                        executor.submit(self._embed_chunk, chunk): chunk.point_id
                        for chunk in chunk_records
                    }

                    for future in as_completed(futures):
                        if cancel_event is not None and cancel_event.is_set():
                            for pending in futures:
                                pending.cancel()
                            raise IngestionCancelledError("Embedding generation cancelled.")

                        point = future.result()
                        points_batch.append(point)
                        inserted_ids.append(str(point.id))
                        completed += 1
                        progress_cb(completed, total)

                        if len(points_batch) >= 32:
                            self.qdrant.upsert(
                                collection_name=COLLECTION_NAME,
                                points=points_batch,
                            )
                            points_batch = []

                if points_batch:
                    self.qdrant.upsert(
                        collection_name=COLLECTION_NAME,
                        points=points_batch,
                    )
        except Exception:
            if inserted_ids:
                self.qdrant.delete(
                    collection_name=COLLECTION_NAME,
                    points_selector=PointIdsList(points=inserted_ids),
                )
            raise

        return total

    def ingest_pdf_path(
        self,
        pdf_path: str | Path,
        source_name: str,
        progress_cb: Callable[[int, int], None],
        cancel_event: threading.Event | None = None,
    ) -> int:
        reader = PdfReader(str(pdf_path))
        return self._ingest_reader(
            reader=reader,
            source_name=source_name,
            progress_cb=progress_cb,
            cancel_event=cancel_event,
        )

    def ingest_pdf(
        self,
        pdf_bytes: bytes,
        source_name: str,
        progress_cb: Callable[[int, int], None],
        cancel_event: threading.Event | None = None,
    ) -> int:
        reader = PdfReader(io.BytesIO(pdf_bytes))
        return self._ingest_reader(
            reader=reader,
            source_name=source_name,
            progress_cb=progress_cb,
            cancel_event=cancel_event,
        )

    def answer(self, question: str, limit: int = 5) -> str:
        search, _top_score = self._retrieve(question=question, limit=limit)

        if not search:
            return "No indexed content found. Upload a PDF and generate embeddings first."

        context, citations = self._build_context_and_citations(search)
        answer_text = self._generate_grounded_answer(question=question, context=context)
        confidence = self._confidence_from_hits(search)

        return build_answer_markdown(
            answer_text=answer_text,
            confidence=confidence,
            citations=citations,
        )

    def answer_agentic(self, question: str, limit: int = 5) -> str:
        search, top_score = self._retrieve(question=question, limit=limit)

        if not search:
            return (
                "I couldn't find relevant information in the uploaded documents to answer this "
                "confidently. Please upload more relevant material and try again.\n\n"
                "Decision: INSUFFICIENT_LOCAL_CONTEXT\nTop-1 score: 0.0000"
            )

        stage = gate_retrieval(top_score)
        if stage == GenerationStage.FALLBACK_WEB:
            return (
                "The uploaded documents don't provide strong enough grounding for this question. "
                "Please upload additional relevant documents and try again.\n\n"
                f"{self._agentic_diagnostics(search=search, top_score=top_score, stage=stage)}"
            )

        context, citations = self._build_context_and_citations(search)
        structured = self._generate_structured_answer(question=question, context=context)
        supported = self._to_supported_flag(structured.get("supported", False))
        final_status = resolve_final_status(top_score=top_score, supported=supported)

        if final_status == "LOW_CONFIDENCE":
            if should_include_low_confidence_answer(top_score=top_score, supported=supported):
                answer_text = str(structured.get("answer", "Insufficient information."))
                confidence = self._confidence_from_hits(search)
                markdown = build_answer_markdown(
                    answer_text=answer_text,
                    confidence=confidence,
                    citations=citations,
                )
                return (
                    f"{markdown}\n\nDecision: LOW_CONFIDENCE\n"
                    f"Top-1 score: {top_score:.4f}\n"
                    "Note: Answer is tentative due to moderate retrieval similarity.\n\n"
                    f"{self._agentic_diagnostics(search=search, top_score=top_score, stage=stage, supported=supported, final_status=final_status)}"
                )

            return (
                "I found related passages, but there isn't enough grounded support to answer "
                "reliably from local documents. Please refine your question or upload more "
                "specific sources.\n\n"
                f"Model supported flag: {supported}\n"
                f"Top-1 score: {top_score:.4f}\n"
                f"Decision: LOW_CONFIDENCE\n\n"
                f"{self._agentic_diagnostics(search=search, top_score=top_score, stage=stage, supported=supported, final_status=final_status)}"
            )

        answer_text = str(structured.get("answer", "No answer generated."))
        confidence = self._confidence_from_hits(search)
        markdown = build_answer_markdown(
            answer_text=answer_text,
            confidence=confidence,
            citations=citations,
        )
        diagnostics = self._agentic_diagnostics(
            search=search,
            top_score=top_score,
            stage=stage,
            supported=supported,
            final_status=final_status,
            citations=citations,
        )
        return f"{markdown}\n\nDecision: GROUNDED\nTop-1 score: {top_score:.4f}\n\n{diagnostics}"

    def _retrieve(self, question: str, limit: int) -> tuple[list[object], float]:
        embedding_response = self.openai.embeddings.create(
            model=EMBEDDING_MODEL,
            input=question,
        )
        embedding = embedding_response.data[0].embedding
        query_response = self.qdrant.query_points(
            collection_name=COLLECTION_NAME,
            query=embedding,
            limit=limit,
            with_payload=True,
        )
        search = query_response.points

        if not search:
            return [], 0.0

        top_score = float(search[0].score or 0.0)
        return search, top_score

    def _build_context_and_citations(self, search: list[object]) -> tuple[str, list[CitationDict]]:
        context_lines = []
        citations: list[CitationDict] = []
        for i, hit in enumerate(search, start=1):
            payload = hit.payload or {}
            source = str(payload.get("source", "unknown"))
            page = int(payload.get("page", 0))
            text = str(payload.get("text", ""))
            context_lines.append(f"[{i}] {source} p.{page}: {text}")
            citations.append(
                {
                    "index": i,
                    "source": source,
                    "page": page,
                    "snippet": text[:180],
                }
            )

        return "\n".join(context_lines), citations

    def _generate_grounded_answer(self, question: str, context: str) -> str:
        prompt = (
            "Use only the provided context to answer the question. "
            "Cite sources inline using bracketed numbers like [1].\n\n"
            f"Question: {question}\n\nContext:\n{context}"
        )

        response = self.openai.chat.completions.create(
            model=os.getenv("OPENROUTER_MODEL", CHAT_MODEL),
            messages=[
                {"role": "system", "content": "You are a precise assistant that cites sources."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
        )
        return response.choices[0].message.content or "No answer generated."

    def _generate_structured_answer(self, question: str, context: str) -> dict[str, object]:
        prompt = (
            "Answer strictly from the numbered context passages. "
            "Return JSON only with fields: answer (string), supported (boolean). "
            "Use inline citations like [1], [2] inside answer when relevant.\n\n"
            f"Question: {question}\n\nContext:\n{context}"
        )

        response = self.openai.chat.completions.create(
            model=os.getenv("OPENROUTER_MODEL", CHAT_MODEL),
            messages=[
                {
                    "role": "system",
                    "content": "You are a strict JSON generator. Output only JSON.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0,
        )
        content = response.choices[0].message.content or "{}"

        parsed = parse_json_object(content)
        if parsed is None:
            return {"answer": "Insufficient information.", "supported": False}

        if not isinstance(parsed, dict):
            return {"answer": "Insufficient information.", "supported": False}

        return parsed

    def _confidence_from_hits(self, search: list[object]) -> float:
        scores = [hit.score or 0.0 for hit in search]
        return min(1.0, max(0.0, (sum(scores[:3]) / max(1, len(scores[:3])))))

    def _to_supported_flag(self, value: object) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.strip().lower() in {"true", "yes", "1"}
        return False

    def _agentic_diagnostics(
        self,
        search: list[object],
        top_score: float,
        stage: GenerationStage,
        supported: bool | None = None,
        final_status: str | None = None,
        citations: list[CitationDict] | None = None,
    ) -> str:
        top_scores = [f"{(hit.score or 0.0):.4f}" for hit in search[:5]]
        lines = [
            "Agentic diagnostics:",
            f"- Gate stage: {stage.value}",
            f"- Top-1 score: {top_score:.4f}",
            f"- Top scores (up to 5): {', '.join(top_scores)}",
            f"- Retrieved passages: {len(search)}",
        ]

        if supported is not None:
            lines.append(f"- Model supported flag: {supported}")
        if final_status is not None:
            lines.append(f"- Final status: {final_status}")

        evidence = citations or self._build_context_and_citations(search)[1]
        if evidence:
            preview = []
            for citation in evidence[:3]:
                preview.append(
                    f"[{citation['index']}] {citation['source']} p.{citation['page']} — {citation['snippet'][:90]}"
                )
            lines.append("- Evidence preview:")
            lines.extend(f"  {item}" for item in preview)

        return "\n".join(lines)

def create_service_from_env() -> RAGService:
    qdrant_url = os.getenv("QDRANT_URL", "http://qdrant:6333")
    api_key = os.getenv("OPENROUTER_API_KEY", "")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is required")
    return RAGService(qdrant_url=qdrant_url, openrouter_api_key=api_key)
