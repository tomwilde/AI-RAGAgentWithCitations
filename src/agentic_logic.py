from __future__ import annotations

import json
from enum import Enum


class GenerationStage(str, Enum):
    FALLBACK_WEB = "fallback_web"
    GENERATE_LOW_CONFIDENCE = "generate_low_confidence"
    GENERATE_NORMAL = "generate_normal"


def gate_retrieval(top_score: float) -> GenerationStage:
    if top_score < 0.15:
        return GenerationStage.FALLBACK_WEB
    if top_score < 0.30:
        return GenerationStage.GENERATE_LOW_CONFIDENCE
    return GenerationStage.GENERATE_NORMAL


def resolve_final_status(top_score: float, supported: bool) -> str:
    if not supported:
        return "LOW_CONFIDENCE"
    if top_score >= 0.30:
        return "GROUNDED"
    return "LOW_CONFIDENCE"


def should_include_low_confidence_answer(top_score: float, supported: bool) -> bool:
    return supported and 0.15 <= top_score < 0.30


def parse_json_object(content: str) -> dict[str, object] | None:
    trimmed = content.strip()

    try:
        parsed = json.loads(trimmed)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    start = trimmed.find("{")
    end = trimmed.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None

    candidate = trimmed[start : end + 1]
    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError:
        return None

    if isinstance(parsed, dict):
        return parsed
    return None
