from src.agentic_logic import (
    GenerationStage,
    gate_retrieval,
    parse_json_object,
    resolve_final_status,
    should_include_low_confidence_answer,
)


def test_gate_retrieval_routes_to_fallback_when_score_below_threshold() -> None:
    stage = gate_retrieval(top_score=0.10)
    assert stage == GenerationStage.FALLBACK_WEB


def test_gate_retrieval_marks_low_confidence_band() -> None:
    stage = gate_retrieval(top_score=0.22)
    assert stage == GenerationStage.GENERATE_LOW_CONFIDENCE


def test_gate_retrieval_marks_grounded_band() -> None:
    stage = gate_retrieval(top_score=0.35)
    assert stage == GenerationStage.GENERATE_NORMAL


def test_model_honesty_flag_overrides_mid_similarity_band() -> None:
    result = resolve_final_status(top_score=0.2185, supported=False)
    assert result == "LOW_CONFIDENCE"


def test_grounded_requires_supported_and_high_score() -> None:
    result = resolve_final_status(top_score=0.40, supported=True)
    assert result == "GROUNDED"


def test_mid_score_supported_can_return_tentative_answer() -> None:
    assert should_include_low_confidence_answer(top_score=0.22, supported=True)


def test_unsupported_never_returns_tentative_answer() -> None:
    assert not should_include_low_confidence_answer(top_score=0.25, supported=False)


def test_parse_json_object_accepts_plain_json() -> None:
    parsed = parse_json_object('{"answer":"A [1]","supported":true}')
    assert parsed == {"answer": "A [1]", "supported": True}


def test_parse_json_object_accepts_fenced_json() -> None:
    content = '```json\n{"answer":"A [1]", "supported": false}\n```'
    parsed = parse_json_object(content)
    assert parsed == {"answer": "A [1]", "supported": False}


def test_parse_json_object_returns_none_for_non_json() -> None:
    assert parse_json_object("not json") is None
