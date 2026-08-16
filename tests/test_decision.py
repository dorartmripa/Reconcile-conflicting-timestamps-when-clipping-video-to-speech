from src.reconciler import align_words, reconcile
from tests.helpers import paired


def test_high_confidence_stt_with_consistent_neighbors_prefers_stt():
    tokens = ["one", "timestamp", "three"]
    metadata, stt = paired(
        tokens,
        [1.0, 2.30, 2.8],
        [1.0, 1.55, 2.8],
        [0.95, 0.96, 0.95],
    )
    result = reconcile(align_words(tokens, metadata, stt))
    conflict = next(item for item in result.conflicts if item.word == "timestamp")
    assert conflict.decision == "STT"
    assert conflict.final_timestamp == 1.55
    assert "confidence" in conflict.reason.lower() or "score" in conflict.reason.lower()


def test_low_confidence_stt_with_consistent_metadata_prefers_metadata():
    tokens = ["this", "system", "detects"]
    metadata, stt = paired(
        tokens,
        [3.42, 4.55, 5.20],
        [3.40, 3.85, 5.18],
        [0.91, 0.30, 0.92],
    )
    result = reconcile(align_words(tokens, metadata, stt))
    conflict = next(item for item in result.conflicts if item.word == "system")
    assert conflict.decision == "metadata"
    assert conflict.final_timestamp == 4.55
    assert conflict.stt_confidence == 0.30


def test_decision_is_not_hardcoded_to_one_source(sample_result):
    decisions = {conflict.decision for conflict in sample_result.conflicts}
    assert "STT" in decisions
    assert "metadata" in decisions


def test_every_conflict_has_explainable_fields(sample_result):
    assert len(sample_result.conflicts) >= 3
    for conflict in sample_result.conflicts:
        assert conflict.word
        assert conflict.difference > 0.5
        assert conflict.decision in {"STT", "metadata", "blend", "untrusted"}
        assert conflict.reason
        assert "stt_score" in conflict.factors
        assert "metadata_score" in conflict.factors
        assert "stt_neighbor_consistency" in conflict.factors
        assert "drift_detected" in conflict.factors
