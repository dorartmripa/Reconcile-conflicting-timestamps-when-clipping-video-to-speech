from src.models import CONFLICT_THRESHOLD_SECONDS, WordTimestamp
from src.reconciler import align_words, detect_conflicts
from tests.helpers import paired


def test_no_conflict_when_difference_is_small():
    tokens = ["hello", "world"]
    metadata, stt = paired(tokens, [1.0, 1.5], [1.2, 1.7], [0.9, 0.9])
    aligned = align_words(tokens, metadata, stt)
    assert detect_conflicts(aligned) == []
    assert all(item.difference <= CONFLICT_THRESHOLD_SECONDS for item in aligned)


def test_conflict_when_difference_exceeds_half_second():
    tokens = ["hello"]
    metadata, stt = paired(tokens, [12.20], [12.91], [0.94])
    aligned = align_words(tokens, metadata, stt)
    conflicts = detect_conflicts(aligned)
    assert len(conflicts) == 1
    assert conflicts[0].word == "hello"
    assert abs(conflicts[0].difference - 0.71) < 1e-9


def test_end_mismatch_is_a_conflict():
    tokens = ["hello"]
    metadata = [WordTimestamp("hello", 1.0, 1.2, source="metadata")]
    stt = [WordTimestamp("hello", 1.1, 2.0, 0.9, "stt")]
    aligned = align_words(tokens, metadata, stt)
    assert aligned[0].difference <= CONFLICT_THRESHOLD_SECONDS
    assert aligned[0].end_difference > CONFLICT_THRESHOLD_SECONDS
    assert aligned[0].is_conflict


def test_sample_data_contains_at_least_three_conflicts(sample_aligned):
    conflicts = detect_conflicts(sample_aligned)
    large = [item for item in conflicts if item.difference > 0.5]
    assert len(large) >= 3
    words = {item.word.lower() for item in large}
    assert {"timestamp", "system", "conflicts"}.issubset(words)
