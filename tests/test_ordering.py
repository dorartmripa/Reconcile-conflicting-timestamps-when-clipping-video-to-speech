from src.reconciler import align_words, is_monotonic, reconcile
from tests.helpers import paired


def test_is_monotonic_rejects_backwards_candidate():
    times = [1.0, 1.4, 2.0]
    assert is_monotonic(times, 1, 1.4) is True
    assert is_monotonic(times, 1, 0.5) is False
    assert is_monotonic(times, 1, 2.5) is False


def test_backwards_stt_is_rejected_in_favor_of_metadata():
    tokens = ["first", "second", "third"]
    metadata, stt = paired(
        tokens,
        [1.00, 1.80, 2.40],
        [1.00, 0.40, 2.40],
        [0.99, 0.99, 0.99],
    )
    result = reconcile(align_words(tokens, metadata, stt))
    conflict = result.conflicts[0]
    assert conflict.word == "second"
    assert conflict.decision == "metadata"
    reason = conflict.reason.lower()
    assert "chronological" in reason or "order" in reason or "backwards" in reason
    starts = [word.start for word in result.final_words]
    assert starts == sorted(starts)


def test_final_timestamps_remain_non_decreasing(sample_result):
    starts = [word.start for word in sample_result.final_words]
    assert starts == sorted(starts)
