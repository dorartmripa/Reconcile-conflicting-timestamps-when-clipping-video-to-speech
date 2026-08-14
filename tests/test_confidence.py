from src.reconciler import align_words, reconcile
from tests.helpers import paired


def test_missing_confidence_is_treated_as_zero_and_can_favor_metadata():
    tokens = ["alpha", "beta", "gamma"]
    metadata, stt = paired(
        tokens,
        [1.0, 2.2, 3.0],
        [1.0, 1.4, 3.0],
        [0.9, 0.9, 0.9],
    )
    stt[1].confidence = None
    result = reconcile(align_words(tokens, metadata, stt))
    conflict = result.conflicts[0]
    assert conflict.word == "beta"
    assert conflict.stt_confidence is None
    assert conflict.decision == "metadata"
    assert conflict.factors["stt_confidence"] == 0.0


def test_confidence_changes_the_winner():
    tokens = ["left", "pivot", "right"]
    meta_starts = [1.0, 2.4, 3.0]
    stt_starts = [1.0, 1.7, 3.0]

    high_meta, high_stt = paired(tokens, meta_starts, stt_starts, [0.9, 0.97, 0.9])
    low_meta, low_stt = paired(tokens, meta_starts, stt_starts, [0.9, 0.20, 0.9])

    high = reconcile(align_words(tokens, high_meta, high_stt)).conflicts[0]
    low = reconcile(align_words(tokens, low_meta, low_stt)).conflicts[0]

    assert high.decision == "STT"
    assert low.decision == "metadata"
