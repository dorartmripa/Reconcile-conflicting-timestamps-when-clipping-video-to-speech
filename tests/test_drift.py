from src.reconciler import align_words, estimate_offset, reconcile
from tests.helpers import paired


def test_estimate_offset_uses_high_confidence_median():
    tokens = ["a", "b", "c", "d"]
    metadata, stt = paired(
        tokens,
        [1.0, 2.0, 3.0, 4.0],
        [1.6, 2.6, 3.6, 4.6],
        [0.95, 0.95, 0.95, 0.95],
    )
    aligned = align_words(tokens, metadata, stt)
    offset, mad = estimate_offset(aligned)
    assert abs(offset - 0.6) < 1e-9
    assert mad == 0.0


def test_systematic_drift_is_detected_and_logged():
    tokens = ["w1", "w2", "w3", "w4", "w5"]
    metadata, stt = paired(
        tokens,
        [0.5, 1.5, 2.5, 3.5, 4.5],
        [1.1, 2.1, 3.1, 4.1, 5.1],
        [0.95, 0.94, 0.96, 0.93, 0.97],
    )
    result = reconcile(align_words(tokens, metadata, stt))
    assert result.drift_detected is True
    assert abs(result.estimated_offset - 0.6) < 1e-9
    assert all(conflict.factors["drift_detected"] for conflict in result.conflicts)
    # High-confidence STT should win when the gap is a stable offset.
    assert all(conflict.decision == "STT" for conflict in result.conflicts)


def test_linear_stretch_sets_nonzero_slope():
    tokens = ["w1", "w2", "w3", "w4", "w5"]
    metadata, stt = paired(
        tokens,
        [1.0, 2.0, 3.0, 4.0, 5.0],
        [1.1, 2.2, 3.3, 4.4, 5.5],
        [0.95, 0.95, 0.95, 0.95, 0.95],
    )
    result = reconcile(align_words(tokens, metadata, stt))
    assert result.drift_detected is True
    assert result.estimated_slope > 0.05


def test_inconsistent_gaps_are_not_called_drift():
    tokens = ["w1", "w2", "w3", "w4"]
    metadata, stt = paired(
        tokens,
        [0.5, 1.5, 2.5, 3.5],
        [0.5, 2.4, 2.6, 5.2],
        [0.95, 0.95, 0.95, 0.95],
    )
    result = reconcile(align_words(tokens, metadata, stt))
    assert result.drift_detected is False
