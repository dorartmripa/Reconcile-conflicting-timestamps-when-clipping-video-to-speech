from src.reconciler import format_conflict_logs


def test_conflict_log_matches_required_shape(sample_result):
    text = format_conflict_logs(sample_result.conflicts)
    assert "Conflict #1" in text
    assert "Word:" in text
    assert "Metadata:" in text
    assert "STT:" in text
    assert "Difference:" in text
    assert "STT confidence:" in text
    assert "Factors considered:" in text
    assert "Decision:" in text
    assert "Reason:" in text
    assert "Final timestamp:" in text
    assert text.count("Conflict #") >= 3
