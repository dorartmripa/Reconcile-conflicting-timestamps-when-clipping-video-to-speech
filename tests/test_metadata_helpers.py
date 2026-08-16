from backend.metadata import naive_ingest_markers, stale_metadata_from_stt
from backend.models import WordTimestamp, CONFLICT_THRESHOLD_SECONDS
from backend.transcription import parse_transcript_text


def _stt_words():
    tokens = parse_transcript_text("one two three four five six seven eight")
    return [
        WordTimestamp(token, i * 0.8, i * 0.8 + 0.4, 0.9, "stt")
        for i, token in enumerate(tokens)
    ], tokens


def test_stale_metadata_from_stt_has_at_least_three_large_conflicts():
    stt, tokens = _stt_words()
    meta, _note = stale_metadata_from_stt(stt, tokens)
    large = [
        abs(left.start - right.start)
        for left, right in zip(stt, meta)
        if abs(left.start - right.start) > CONFLICT_THRESHOLD_SECONDS
    ]
    assert len(large) >= 3


def test_naive_ingest_is_independent_of_stt_times():
    stt, tokens = _stt_words()
    meta, note = naive_ingest_markers(tokens, duration=20.0)
    assert "independent" in note.lower()
    assert [word.start for word in meta] != [word.start for word in stt]
    assert meta[0].source == "naive_ingest_markers"
