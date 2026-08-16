from src.models import WordTimestamp
from src.reconciler import align_words, reconcile
from backend.transcription import _align_to_transcript, parse_transcript_text


def test_alignment_interpolates_missing_whisper_tokens():
    transcript = parse_transcript_text("hello missing world")
    stt = [
        WordTimestamp("hello", 0.5, 0.9, 0.9, "whisper"),
        WordTimestamp("world", 2.0, 2.4, 0.9, "whisper"),
    ]
    aligned = _align_to_transcript(transcript, stt)
    assert [item.word for item in aligned] == ["hello", "missing", "world"]
    assert aligned[1].source == "whisper_interpolated"
    assert aligned[1].interpolated is True
    assert aligned[1].confidence == 0.28
    assert aligned[0].start < aligned[1].start <= aligned[2].start


def test_alignment_matches_similar_tokens():
    transcript = parse_transcript_text("timestamps demo")
    stt = [
        WordTimestamp("timestamp", 1.0, 1.5, 0.9, "whisper"),
        WordTimestamp("demo", 1.6, 2.0, 0.9, "whisper"),
    ]
    aligned = _align_to_transcript(transcript, stt)
    assert aligned[0].interpolated is False
    assert aligned[0].start == 1.0


def test_clip_window_uses_final_last_word_not_metadata(sample_aligned):
    result = reconcile(sample_aligned, padding=0.0)
    reliably = next(word for word in result.final_words if word.word == "reliably")
    metadata_reliably = next(
        item.metadata.end for item in sample_aligned if item.word == "reliably"
    )
    assert result.clip_end == reliably.end
    assert result.clip_end != metadata_reliably
    assert reliably.source == "stt"


def test_padding_does_not_invert_tiny_clips():
    tokens = ["a", "b"]
    metadata = [
        WordTimestamp("a", 0.02, 0.08, source="metadata"),
        WordTimestamp("b", 0.10, 0.16, source="metadata"),
    ]
    stt = [
        WordTimestamp("a", 0.02, 0.08, 0.95, "stt"),
        WordTimestamp("b", 0.10, 0.16, 0.95, "stt"),
    ]
    result = reconcile(align_words(tokens, metadata, stt), padding=0.5)
    assert result.clip_end > result.clip_start
