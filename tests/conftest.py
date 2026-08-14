import json
from pathlib import Path

import pytest

from src.reconciler import align_words, reconcile
from src.transcription import load_transcript_words


ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "sample"


@pytest.fixture
def sample_aligned():
    words = load_transcript_words(SAMPLE / "transcript.txt")
    stt_raw = json.loads((SAMPLE / "stt_timestamps.json").read_text())["words"]
    meta_raw = json.loads((SAMPLE / "metadata_timestamps.json").read_text())["words"]

    from src.models import WordTimestamp

    metadata = [
        WordTimestamp(
            word=words[i],
            start=float(item["start"]),
            end=float(item["end"]),
            source="metadata",
        )
        for i, item in enumerate(meta_raw)
    ]
    stt = [
        WordTimestamp(
            word=words[i],
            start=float(item["start"]),
            end=float(item["end"]),
            confidence=float(item["confidence"]),
            source="stt",
        )
        for i, item in enumerate(stt_raw)
    ]
    return align_words(words, metadata, stt)


@pytest.fixture
def sample_result(sample_aligned):
    return reconcile(sample_aligned)
