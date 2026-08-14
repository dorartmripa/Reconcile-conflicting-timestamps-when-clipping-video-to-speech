"""Speech-to-text timestamps.

Default demo backend loads a fixture so the assessment can be demonstrated
without downloading a Whisper model.

The whisper backend uses faster-whisper locally when installed, and
captures per-word probability as confidence.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from src.models import WordTimestamp


def normalize_word(word: str) -> str:
    return re.sub(r"[^a-z0-9']+", "", word.lower())


def load_transcript_words(transcript_path: Path) -> list[str]:
    text = transcript_path.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"Transcript is empty: {transcript_path}")
    return [part for part in re.split(r"\s+", text) if part]


def transcribe(
    video_path: Path,
    transcript_words: list[str],
    backend: str = "demo",
    stt_json: Path | None = None,
    whisper_model: str = "tiny.en",
) -> tuple[list[WordTimestamp], str]:
    if backend == "demo":
        return _demo_transcribe(video_path, transcript_words, stt_json)
    if backend == "whisper":
        return _whisper_transcribe(video_path, transcript_words, whisper_model)
    raise ValueError(f"Unknown STT backend: {backend}")


def _demo_transcribe(
    video_path: Path,
    transcript_words: list[str],
    stt_json: Path | None,
) -> tuple[list[WordTimestamp], str]:
    path = stt_json or (video_path.parent / "stt_timestamps.json")
    if not path.exists():
        raise FileNotFoundError(
            f"Demo STT fixture not found at {path}. Pass --stt-json or "
            "use --stt-backend whisper."
        )

    payload = json.loads(path.read_text(encoding="utf-8"))
    raw_words = payload["words"]
    if len(raw_words) != len(transcript_words):
        raise ValueError(
            f"STT fixture has {len(raw_words)} words but transcript has "
            f"{len(transcript_words)}."
        )

    words = [
        WordTimestamp(
            word=transcript_words[i],
            start=float(raw_words[i]["start"]),
            end=float(raw_words[i]["end"]),
            confidence=float(raw_words[i]["confidence"]),
            source="stt_demo",
        )
        for i in range(len(transcript_words))
    ]
    note = payload.get(
        "description",
        "Loaded demo speech-to-text word timestamps (Whisper-shaped fixture).",
    )
    return words, note


def _whisper_transcribe(
    video_path: Path,
    transcript_words: list[str],
    model_size: str,
) -> tuple[list[WordTimestamp], str]:
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise ImportError(
            "faster-whisper is required for --stt-backend whisper. "
            "Install with: pip install faster-whisper"
        ) from exc

    model = WhisperModel(model_size, device="cpu", compute_type="int8")
    segments, _info = model.transcribe(
        str(video_path),
        word_timestamps=True,
        language="en",
    )

    stt_words: list[WordTimestamp] = []
    for segment in segments:
        for word in segment.words or []:
            token = (word.word or "").strip()
            if not token:
                continue
            stt_words.append(
                WordTimestamp(
                    word=token,
                    start=float(word.start),
                    end=float(word.end),
                    confidence=float(word.probability),
                    source="whisper",
                )
            )

    aligned = _align_to_transcript(transcript_words, stt_words)
    note = (
        f"Generated word timestamps with faster-whisper ({model_size}). "
        "Confidence is the model's per-word probability."
    )
    return aligned, note


def _align_to_transcript(
    transcript_words: list[str],
    stt_words: list[WordTimestamp],
) -> list[WordTimestamp]:
    """Greedy sequential match of STT tokens onto the expected transcript."""

    aligned: list[WordTimestamp] = []
    cursor = 0
    for expected in transcript_words:
        target = normalize_word(expected)
        match: WordTimestamp | None = None
        for j in range(cursor, len(stt_words)):
            if normalize_word(stt_words[j].word) == target:
                match = stt_words[j]
                cursor = j + 1
                break
        if match is None:
            raise ValueError(
                f"Whisper output did not contain transcript word '{expected}'."
            )
        aligned.append(
            WordTimestamp(
                word=expected,
                start=match.start,
                end=match.end,
                confidence=match.confidence,
                source=match.source,
            )
        )
    return aligned
