"""Speech-to-text timestamps.

The default demo backend loads a fixture so the bundled sample can run
without a model download. Any other video uses local faster-whisper.
Transcript words are aligned to ASR tokens with a sequence alignment
(match / similar / gap), then remaining holes are interpolated.
"""

from __future__ import annotations

import json
import re
import threading
from pathlib import Path

from backend.models import WordTimestamp


_MODEL_CACHE: dict[str, object] = {}
_WHISPER_LOCK = threading.Lock()


def normalize_word(word: str) -> str:
    return re.sub(r"[^a-z0-9']+", "", word.lower())


def parse_transcript_text(text: str) -> list[str]:
    cleaned = (text or "").strip()
    if not cleaned:
        raise ValueError("Transcript is empty.")
    return [part for part in re.split(r"\s+", cleaned) if part]


def load_transcript_words(transcript_path: Path) -> list[str]:
    return parse_transcript_text(transcript_path.read_text(encoding="utf-8"))


def whisper_available() -> bool:
    try:
        import faster_whisper  # noqa: F401
    except ImportError:
        return False
    return True


def transcribe(
    video_path: Path,
    transcript_words: list[str],
    backend: str = "demo",
    stt_json: Path | None = None,
    whisper_model: str = "tiny.en",
    vad_filter: bool = False,
) -> tuple[list[WordTimestamp], str]:
    if backend == "demo":
        return _demo_transcribe(video_path, transcript_words, stt_json)
    if backend == "whisper":
        return _whisper_transcribe(
            video_path, transcript_words, whisper_model, vad_filter=vad_filter
        )
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


def _load_whisper_model(model_size: str):
    if model_size in _MODEL_CACHE:
        return _MODEL_CACHE[model_size]
    from faster_whisper import WhisperModel

    last_error: Exception | None = None
    for compute_type in ("int8", "float32"):
        try:
            model = WhisperModel(model_size, device="cpu", compute_type=compute_type)
            _MODEL_CACHE[model_size] = model
            return model
        except Exception as exc:  # noqa: BLE001 - try the next compute type
            last_error = exc
    raise RuntimeError(f"Could not load Whisper model '{model_size}': {last_error}")


def _whisper_transcribe(
    video_path: Path,
    transcript_words: list[str],
    model_size: str,
    vad_filter: bool = False,
) -> tuple[list[WordTimestamp], str]:
    if not whisper_available():
        raise ImportError(
            "faster-whisper is not installed in this Python environment. "
            "Stop the backend, run `pip install -r requirements.txt`, then start it again."
        )

    prompt = " ".join(transcript_words)[:400]
    with _WHISPER_LOCK:
        model = _load_whisper_model(model_size)
        segments, _info = model.transcribe(
            str(video_path),
            word_timestamps=True,
            language="en",
            initial_prompt=prompt or None,
            vad_filter=vad_filter,
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

    if not stt_words:
        raise ValueError(
            "Whisper produced no word timestamps. The video may have no speech, "
            "or the audio track could not be read."
        )

    aligned = _align_to_transcript(transcript_words, stt_words)
    interpolated = sum(1 for item in aligned if item.interpolated)
    note = (
        f"Generated word timestamps with faster-whisper ({model_size}"
        f"{', VAD on' if vad_filter else ', VAD off'}). "
        "Confidence is the model's per-word probability. "
        "Tokens were sequence-aligned onto the transcript."
    )
    if interpolated:
        note += (
            f" {interpolated} transcript word(s) were interpolated after alignment."
        )
    return aligned, note


def _token_cost(expected: str, observed: str) -> float:
    left = normalize_word(expected)
    right = normalize_word(observed)
    if not left or not right:
        return 1.0
    if left == right:
        return 0.0
    if left.startswith(right) or right.startswith(left):
        return 0.3
    if len(left) >= 4 and len(right) >= 4 and left[:4] == right[:4]:
        return 0.35
    return 1.0


def _align_to_transcript(
    transcript_words: list[str],
    stt_words: list[WordTimestamp],
) -> list[WordTimestamp]:
    """Needleman–Wunsch-style alignment, then interpolate unmatched transcript words."""

    n = len(transcript_words)
    m = len(stt_words)
    gap = 0.8
    inf = 10 ** 9
    dp = [[inf] * (m + 1) for _ in range(n + 1)]
    ptr = [[None] * (m + 1) for _ in range(n + 1)]
    dp[0][0] = 0.0
    for i in range(1, n + 1):
        dp[i][0] = i * gap
        ptr[i][0] = "up"
    for j in range(1, m + 1):
        dp[0][j] = j * gap
        ptr[0][j] = "left"

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            diag = dp[i - 1][j - 1] + _token_cost(transcript_words[i - 1], stt_words[j - 1].word)
            up = dp[i - 1][j] + gap
            left = dp[i][j - 1] + gap
            best = min(diag, up, left)
            dp[i][j] = best
            if best == diag:
                ptr[i][j] = "diag"
            elif best == up:
                ptr[i][j] = "up"
            else:
                ptr[i][j] = "left"

    matches: list[WordTimestamp | None] = [None] * n
    i, j = n, m
    while i > 0 or j > 0:
        move = ptr[i][j]
        if move == "diag":
            if _token_cost(transcript_words[i - 1], stt_words[j - 1].word) < 0.8:
                matches[i - 1] = stt_words[j - 1]
            i -= 1
            j -= 1
        elif move == "up":
            i -= 1
        elif move == "left":
            j -= 1
        else:
            break

    span_start = stt_words[0].start
    span_end = max(stt_words[-1].end, span_start + 0.05)
    aligned: list[WordTimestamp] = []
    for index, expected in enumerate(transcript_words):
        hit = matches[index]
        if hit is not None:
            aligned.append(
                WordTimestamp(
                    word=expected,
                    start=hit.start,
                    end=max(hit.end, hit.start + 0.05),
                    confidence=hit.confidence,
                    source=hit.source or "whisper",
                    interpolated=False,
                )
            )
            continue
        prev = next((matches[k] for k in range(index - 1, -1, -1) if matches[k]), None)
        nxt = next((matches[k] for k in range(index + 1, n) if matches[k]), None)
        if prev and nxt:
            start = prev.end
            end = max(start + 0.05, min(nxt.start, start + 0.35))
        elif prev:
            start = prev.end + 0.04
            end = start + 0.25
        elif nxt:
            end = nxt.start
            start = max(0.0, end - 0.25)
        else:
            frac = index / max(n - 1, 1)
            start = span_start + frac * (span_end - span_start)
            end = start + 0.25
        aligned.append(
            WordTimestamp(
                word=expected,
                start=max(0.0, start),
                end=max(end, start + 0.05),
                confidence=0.28,
                source="whisper_interpolated",
                interpolated=True,
            )
        )
    return aligned
