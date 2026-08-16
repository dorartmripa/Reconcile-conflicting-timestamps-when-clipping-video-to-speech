"""Video metadata timestamp source.

Real MP4 files rarely contain per-word timestamps. Editors and ingest
pipelines *do* often store approximate word/marker times that go stale
when the timeline is recut or when audio is shifted.

This module:

1. Tries to read chapter markers from the video (ffprobe) if present.
2. Otherwise loads a sidecar metadata file that represents those stale
   editor/ingest markers.

The sidecar is a stub of the *source*, not of the reconciliation logic.
Conflict detection and the decision engine always operate on real numbers.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from backend.cutter import find_ffprobe
from backend.models import WordTimestamp


METADATA_SOURCE_NAME = "stale_editor_markers"


def load_metadata_timestamps(
    video_path: Path,
    transcript_words: list[str],
    metadata_path: Path | None = None,
) -> tuple[list[WordTimestamp], str]:
    """Return per-word metadata timestamps and a human-readable origin note."""

    chapters = _read_chapters(video_path)
    if chapters and len(chapters) >= len(transcript_words):
        words = [
            WordTimestamp(
                word=transcript_words[i],
                start=float(chapters[i]["start"]),
                end=float(chapters[i].get("end", chapters[i]["start"] + 0.3)),
                confidence=None,
                source="video_chapters",
            )
            for i in range(len(transcript_words))
        ]
        return words, "Extracted chapter timestamps from the video container."

    if metadata_path is None:
        metadata_path = video_path.parent / "metadata_timestamps.json"

    if metadata_path.exists():
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        return timestamps_from_payload(payload, transcript_words)

    raise FileNotFoundError(
        "No per-word timestamps in the video container, and no metadata "
        f"sidecar found at {metadata_path}. Provide --metadata."
    )


def timestamps_from_payload(
    payload: dict,
    transcript_words: list[str],
) -> tuple[list[WordTimestamp], str]:
    raw_words = payload["words"]
    if len(raw_words) != len(transcript_words):
        raise ValueError(
            f"Metadata has {len(raw_words)} words but transcript has "
            f"{len(transcript_words)}."
        )

    words = [
        WordTimestamp(
            word=transcript_words[i],
            start=float(raw_words[i]["start"]),
            end=float(raw_words[i]["end"]),
            confidence=None,
            source=METADATA_SOURCE_NAME,
        )
        for i in range(len(transcript_words))
    ]
    note = payload.get(
        "description",
        "Loaded stale editor/ingest markers from a sidecar JSON file.",
    )
    return words, note


def naive_ingest_markers(
    transcript_words: list[str],
    duration: float,
) -> tuple[list[WordTimestamp], str]:
    """Evenly spaced markers across media duration.

    Independent of STT: this is what a first ingest pass might store before
    speech recognition exists. Not derived from the current ASR times.
    """

    count = len(transcript_words)
    duration = max(float(duration), max(count, 1) * 0.35)
    slot = duration / count
    words = []
    for index, token in enumerate(transcript_words):
        start = index * slot
        end = start + min(0.45, slot * 0.75)
        words.append(
            WordTimestamp(
                word=token,
                start=start,
                end=end,
                confidence=None,
                source="naive_ingest_markers",
            )
        )
    note = (
        "No chapter markers or sidecar were provided. Used naive even-spacing "
        "ingest markers across the media duration (independent of STT)."
    )
    return words, note


def stale_metadata_from_stt(
    stt_words: list[WordTimestamp],
    transcript_words: list[str],
) -> tuple[list[WordTimestamp], str]:
    """Legacy helper kept for tests. The live pipeline does not call this.

    It is not an independent source: it copies STT times and adds lag.
    """

    n = len(stt_words)
    conflict_indices: set[int] = set()
    if n >= 4:
        conflict_indices = {max(1, n // 4), n // 2, min(n - 2, (3 * n) // 4), n - 1}
    offsets = {
        max(1, n // 4): 0.75,
        n // 2: 0.70,
        min(n - 2, (3 * n) // 4): 0.85,
        n - 1: 0.85,
    }

    words = []
    for i, item in enumerate(stt_words):
        lag = offsets[i] if i in conflict_indices else 0.03
        start = max(0.0, item.start + lag)
        end = max(start + 0.05, item.end + lag)
        words.append(
            WordTimestamp(
                word=transcript_words[i],
                start=start,
                end=end,
                confidence=None,
                source=METADATA_SOURCE_NAME,
            )
        )
    note = (
        "Synthesized stale markers from STT (not used by the live pipeline)."
    )
    return words, note


def _read_chapters(video_path: Path) -> list[dict]:
    """Best-effort chapter extraction; empty if ffprobe/chapters are missing."""

    try:
        ffprobe = find_ffprobe()
    except FileNotFoundError:
        return []
    try:
        result = subprocess.run(
            [
                ffprobe,
                "-v",
                "error",
                "-print_format",
                "json",
                "-show_chapters",
                str(video_path),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return []

    data = json.loads(result.stdout or "{}")
    chapters = []
    for chapter in data.get("chapters") or []:
        start = chapter.get("start_time")
        end = chapter.get("end_time")
        if start is None:
            continue
        chapters.append({"start": float(start), "end": float(end or start)})
    return chapters
