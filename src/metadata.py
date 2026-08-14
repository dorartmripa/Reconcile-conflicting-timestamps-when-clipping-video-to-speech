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

from src.models import WordTimestamp


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

    if not metadata_path.exists():
        raise FileNotFoundError(
            "No per-word timestamps in the video container, and no metadata "
            f"sidecar found at {metadata_path}. Provide --metadata."
        )

    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
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


def _read_chapters(video_path: Path) -> list[dict]:
    """Best-effort chapter extraction; empty if ffprobe/chapters are missing."""

    try:
        result = subprocess.run(
            [
                "ffprobe",
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
