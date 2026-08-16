"""Shared pipeline used by the CLI and the web API."""

from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from backend.cutter import cut_video, extract_audio, probe_duration
from backend.identity import is_same_media
from backend.metadata import (
    load_metadata_timestamps,
    naive_ingest_markers,
    timestamps_from_payload,
)
from backend.models import ReconciliationResult, WordTimestamp
from backend.reconciler import align_words, detect_conflicts, reconcile
from backend.report import generate_report
from backend.transcription import (
    parse_transcript_text,
    transcribe,
    whisper_available,
)


ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "sample"

ProgressFn = Callable[[str, str], None]

STEPS = [
    ("extract_audio", "Extracting the video's audio"),
    ("stt", "Running speech-to-text"),
    ("metadata", "Loading metadata timestamps"),
    ("compare", "Comparing the two timestamp sources"),
    ("conflicts", "Detecting conflicts greater than 0.5 seconds"),
    ("reconcile", "Applying the reconciliation decision engine"),
    ("cut", "Cutting the video with FFmpeg"),
]


def is_bundled_sample(
    video_path: Path,
    transcript_words: list[str],
) -> bool:
    sample_video = SAMPLE / "input.mp4"
    sample_words = parse_transcript_text((SAMPLE / "transcript.txt").read_text())
    return transcript_words == sample_words and is_same_media(video_path, sample_video)


def run_pipeline(
    *,
    video_path: Path,
    transcript_text: str,
    work_dir: Path,
    metadata_path: Path | None = None,
    stt_backend: str = "auto",
    stt_json: Path | None = None,
    whisper_model: str = "tiny.en",
    vad_filter: bool = False,
    padding: float = 0.15,
    progress: ProgressFn | None = None,
) -> dict:
    def report(step: str, detail: str) -> None:
        if progress:
            progress(step, detail)

    work_dir.mkdir(parents=True, exist_ok=True)
    transcript_words = parse_transcript_text(transcript_text)
    transcript_file = work_dir / "transcript.txt"
    transcript_file.write_text(transcript_text.strip() + "\n", encoding="utf-8")

    report("extract_audio", "Extracting 16 kHz mono audio with FFmpeg")
    wav_path = work_dir / "audio.wav"
    extract_audio(video_path, wav_path)
    media_duration = probe_duration(video_path)

    bundled = is_bundled_sample(video_path, transcript_words)
    report(
        "stt",
        (
            "Loading demo STT fixture for the bundled sample"
            if bundled and stt_backend in {"auto", "demo"}
            else f"Loading Whisper {whisper_model} and transcribing"
        ),
    )
    stt_words, stt_note, resolved_stt = _resolve_stt(
        audio_path=wav_path,
        video_path=video_path,
        transcript_words=transcript_words,
        stt_backend=stt_backend,
        stt_json=stt_json,
        whisper_model=whisper_model,
        vad_filter=vad_filter,
        bundled_sample=bundled,
    )

    report("metadata", "Loading metadata timestamps")
    metadata_words, metadata_note, metadata_kind = _resolve_metadata(
        video_path=video_path,
        transcript_words=transcript_words,
        metadata_path=metadata_path,
        media_duration=media_duration,
        bundled_sample=bundled,
        audio_duration=probe_duration(wav_path),
    )

    report("compare", "Aligning metadata and STT on the transcript")
    aligned = align_words(transcript_words, metadata_words, stt_words)

    report("conflicts", "Flagging pairs that differ by more than 0.5 seconds")
    conflict_pairs = detect_conflicts(aligned)

    report("reconcile", "Scoring both sources and choosing a final timestamp")
    result = reconcile(aligned, padding=padding)

    report("cut", "Cutting the clip from the reconciled start and end")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    clip_path = work_dir / f"clip_{stamp}.mp4"
    cut_video(
        video_path,
        result.clip_start,
        result.clip_end,
        clip_path,
        media_duration=media_duration,
    )
    actual_duration = probe_duration(clip_path)

    html_path = work_dir / f"report_{stamp}.html"
    generate_report(
        result=result,
        video_path=video_path,
        transcript_path=transcript_file,
        transcript_text=transcript_text.strip(),
        metadata_note=metadata_note,
        stt_note=stt_note,
        output_video=clip_path,
        clip_start=result.clip_start,
        clip_end=result.clip_end,
        output_duration=actual_duration,
        report_path=html_path,
    )

    payload = serialize_result(
        result=result,
        transcript_text=transcript_text.strip(),
        metadata_note=metadata_note,
        stt_note=stt_note,
        stt_backend=resolved_stt,
        whisper_model=whisper_model if resolved_stt == "whisper" else None,
        metadata_kind=metadata_kind,
        clip_path=clip_path,
        clip_start=result.clip_start,
        clip_end=result.clip_end,
        output_duration=actual_duration,
        audio_path=wav_path,
        conflict_count=len(conflict_pairs),
    )
    (work_dir / "result.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    return payload


def _resolve_stt(
    *,
    audio_path: Path,
    video_path: Path,
    transcript_words: list[str],
    stt_backend: str,
    stt_json: Path | None,
    whisper_model: str,
    vad_filter: bool,
    bundled_sample: bool,
) -> tuple[list[WordTimestamp], str, str]:
    sample_stt = SAMPLE / "stt_timestamps.json"

    backend = stt_backend
    if backend == "auto":
        if bundled_sample and sample_stt.exists():
            backend = "demo"
        elif whisper_available():
            backend = "whisper"
        else:
            raise RuntimeError(
                "Speech-to-text is unavailable for this video. "
                "Restart the backend after `pip install -r requirements.txt` "
                "(includes faster-whisper), or analyze the bundled sample video "
                "with the sample transcript."
            )

    if backend == "demo":
        if not bundled_sample and stt_json is None:
            raise RuntimeError(
                "Demo STT fixtures are only used for the bundled sample video. "
                "This file's fingerprint does not match sample/input.mp4."
            )
        fixture = stt_json or sample_stt
        words, note = transcribe(
            video_path,
            transcript_words,
            backend="demo",
            stt_json=fixture,
        )
        return words, note, "demo"

    words, note = transcribe(
        audio_path,
        transcript_words,
        backend="whisper",
        whisper_model=whisper_model,
        vad_filter=vad_filter,
    )
    return words, note, "whisper"


def _resolve_metadata(
    *,
    video_path: Path,
    transcript_words: list[str],
    metadata_path: Path | None,
    media_duration: float | None,
    bundled_sample: bool,
    audio_duration: float | None,
) -> tuple[list[WordTimestamp], str, str]:
    if metadata_path and metadata_path.exists():
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        words, note = timestamps_from_payload(payload, transcript_words)
        return words, note, "sidecar"

    if bundled_sample:
        sample_meta = SAMPLE / "metadata_timestamps.json"
        payload = json.loads(sample_meta.read_text(encoding="utf-8"))
        words, note = timestamps_from_payload(payload, transcript_words)
        return words, note, "sample"

    try:
        words, note = load_metadata_timestamps(video_path, transcript_words, None)
        kind = "chapters" if "chapter" in note.lower() else "sidecar"
        return words, note, kind
    except FileNotFoundError:
        duration = media_duration or audio_duration or max(len(transcript_words) * 0.5, 1.0)
        words, note = naive_ingest_markers(transcript_words, duration)
        return words, note, "naive_ingest"


def serialize_result(
    *,
    result: ReconciliationResult,
    transcript_text: str,
    metadata_note: str,
    stt_note: str,
    stt_backend: str,
    whisper_model: str | None,
    metadata_kind: str,
    clip_path: Path,
    clip_start: float,
    clip_end: float,
    output_duration: float | None,
    audio_path: Path,
    conflict_count: int,
) -> dict:
    decisions = Counter(conflict.decision for conflict in result.conflicts)
    trusted = "none"
    if decisions:
        trusted = decisions.most_common(1)[0][0]

    timeline = []
    for item, final in zip(result.aligned, result.final_words):
        timeline.append(
            {
                "index": item.index,
                "word": item.word,
                "metadata": round(item.metadata.start, 4),
                "stt": round(item.stt.start, 4),
                "final": round(final.start, 4),
                "difference": round(item.difference, 4),
                "end_difference": round(item.end_difference, 4),
                "confidence": item.stt.confidence,
                "conflict": item.is_conflict,
                "decision": final.source,
                "interpolated": bool(final.interpolated or item.stt.interpolated),
                "trusted": final.trusted,
            }
        )

    conflicts = []
    for conflict in result.conflicts:
        conflicts.append(
            {
                "number": conflict.number,
                "word": conflict.word,
                "metadata": round(conflict.metadata_timestamp, 4),
                "stt": round(conflict.stt_timestamp, 4),
                "difference": round(conflict.difference, 4),
                "confidence": conflict.stt_confidence,
                "decision": conflict.decision,
                "reason": conflict.reason,
                "final": round(conflict.final_timestamp, 4),
                "factors": conflict.factors,
                "interpolated": bool(conflict.factors.get("interpolated_stt")),
            }
        )

    return {
        "status": "complete",
        "words": len(result.aligned),
        "conflicts_detected": conflict_count,
        "conflicts_resolved": len(result.conflicts),
        "trusted_source": trusted,
        "decision_counts": dict(decisions),
        "drift_detected": result.drift_detected,
        "estimated_offset": result.estimated_offset,
        "estimated_slope": result.estimated_slope,
        "offset_mad": result.offset_mad,
        "notes": result.notes,
        "transcript": transcript_text,
        "metadata_note": metadata_note,
        "metadata_kind": metadata_kind,
        "metadata_generated": metadata_kind == "naive_ingest",
        "stt_note": stt_note,
        "stt_backend": stt_backend,
        "whisper_model": whisper_model,
        "clip": {
            "path": str(clip_path),
            "start": clip_start,
            "end": clip_end,
            "requested_duration": clip_end - clip_start,
            "actual_duration": output_duration,
            "size_bytes": clip_path.stat().st_size if clip_path.exists() else 0,
        },
        "audio_path": str(audio_path),
        "timeline": timeline,
        "conflicts": conflicts,
    }
