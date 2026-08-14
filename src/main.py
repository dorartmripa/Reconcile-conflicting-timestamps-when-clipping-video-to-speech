"""CLI for the Timestamp Reconciliation Agent."""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

from src.cutter import cut_video, probe_duration
from src.metadata import load_metadata_timestamps
from src.reconciler import align_words, format_conflict_logs, reconcile
from src.report import generate_report
from src.transcription import load_transcript_words, transcribe


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Reconcile conflicting metadata and speech-to-text timestamps, "
            "then clip a video to the final speech window."
        )
    )
    parser.add_argument("--video", required=True, type=Path, help="Input video file")
    parser.add_argument(
        "--transcript", required=True, type=Path, help="Plain-text transcript"
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=None,
        help="Sidecar JSON of stale editor/ingest word timestamps",
    )
    parser.add_argument(
        "--stt-backend",
        choices=("demo", "whisper"),
        default="demo",
        help="demo uses a fixture (default). whisper runs faster-whisper locally.",
    )
    parser.add_argument(
        "--stt-json",
        type=Path,
        default=None,
        help="Demo STT fixture JSON (defaults to sample/stt_timestamps.json)",
    )
    parser.add_argument("--whisper-model", default="tiny.en")
    parser.add_argument("--output-dir", type=Path, default=Path("output"))
    parser.add_argument(
        "--padding",
        type=float,
        default=0.15,
        help="Seconds of padding on the clip start/end",
    )
    return parser


def run(args: argparse.Namespace) -> int:
    video_path = args.video.resolve()
    transcript_path = args.transcript.resolve()
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")
    if not transcript_path.exists():
        raise FileNotFoundError(f"Transcript not found: {transcript_path}")

    transcript_words = load_transcript_words(transcript_path)
    transcript_text = transcript_path.read_text(encoding="utf-8").strip()

    metadata_words, metadata_note = load_metadata_timestamps(
        video_path, transcript_words, args.metadata
    )
    stt_words, stt_note = transcribe(
        video_path,
        transcript_words,
        backend=args.stt_backend,
        stt_json=args.stt_json,
        whisper_model=args.whisper_model,
    )

    aligned = align_words(transcript_words, metadata_words, stt_words)
    result = reconcile(aligned, padding=args.padding)

    print(f"Words aligned: {len(aligned)}")
    print(f"Conflicts > 0.5s: {len(result.conflicts)}")
    print(f"Metadata source: {metadata_note}")
    print(f"STT source: {stt_note}")
    for note in result.notes:
        print(note)
    print()
    if result.conflicts:
        print(format_conflict_logs(result.conflicts))
        print()
    else:
        print("No conflicts above 0.5 seconds.\n")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    clip_path = output_dir / f"clip_{stamp}.mp4"
    report_path = output_dir / f"report_{stamp}.html"

    clip_start = result.clip_start
    clip_end = result.clip_end
    cut_video(video_path, clip_start, clip_end, clip_path)
    actual_duration = probe_duration(clip_path)

    generate_report(
        result=result,
        video_path=video_path,
        transcript_path=transcript_path,
        transcript_text=transcript_text,
        metadata_note=metadata_note,
        stt_note=stt_note,
        output_video=clip_path,
        clip_start=clip_start,
        clip_end=clip_end,
        output_duration=actual_duration,
        report_path=report_path,
    )

    print(f"Output video: {clip_path}")
    print(f"Clip window: {clip_start:.3f}s → {clip_end:.3f}s")
    if actual_duration is not None:
        print(f"Output duration: {actual_duration:.3f}s")
    print(f"HTML report: {report_path}")
    return 0


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    try:
        raise SystemExit(run(args))
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
