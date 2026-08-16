"""CLI for the Timestamp Reconciliation Agent."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from backend.pipeline import run_pipeline
from backend.reconciler import format_conflict_logs


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
        choices=("auto", "demo", "whisper"),
        default="auto",
        help="auto uses Whisper if installed, otherwise the sample demo fixture.",
    )
    parser.add_argument(
        "--stt-json",
        type=Path,
        default=None,
        help="Demo STT fixture JSON",
    )
    parser.add_argument("--whisper-model", default="tiny.en")
    parser.add_argument("--vad-filter", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=Path("output"))
    parser.add_argument("--padding", type=float, default=0.15)
    return parser


def run(args: argparse.Namespace) -> int:
    video_path = args.video.resolve()
    transcript_path = args.transcript.resolve()
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")
    if not transcript_path.exists():
        raise FileNotFoundError(f"Transcript not found: {transcript_path}")

    transcript_text = transcript_path.read_text(encoding="utf-8")
    payload = run_pipeline(
        video_path=video_path,
        transcript_text=transcript_text,
        work_dir=args.output_dir.resolve(),
        metadata_path=args.metadata.resolve() if args.metadata else None,
        stt_backend=args.stt_backend,
        stt_json=args.stt_json,
        whisper_model=args.whisper_model,
        vad_filter=args.vad_filter,
        padding=args.padding,
    )

    print(f"Words aligned: {payload['words']}")
    print(f"Conflicts > 0.5s: {payload['conflicts_detected']}")
    print(f"Metadata source: {payload['metadata_note']}")
    print(f"STT source: {payload['stt_note']}")
    for note in payload["notes"]:
        print(note)
    print()

    from backend.models import ConflictLog

    logs = [
        ConflictLog(
            number=row["number"],
            word=row["word"],
            metadata_timestamp=row["metadata"],
            stt_timestamp=row["stt"],
            difference=row["difference"],
            stt_confidence=row["confidence"],
            factors=row["factors"],
            decision=row["decision"],
            reason=row["reason"],
            final_timestamp=row["final"],
        )
        for row in payload["conflicts"]
    ]
    if logs:
        print(format_conflict_logs(logs))
        print()
    else:
        print("No conflicts above 0.5 seconds.\n")

    clip = payload["clip"]
    print(f"Output video: {clip['path']}")
    print(f"Clip window: {clip['start']:.3f}s → {clip['end']:.3f}s")
    if clip["actual_duration"] is not None:
        print(f"Output duration: {clip['actual_duration']:.3f}s")
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
