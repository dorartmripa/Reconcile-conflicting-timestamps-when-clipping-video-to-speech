"""Cut a video with FFmpeg using the reconciled timestamp window."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path


def find_ffmpeg() -> str:
    path = shutil.which("ffmpeg")
    if path:
        return path
    for candidate in (
        "/opt/homebrew/bin/ffmpeg",
        "/usr/local/bin/ffmpeg",
    ):
        if Path(candidate).exists():
            return candidate
    raise FileNotFoundError(
        "FFmpeg was not found on PATH. Install it (e.g. `brew install ffmpeg`)."
    )


def cut_video(
    video_path: Path,
    start: float,
    end: float,
    output_path: Path,
) -> Path:
    if end <= start:
        raise ValueError(f"Invalid clip window: start={start:.3f}, end={end:.3f}")

    ffmpeg = find_ffmpeg()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    duration = end - start

    command = [
        ffmpeg,
        "-y",
        "-ss",
        f"{start:.3f}",
        "-i",
        str(video_path),
        "-t",
        f"{duration:.3f}",
        "-c:v",
        "libx264",
        "-c:a",
        "aac",
        "-movflags",
        "+faststart",
        str(output_path),
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0 or not output_path.exists():
        raise RuntimeError(
            "FFmpeg failed to cut the video.\n"
            f"Command: {' '.join(command)}\n"
            f"{result.stderr[-2000:]}"
        )
    return output_path


def probe_duration(video_path: Path) -> float | None:
    ffprobe = shutil.which("ffprobe") or "/opt/homebrew/bin/ffprobe"
    if not Path(ffprobe).exists():
        return None
    result = subprocess.run(
        [
            ffprobe,
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(video_path),
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return None
    try:
        return float(result.stdout.strip())
    except ValueError:
        return None
