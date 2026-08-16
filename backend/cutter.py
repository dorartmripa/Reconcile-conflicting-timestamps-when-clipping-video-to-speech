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


def find_ffprobe() -> str:
    path = shutil.which("ffprobe")
    if path:
        return path
    for candidate in (
        "/opt/homebrew/bin/ffprobe",
        "/usr/local/bin/ffprobe",
    ):
        if Path(candidate).exists():
            return candidate
    raise FileNotFoundError("ffprobe was not found on PATH.")


def extract_audio(video_path: Path, wav_path: Path) -> Path:
    """Extract 16 kHz mono WAV for STT. This is a real FFmpeg step."""

    ffmpeg = find_ffmpeg()
    wav_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        ffmpeg,
        "-y",
        "-i",
        str(video_path),
        "-vn",
        "-acodec",
        "pcm_s16le",
        "-ar",
        "16000",
        "-ac",
        "1",
        str(wav_path),
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0 or not wav_path.exists():
        raise RuntimeError(
            "FFmpeg failed to extract audio.\n"
            f"Command: {' '.join(command)}\n"
            f"{result.stderr[-2000:]}"
        )
    return wav_path


def cut_video(
    video_path: Path,
    start: float,
    end: float,
    output_path: Path,
    media_duration: float | None = None,
) -> Path:
    if media_duration is None:
        media_duration = probe_duration(video_path)
    start = max(0.0, start)
    if media_duration is not None:
        end = min(end, media_duration)
        start = min(start, max(0.0, media_duration - 0.05))
    if end <= start:
        end = start + 0.05
        if media_duration is not None:
            end = min(end, media_duration)
            if end <= start:
                raise ValueError(
                    f"Invalid clip window after duration cap: start={start:.3f}, end={end:.3f}"
                )

    ffmpeg = find_ffmpeg()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Output-side seek: decode then cut, more accurate for speech boundaries.
    duration = end - start
    command = [
        ffmpeg,
        "-y",
        "-i",
        str(video_path),
        "-ss",
        f"{start:.3f}",
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
    try:
        ffprobe = find_ffprobe()
    except FileNotFoundError:
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
