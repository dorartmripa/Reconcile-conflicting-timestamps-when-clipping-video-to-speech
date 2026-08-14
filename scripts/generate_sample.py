"""Generate a short demo MP4 with spoken audio (macOS `say` + FFmpeg)."""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.cutter import find_ffmpeg


SAMPLE = ROOT / "sample"
TRANSCRIPT = SAMPLE / "transcript.txt"
OUTPUT = SAMPLE / "input.mp4"


def main() -> None:
    text = TRANSCRIPT.read_text(encoding="utf-8").strip()
    ffmpeg = find_ffmpeg()
    SAMPLE.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        aiff_path = Path(tmp) / "speech.aiff"
        wav_path = Path(tmp) / "speech.wav"
        subprocess.run(
            ["say", "-r", "160", "-o", str(aiff_path), text],
            check=True,
        )
        subprocess.run(
            [ffmpeg, "-y", "-i", str(aiff_path), str(wav_path)],
            check=True,
            capture_output=True,
        )
        command = [
            ffmpeg,
            "-y",
            "-f",
            "lavfi",
            "-i",
            "color=c=0x0f172a:s=1280x720:r=30:d=10",
            "-i",
            str(wav_path),
            "-filter_complex",
            "[1:a]apad[a]",
            "-map",
            "0:v",
            "-map",
            "[a]",
            "-t",
            "10",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-movflags",
            "+faststart",
            str(OUTPUT),
        ]
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(result.stderr[-2000:])

    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()
