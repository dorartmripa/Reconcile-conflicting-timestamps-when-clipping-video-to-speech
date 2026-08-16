from pathlib import Path

from backend.identity import file_fingerprint, is_same_media
from backend.pipeline import is_bundled_sample
from backend.transcription import parse_transcript_text


ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "sample"


def test_sample_video_matches_itself():
    video = SAMPLE / "input.mp4"
    assert is_same_media(video, video)
    assert file_fingerprint(video) == file_fingerprint(video)


def test_fixture_requires_sample_video_not_just_transcript(tmp_path):
    transcript = parse_transcript_text((SAMPLE / "transcript.txt").read_text())
    other = tmp_path / "other.mp4"
    other.write_bytes((SAMPLE / "input.mp4").read_bytes() + b"not-the-sample")
    assert is_bundled_sample(SAMPLE / "input.mp4", transcript) is True
    assert is_bundled_sample(other, transcript) is False
