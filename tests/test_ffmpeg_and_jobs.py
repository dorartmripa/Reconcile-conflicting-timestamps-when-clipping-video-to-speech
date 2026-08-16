from pathlib import Path

import pytest

from backend.cutter import cut_video


def test_cut_video_missing_ffmpeg(monkeypatch, tmp_path):
    def boom():
        raise FileNotFoundError("FFmpeg was not found on PATH.")

    monkeypatch.setattr("backend.cutter.find_ffmpeg", boom)
    monkeypatch.setattr("backend.cutter.probe_duration", lambda path: 10.0)
    with pytest.raises(FileNotFoundError):
        cut_video(tmp_path / "in.mp4", 0.0, 1.0, tmp_path / "out.mp4", media_duration=10.0)


def test_job_json_is_persisted_after_api_run():
    from fastapi.testclient import TestClient
    from backend.app import OUTPUT, app

    sample = Path(__file__).resolve().parents[1] / "sample"
    client = TestClient(app)
    transcript = (sample / "transcript.txt").read_text()
    with (sample / "input.mp4").open("rb") as video:
        response = client.post(
            "/api/jobs",
            files={"video": ("input.mp4", video, "video/mp4")},
            data={"transcript_text": transcript},
        )
    job_id = response.json()["id"]
    assert (OUTPUT / job_id / "job.json").exists()
