import json
import time
from pathlib import Path

from fastapi.testclient import TestClient

from backend.app import app


ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "sample"


def test_health():
    client = TestClient(app)
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json()["ok"] is True
    assert "whisper" in response.json()
    assert len(response.json()["steps"]) >= 6


def test_upload_analyze_returns_real_conflicts_and_clip():
    client = TestClient(app)
    transcript = (SAMPLE / "transcript.txt").read_text()
    with (SAMPLE / "input.mp4").open("rb") as video:
        response = client.post(
            "/api/jobs",
            files={"video": ("input.mp4", video, "video/mp4")},
            data={"transcript_text": transcript},
        )
    assert response.status_code == 200, response.text
    job_id = response.json()["id"]

    payload = None
    for _ in range(60):
        payload = client.get(f"/api/jobs/{job_id}").json()
        if payload["status"] in {"complete", "error"}:
            break
        time.sleep(0.25)

    assert payload["status"] == "complete", payload.get("error")
    result = payload["result"]
    assert result["conflicts_detected"] >= 3
    assert result["conflicts_resolved"] == result["conflicts_detected"]
    assert result["trusted_source"] in {"STT", "metadata", "ordering_repair"}
    assert Path(result["clip"]["path"]).exists()
    assert result["clip"]["size_bytes"] > 1000
    clip = client.get(f"/api/jobs/{job_id}/clip")
    assert clip.status_code == 200
    assert clip.headers["content-type"].startswith("video/")
    assert len(clip.content) > 1000

    words = {row["word"] for row in result["conflicts"]}
    assert "timestamp" in words
    decisions = {row["decision"] for row in result["conflicts"]}
    assert "STT" in decisions
    assert "metadata" in decisions
