"""FastAPI app: upload → analyze → reconcile → cut → report."""

from __future__ import annotations

import json
import queue
import threading
import uuid
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles

from backend.pipeline import STEPS, run_pipeline
from backend.transcription import whisper_available


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "output" / "jobs"
FRONTEND_DIST = ROOT / "frontend" / "dist"
ALLOWED_MODELS = ("tiny.en", "base.en", "small.en")

app = FastAPI(title="Timestamp Reconciliation Agent")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

JOBS: dict[str, dict[str, Any]] = {}
JOBS_LOCK = threading.Lock()
_JOB_QUEUE: queue.Queue[dict[str, Any]] = queue.Queue()
_WORKER_LOCK = threading.Lock()
_WORKER_STARTED = False


def _job_path(job_id: str) -> Path:
    return OUTPUT / job_id / "job.json"


def _persist(job: dict[str, Any]) -> None:
    path = _job_path(job["id"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(job, default=str), encoding="utf-8")


def _update_job(job_id: str, **fields: Any) -> None:
    with JOBS_LOCK:
        JOBS[job_id].update(fields)
        snapshot = dict(JOBS[job_id])
    _persist(snapshot)


def _load_job(job_id: str) -> dict[str, Any] | None:
    with JOBS_LOCK:
        if job_id in JOBS:
            return JOBS[job_id]
    path = _job_path(job_id)
    if not path.exists():
        return None
    job = json.loads(path.read_text(encoding="utf-8"))
    with JOBS_LOCK:
        JOBS[job_id] = job
    return job


def _ensure_worker() -> None:
    global _WORKER_STARTED
    with _WORKER_LOCK:
        if _WORKER_STARTED:
            return
        thread = threading.Thread(target=_worker_loop, daemon=True, name="reconcile-worker")
        thread.start()
        _WORKER_STARTED = True


def _worker_loop() -> None:
    while True:
        payload = _JOB_QUEUE.get()
        try:
            _run_job(**payload)
        finally:
            _JOB_QUEUE.task_done()


def _run_job(
    job_id: str,
    video_path: Path,
    transcript_text: str,
    metadata_path: Path | None,
    work_dir: Path,
    whisper_model: str,
    vad_filter: bool,
) -> None:
    def progress(step: str, detail: str) -> None:
        completed = [name for name, _ in STEPS]
        idx = completed.index(step) if step in completed else 0
        _update_job(
            job_id,
            status="running",
            step=step,
            step_label=detail,
            completed_steps=completed[:idx],
        )

    try:
        result = run_pipeline(
            video_path=video_path,
            transcript_text=transcript_text,
            work_dir=work_dir,
            metadata_path=metadata_path,
            stt_backend="auto",
            whisper_model=whisper_model,
            vad_filter=vad_filter,
            progress=progress,
        )
        _update_job(
            job_id,
            status="complete",
            step="complete",
            step_label="Done",
            completed_steps=[name for name, _ in STEPS],
            result=result,
        )
    except Exception as exc:
        _update_job(
            job_id,
            status="error",
            step="error",
            step_label=str(exc),
            error=str(exc),
        )


@app.get("/api/health")
def health() -> dict:
    return {
        "ok": True,
        "whisper": whisper_available(),
        "models": list(ALLOWED_MODELS),
        "steps": [{"id": sid, "label": label} for sid, label in STEPS],
    }


@app.get("/api/sample/transcript", response_class=PlainTextResponse)
def sample_transcript() -> str:
    path = ROOT / "sample" / "transcript.txt"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Sample transcript missing")
    return path.read_text(encoding="utf-8")


@app.post("/api/jobs")
async def create_job(
    video: UploadFile = File(...),
    transcript_text: str = Form(""),
    transcript_file: UploadFile | None = File(None),
    metadata_file: UploadFile | None = File(None),
    whisper_model: str = Form("tiny.en"),
    vad_filter: str = Form("false"),
) -> dict:
    if transcript_file is not None:
        raw = (await transcript_file.read()).decode("utf-8")
        if raw.strip():
            transcript_text = raw
    transcript_text = (transcript_text or "").strip()
    if not transcript_text:
        raise HTTPException(status_code=400, detail="Provide a transcript (paste or file).")
    if not video.filename:
        raise HTTPException(status_code=400, detail="Upload a video file.")
    if whisper_model not in ALLOWED_MODELS:
        raise HTTPException(status_code=400, detail=f"Unsupported model: {whisper_model}")
    vad_enabled = str(vad_filter).lower() in {"1", "true", "on", "yes"}

    job_id = uuid.uuid4().hex[:12]
    work_dir = OUTPUT / job_id
    work_dir.mkdir(parents=True, exist_ok=True)

    suffix = Path(video.filename).suffix or ".mp4"
    video_path = work_dir / f"input{suffix}"
    video_path.write_bytes(await video.read())
    if video_path.stat().st_size == 0:
        raise HTTPException(status_code=400, detail="Uploaded video is empty.")

    metadata_path = None
    if metadata_file is not None and metadata_file.filename:
        metadata_path = work_dir / "metadata_timestamps.json"
        metadata_path.write_bytes(await metadata_file.read())

    job = {
        "id": job_id,
        "status": "queued",
        "step": "queued",
        "step_label": "Queued behind other jobs",
        "completed_steps": [],
        "steps": [{"id": sid, "label": label} for sid, label in STEPS],
        "result": None,
        "error": None,
        "whisper_model": whisper_model,
        "vad_filter": vad_enabled,
    }
    with JOBS_LOCK:
        JOBS[job_id] = job
    _persist(job)
    _ensure_worker()
    _JOB_QUEUE.put(
        {
            "job_id": job_id,
            "video_path": video_path,
            "transcript_text": transcript_text,
            "metadata_path": metadata_path,
            "work_dir": work_dir,
            "whisper_model": whisper_model,
            "vad_filter": vad_enabled,
        }
    )
    return {"id": job_id}


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str) -> dict:
    job = _load_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


@app.get("/api/jobs/{job_id}/clip")
def get_clip(job_id: str, download: bool = False) -> FileResponse:
    job = _load_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    if job.get("status") != "complete" or not job.get("result"):
        raise HTTPException(status_code=409, detail="Clip is not ready")
    clip_path = Path(job["result"]["clip"]["path"])
    if not clip_path.exists():
        raise HTTPException(status_code=404, detail="Clip file missing")
    filename = "reconciled-clip.mp4"
    return FileResponse(
        clip_path,
        media_type="video/mp4",
        filename=filename if download else None,
        content_disposition_type="attachment" if download else "inline",
    )


@app.get("/")
def index():
    index_path = FRONTEND_DIST / "index.html"
    if index_path.exists():
        return FileResponse(index_path)
    return {
        "ok": True,
        "message": "API is running. Build the UI with: cd frontend && npm run build",
        "ui": None,
        "health": "/api/health",
    }


if (FRONTEND_DIST / "assets").exists():
    app.mount("/assets", StaticFiles(directory=FRONTEND_DIST / "assets"), name="assets")
