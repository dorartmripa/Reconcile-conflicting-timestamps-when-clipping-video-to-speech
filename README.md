# Timestamp Reconciliation Agent

Web app + Python backend that takes a **video** and a **transcript**, reads timestamps from two independent sources, detects conflicts, decides which time to trust, and **cuts a real clip with FFmpeg**.

The UI shows real backend results. Analyze is not a mock.

## Quick start (one link)

Needs **Python 3.10+** and **FFmpeg** (`brew install ffmpeg` on macOS). Node is only needed if you rebuild the UI.

```bash
git clone https://github.com/dorartmripa/Reconcile-conflicting-timestamps-when-clipping-video-to-speech.git
cd Reconcile-conflicting-timestamps-when-clipping-video-to-speech

python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python3 -m backend
```

Open **http://127.0.0.1:8000**

That single address serves both the UI and the API.

### Demo in the browser

1. Upload `sample/input.mp4`
2. Click **Load sample transcript**
3. Click **Analyze & Reconcile**

You should see ~4 conflicts, mixed STT/metadata decisions, a playable clip, and a download button.

**How to tell it worked:** the report appears, conflicts are listed, and the clip duration is shorter than the original (~10s → ~7s). The sample video is mostly a solid screen, so the picture can look similar even when the trim worked.

Interviewers cannot open *your* localhost. They clone this repo and run the same commands on their machine. To share a temporary public URL while your laptop is running:

```bash
npx cloudflared tunnel --url http://127.0.0.1:8000
```

## Dev UI (optional, two processes)

```bash
# terminal 1
uvicorn backend.app:app --reload --port 8000 --host 127.0.0.1

# terminal 2
cd frontend && npm install && npm run dev
```

Open **http://127.0.0.1:5173**

Rebuild the committed UI after frontend changes:

```bash
cd frontend && npm run build
```

## CLI and tests

```bash
python3 -m src.main --video sample/input.mp4 --transcript sample/transcript.txt
python3 -m pytest -q
```

## Architecture

```
Browser (http://127.0.0.1:8000)
    │
    ├─ static UI from frontend/dist
    └─ /api/* → backend/app.py
                    │
                    ├─ extract audio     backend/cutter.py
                    ├─ STT               backend/transcription.py
                    ├─ metadata          backend/metadata.py
                    ├─ reconcile         backend/reconciler.py
                    └─ FFmpeg cut        backend/cutter.py
```

| Path | Role |
|---|---|
| `frontend/` | React upload UI + report dashboard |
| `frontend/dist/` | Built UI served by FastAPI |
| `backend/` | FastAPI + reconciliation engine |
| `src/` | Thin re-exports for `python -m src.main` |
| `sample/` | Demo video, transcript, timestamp fixtures |
| `tests/` | pytest for engine + API |
| `output/jobs/` | Per-job audio, clip, and result JSON |

## Timestamp sources

**STT**

- Bundled sample video + matching transcript → Whisper-shaped fixture (`sample/stt_timestamps.json`) so the demo does not need a model download
- Any other video → local **faster-whisper** (`tiny.en` by default; UI can select `base.en` / `small.en`)
- Demo fixtures are gated on **video fingerprint + transcript**, not transcript text alone

**Metadata**

1. Uploaded sidecar JSON (if provided)
2. Bundled sample sidecar (only for the bundled sample video)
3. Chapter markers via `ffprobe` (rare on normal MP4s)
4. Otherwise **naive even-spacing ingest markers** across media duration (independent of STT)

The metadata source can be stubbed; the scoring engine is not.

## Reconciliation

A conflict is when start or end times differ by **more than 0.5s** (also interval containment). Each conflict is scored using:

- STT confidence
- Neighbor consistency (including overlap)
- Chronological order
- Linear drift (`offset + slope * t`)

Near-ties blend. If both sources break order, the word is marked **untrusted** and skipped for clip bounds. Weights are documented and checked against `sample/labeled_conflicts.json`.

Sample decisions:

| Word | Diff | Conf | Decision |
|---|---|---|---|
| timestamp | 0.75s | 0.96 | STT |
| system | 0.70s | 0.30 | metadata |
| conflicts | 0.85s | 0.94 | STT |
| reliably | 0.85s | 0.95 | STT |

The last decision changes the clip end.

## Known limitations

- Sample STT is a fixture unless the video is not the bundled sample
- Typical MP4s have no per-word metadata; sidecar / naive ingest stands in
- Score weights are hand-tuned (validated on a small labeled set)
- Jobs are queued in-process; Whisper runs under a lock
- One clip window (first trusted word → last), not a per-word montage
- First live Whisper run may download the model

## What I would improve with more time

- Stronger forced alignment (e.g. WhisperX)
- Phrase-level cuts / EDL export
- Larger labeled set to calibrate score weights
- Persist job queue beyond a single process
