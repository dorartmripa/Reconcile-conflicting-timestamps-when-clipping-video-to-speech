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

## Try it with your own video

1. Upload **your own** video (MP4/MOV with clear speech works best).
2. Paste or upload a transcript of the spoken words **in order**.
3. Optionally upload a metadata JSON sidecar of per-word times. If you skip this, the app uses chapter markers when present, otherwise naive even-spacing ingest markers (independent of STT).
4. Pick a Whisper model if you want (`tiny.en` is fastest; `base.en` / `small.en` are more accurate).
5. Click **Analyze & Reconcile**.

The first run on your own video downloads the Whisper model and can take a minute.

### How to tell it worked

- Processing steps complete with **no red error**
- A **report** appears: words aligned, conflicts detected/resolved, trusted source
- The **conflict table** lists decisions and reasons (if any gaps &gt; 0.5s)
- The **output clip plays** and **Download** saves an MP4
- Clip window / duration usually differs from the full upload (trim to the reconciled speech span)

If the picture looks similar to your upload, that can still be correct: this tool trims to a speech window, it does not restyle the video.

### Tips for your own footage

- Keep clips short (about 10–30 seconds) for a smooth first try
- Match the transcript to what is actually said (fillers and wrong words hurt alignment)
- Prefer clear English speech; noisy audio is harder for `tiny.en`
- Upload metadata JSON when you have editor/ingest times so both sources are realistic

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
| `sample/` | Optional fixtures for automated tests |
| `tests/` | pytest for engine + API |
| `output/jobs/` | Per-job audio, clip, and result JSON |

## Timestamp sources

**STT (speech-to-text)**

- Your own videos use local **faster-whisper** (`tiny.en` by default; UI can select `base.en` / `small.en`)
- Word times include confidence (model probability)
- Transcript tokens are sequence-aligned to ASR output; unmatched words are interpolated and labeled

**Metadata**

1. Uploaded sidecar JSON (recommended when you have it)
2. Chapter markers via `ffprobe` (rare on normal MP4s)
3. Otherwise **naive even-spacing ingest markers** across media duration (independent of STT)

The metadata source may be approximate; the scoring engine is real.

## Reconciliation

A conflict is when start or end times differ by **more than 0.5s** (also interval containment). Each conflict is scored using:

- STT confidence
- Neighbor consistency (including overlap)
- Chronological order
- Linear drift (`offset + slope * t`)

Near-ties blend. If both sources break order, the word is marked **untrusted** and skipped for clip bounds. Score weights are documented and checked against a small labeled set in `sample/labeled_conflicts.json`.

Example of mixed decisions (from the test fixtures):

| Word | Diff | Conf | Decision |
|---|---|---|---|
| timestamp | 0.75s | 0.96 | STT |
| system | 0.70s | 0.30 | metadata |
| conflicts | 0.85s | 0.94 | STT |
| reliably | 0.85s | 0.95 | STT |

High-confidence STT tends to win; low-confidence STT can lose to consistent metadata. The last trusted word helps set the clip end.

## CLI and tests

```bash
# Your video
python3 -m src.main --video /path/to/video.mp4 --transcript /path/to/transcript.txt

# Optional metadata sidecar
python3 -m src.main --video /path/to/video.mp4 --transcript /path/to/transcript.txt --metadata /path/to/metadata_timestamps.json

python3 -m pytest -q
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

## Known limitations

- First Whisper run downloads the model and is slower than later runs
- Typical MP4s have no per-word metadata; sidecar / naive ingest stands in
- Score weights are hand-tuned (validated on a small labeled set)
- Jobs are queued in-process; Whisper runs under a lock
- One clip window (first trusted word → last), not a per-word montage
- Alignment quality depends on transcript accuracy and model size

## What I would improve with more time

- Stronger forced alignment (e.g. WhisperX)
- Phrase-level cuts / EDL export
- Larger labeled set to calibrate score weights
- Persist job queue beyond a single process
