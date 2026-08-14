# Timestamp Reconciliation Agent

Python CLI that takes a video and a transcript, reads timestamps from **two independent sources**, detects conflicts, decides which timestamp to trust, and uses the final times to **cut a real clip with FFmpeg**.

This is built as a working internship assessment: no UI, no fake decision engine, and a demo mode that does not require downloading a speech model.

## What it does

1. Load a transcript and a video.
2. Obtain per-word times from **stale editor/ingest metadata**.
3. Obtain per-word times from **speech-to-text** (Whisper-shaped).
4. Mark a **conflict** when the two starts differ by **more than 0.5 seconds**.
5. Score both sources for every conflict (confidence, neighbors, order, drift).
6. Log the decision and the reason.
7. Cut `output/clip_*.mp4` using the reconciled first/last word times.
8. Write a timestamped HTML report for a live demo.

The sample data contains **four** conflicts above 0.5s. The engine does **not** always pick the same source: high-confidence STT wins some rows, low-confidence STT loses to consistent metadata.

## Architecture

```
video + transcript
        │
        ├─ metadata.py      stale editor markers (sidecar JSON, chapters if present)
        ├─ transcription.py demo fixture or local faster-whisper
        │
        ▼
   aligned word pairs
        │
        ▼
   reconciler.py            conflict detection + scoring + monotonic repair
        │
        ├─ stdout conflict logs
        ├─ cutter.py        FFmpeg clip from final start → final end
        └─ report.py        timestamped HTML report
```

| Module | Responsibility |
|---|---|
| `src/main.py` | CLI wiring |
| `src/metadata.py` | Metadata timestamp source |
| `src/transcription.py` | STT timestamp source |
| `src/reconciler.py` | Alignment, conflicts, scores, final times |
| `src/cutter.py` | FFmpeg cutting |
| `src/report.py` | HTML report |
| `sample/` | Demo video, transcript, both timestamp fixtures |
| `tests/` | pytest coverage of the decision engine |

## How the two timestamp sources work

### 1. Metadata (approximate / stale)

Normal MP4s do not store per-word timestamps. In a real pipeline those times usually come from an editor timeline, caption ingest, or a previous alignment pass, and they go stale when audio is shifted.

`metadata.py` first asks `ffprobe` for chapter markers. If the container has none, it loads `sample/metadata_timestamps.json`.

That JSON is a **stub of the source**, not a stub of the algorithm. The reconciler still compares real floats, detects real conflicts, and makes real decisions.

In this demo the sidecar represents markers copied from an older timeline: most words are close, a few markers are wrong by 0.7–0.85s.

### 2. Speech-to-text

Default `--stt-backend demo` loads `sample/stt_timestamps.json`. The schema matches faster-whisper word timings (`start`, `end`, `confidence` as probability).

`--stt-backend whisper` runs **faster-whisper** locally (`tiny.en` by default), uses word timestamps, and stores each word’s `probability` as confidence. Words are then greedily aligned onto the transcript.

Demo mode exists so a 3-minute walkthrough does not depend on a model download.

## Reconciliation algorithm

A conflict is strictly:

```text
|stt_start - metadata_start| > 0.5
```

Non-conflicts still produce a final time: high-confidence STT is used as-is; otherwise the two starts are averaged.

For **each conflict**, both sources get a numeric score.

**STT score**

- `confidence * 40`
- neighbor consistency `* 25` (does this time sit between its neighbors without a huge jump?)
- `+15` if the STT sequence stays chronological
- `+10` if the gap disappears after removing a global offset (drift)
- small penalty if the raw gap is huge (`> 2s`)

**Metadata score**

- `(1 - confidence) * 35` (low STT confidence makes metadata more attractive)
- neighbor consistency `* 25`
- `+15` if metadata stays chronological
- `+12` if drift-corrected metadata agrees with STT

Winner = higher score. Ties go to STT.

After the winner is chosen, if that time would go **backwards** versus the previous *final* word, the engine rejects it and uses the other source. If both are illegal, it applies a `+0.05s` ordering repair.

### Drift

`estimate_offset` takes the median of `(STT - metadata)` on high-confidence words, plus the median absolute deviation.

If `|offset| ≥ 0.15s` and `MAD ≤ 0.25s`, a **systematic offset** is flagged. That does not hide the conflict (raw diff still counts), but it is a scoring factor: a stable 0.6s lag looks like a shifted timeline, not a random error, so high-confidence STT is preferred as audio ground truth.

### Why the rule is designed this way

Speech-to-text is usually the better clock for *when words were actually spoken*, but it is not trustworthy when confidence is low or when a timestamp jumps backwards.

Metadata is usually the better clock when an editor timeline is internally consistent and the recognizer is guessing.

A single hardcoded “always STT” or “always metadata” rule would fail one of those cases. Scoring both sources, then enforcing order, keeps the policy deterministic and explainable in the log.

On the sample file that produces mixed decisions:

| Word | Metadata | STT | Diff | Conf | Decision |
|---|---|---|---|---|---|
| timestamp | 2.30s | 1.55s | 0.75s | 0.96 | STT |
| system | 4.55s | 3.85s | 0.70s | 0.30 | metadata |
| conflicts | 6.25s | 5.40s | 0.85s | 0.94 | STT |
| reliably | 7.15s | 6.30s | 0.85s | 0.95 | STT |

The last decision changes the **clip end**, so the output video is actually tied to reconciliation, not to a hardcoded `ffmpeg -t`.

## Installation

Requires Python 3.10+ and FFmpeg.

```bash
# macOS
brew install ffmpeg

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Optional live Whisper:

```bash
pip install -r requirements-whisper.txt
```

Generate (or regenerate) the sample video:

```bash
python3 scripts/generate_sample.py
```

## How to run

From the project root:

```bash
python3 -m src.main --video sample/input.mp4 --transcript sample/transcript.txt
```

That command uses demo STT + the metadata sidecar. Outputs:

- `output/clip_<UTC timestamp>.mp4`
- `output/report_<UTC timestamp>.html`

Live Whisper (downloads `tiny.en` on first run):

```bash
python3 -m src.main \
  --video sample/input.mp4 \
  --transcript sample/transcript.txt \
  --stt-backend whisper \
  --whisper-model tiny.en
```

Tests:

```bash
python3 -m pytest -q
```

## Example output

```text
Words aligned: 10
Conflicts > 0.5s: 4

Conflict #1
Word: "timestamp"
Metadata: 2.30s
STT: 1.55s
Difference: 0.75s
STT confidence: 0.96
Decision: STT
Reason: High STT confidence and neighboring STT timestamps are consistent.
Final timestamp: 1.55s
```

Open the HTML report for the full table, both sources, factors, and output clip path.

## Known limitations

- Demo STT is a fixture. It is Whisper-*shaped* (word times + probability). It is not a live model run unless you pass `--stt-backend whisper`.
- Metadata is a documented sidecar. Chapter extraction is implemented, but typical MP4s have no per-word chapters.
- Alignment assumes the transcript order is the source of truth. Whisper extra words are skipped; missing transcript words fail.
- The clip is one window from the first reconciled word to the last, not a per-word montage.
- FFmpeg re-encodes (`libx264` / `aac`) so the cut is frame-accurate enough for a demo, not a lossless stream copy.

## What I would improve with more time

- Force-align the transcript with Whisper or wav2vec so STT cannot drift off the expected words.
- Estimate linear drift (`offset + slope * t`), not only a global offset.
- Cut around a chosen phrase, or emit an Edit Decision List, instead of one window.
- Calibrate score weights on labeled conflicts.
- Add speaker-change and silence features as extra trust signals.
- Stream logs as JSONL for later evaluation.
