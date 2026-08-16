import { useEffect, useMemo, useRef, useState } from "react";

const STEPS_FALLBACK = [
  { id: "extract_audio", label: "Extracting the video's audio" },
  { id: "stt", label: "Running speech-to-text" },
  { id: "metadata", label: "Loading metadata timestamps" },
  { id: "compare", label: "Comparing the two timestamp sources" },
  { id: "conflicts", label: "Detecting conflicts greater than 0.5 seconds" },
  { id: "reconcile", label: "Applying the reconciliation decision engine" },
  { id: "cut", label: "Cutting the video with FFmpeg" },
];

function formatSeconds(value) {
  if (value === null || value === undefined || Number.isNaN(value)) return "—";
  return `${Number(value).toFixed(2)}s`;
}

function decisionClass(decision) {
  const key = String(decision || "").toLowerCase();
  if (key === "stt") return "tag stt";
  if (key === "metadata") return "tag meta";
  return "tag other";
}

export default function App() {
  const [videoFile, setVideoFile] = useState(null);
  const [transcriptFile, setTranscriptFile] = useState(null);
  const [metadataFile, setMetadataFile] = useState(null);
  const [transcript, setTranscript] = useState("");
  const [whisperModel, setWhisperModel] = useState("tiny.en");
  const [vadFilter, setVadFilter] = useState(false);
  const [steps, setSteps] = useState(STEPS_FALLBACK);
  const [models, setModels] = useState(["tiny.en", "base.en", "small.en"]);
  const [job, setJob] = useState(null);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const [selectedNumber, setSelectedNumber] = useState(1);
  const [whisperReady, setWhisperReady] = useState(null);
  const pollRef = useRef(null);

  useEffect(() => {
    fetch("/api/health")
      .then((res) => res.json())
      .then((data) => {
        if (data.steps?.length) setSteps(data.steps);
        if (data.models?.length) setModels(data.models);
        setWhisperReady(Boolean(data.whisper));
      })
      .catch(() => {});
    return () => {
      if (pollRef.current) window.clearInterval(pollRef.current);
    };
  }, []);

  const result = job?.result || null;
  const selected = useMemo(() => {
    if (!result?.conflicts?.length) return null;
    return (
      result.conflicts.find((row) => row.number === selectedNumber) ||
      result.conflicts[0]
    );
  }, [result, selectedNumber]);

  async function loadSampleTranscript() {
    setError("");
    const res = await fetch("/api/sample/transcript");
    if (!res.ok) {
      setError("Could not load the sample transcript.");
      return;
    }
    setTranscript(await res.text());
  }

  async function analyze(event) {
    event.preventDefault();
    setError("");
    if (!videoFile) {
      setError("Upload a video file first.");
      return;
    }
    if (!transcript.trim() && !transcriptFile) {
      setError("Paste or upload a transcript.");
      return;
    }

    const body = new FormData();
    body.append("video", videoFile);
    body.append("transcript_text", transcript);
    body.append("whisper_model", whisperModel);
    body.append("vad_filter", vadFilter ? "true" : "false");
    if (transcriptFile) body.append("transcript_file", transcriptFile);
    if (metadataFile) body.append("metadata_file", metadataFile);

    setBusy(true);
    setJob({
      status: "queued",
      step: "queued",
      step_label: "Uploading…",
      completed_steps: [],
      steps,
      result: null,
    });

    try {
      const created = await fetch("/api/jobs", { method: "POST", body });
      const payload = await created.json();
      if (!created.ok) {
        throw new Error(payload.detail || "Upload failed");
      }
      await pollJob(payload.id);
    } catch (err) {
      setBusy(false);
      setError(err.message || String(err));
    }
  }

  async function pollJob(id) {
    if (pollRef.current) window.clearInterval(pollRef.current);

    const tick = async () => {
      const res = await fetch(`/api/jobs/${id}`);
      const data = await res.json();
      if (!res.ok) {
        window.clearInterval(pollRef.current);
        setBusy(false);
        setError(data.detail || "Job not found");
        return;
      }
      setJob(data);
      if (data.status === "complete") {
        window.clearInterval(pollRef.current);
        setBusy(false);
        setSelectedNumber(data.result?.conflicts?.[0]?.number || 1);
      } else if (data.status === "error") {
        window.clearInterval(pollRef.current);
        setBusy(false);
        setError(data.error || "Processing failed");
      }
    };

    await tick();
    pollRef.current = window.setInterval(tick, 500);
  }

  const processing = busy || job?.status === "running" || job?.status === "queued";

  return (
    <div className="page">
      <header className="hero">
        <p className="eyebrow">AI Engineering assessment</p>
        <h1>Timestamp Reconciliation Agent</h1>
        <p className="lede">
          Upload a video and transcript. The backend extracts audio, runs STT,
          compares metadata timestamps, scores every conflict, and cuts a clip
          from the final times.
        </p>
      </header>

      <form className="panel upload" onSubmit={analyze}>
        <div className="grid-2">
          <FileDrop
            label="Video"
            accept="video/mp4,video/quicktime,video/*"
            file={videoFile}
            onFile={setVideoFile}
            hint="MP4 or MOV"
          />
          <FileDrop
            label="Transcript file (optional)"
            accept=".txt,text/plain"
            file={transcriptFile}
            onFile={setTranscriptFile}
            hint="Or paste below"
          />
        </div>
        <label className="field">
          <span>Transcript</span>
          <textarea
            rows={4}
            value={transcript}
            onChange={(e) => setTranscript(e.target.value)}
            placeholder="Paste the spoken words in order…"
          />
        </label>
        <div className="row">
          <button type="button" className="linkish" onClick={loadSampleTranscript}>
            Load sample transcript
          </button>
          <label className="meta-upload">
            Metadata JSON (recommended)
            <input
              type="file"
              accept="application/json,.json"
              onChange={(e) => setMetadataFile(e.target.files?.[0] || null)}
            />
            <span>{metadataFile ? metadataFile.name : "Naive ingest if omitted"}</span>
          </label>
        </div>
        <div className="row">
          <label className="meta-upload">
            Whisper model
            <select
              value={whisperModel}
              onChange={(e) => setWhisperModel(e.target.value)}
            >
              {models.map((name) => (
                <option key={name} value={name}>
                  {name}
                </option>
              ))}
            </select>
          </label>
          <label className="meta-upload">
            <input
              type="checkbox"
              checked={vadFilter}
              onChange={(e) => setVadFilter(e.target.checked)}
            />
            Silence filter (VAD)
          </label>
        </div>
        <button className="primary" type="submit" disabled={processing}>
          {processing ? "Working…" : "Analyze & Reconcile"}
        </button>
        {whisperReady === false ? (
          <p className="error">
            Live speech-to-text is not loaded in the backend. Stop uvicorn, run
            `pip install -r requirements.txt`, then start the API again.
          </p>
        ) : null}
        {error ? <p className="error">{error}</p> : null}
      </form>

      {job ? (
        <section className="panel">
          <h2>Processing</h2>
          {job.step_label ? <p className="notes">{job.step_label}</p> : null}
          <ol className="steps">
            {(job.steps || steps).map((step) => {
              const done = (job.completed_steps || []).includes(step.id);
              const current = job.step === step.id;
              return (
                <li
                  key={step.id}
                  className={done ? "done" : current ? "current" : ""}
                >
                  <span className="dot" />
                  {step.label}
                </li>
              );
            })}
          </ol>
        </section>
      ) : null}

      {result ? (
        <Report
          result={result}
          jobId={job.id}
          selected={selected}
          onSelect={setSelectedNumber}
        />
      ) : null}
    </div>
  );
}

function FileDrop({ label, accept, file, onFile, hint }) {
  return (
    <label className="drop">
      <span className="drop-label">{label}</span>
      <strong>{file ? file.name : "Choose file"}</strong>
      <em>{hint}</em>
      <input
        type="file"
        accept={accept}
        onChange={(e) => onFile(e.target.files?.[0] || null)}
      />
    </label>
  );
}

function Report({ result, jobId, selected, onSelect }) {
  const clipUrl = `/api/jobs/${jobId}/clip`;
  const videoRef = useRef(null);

  function selectRow(number) {
    onSelect(number);
    const row = result.conflicts.find((item) => item.number === number);
    if (!row || !videoRef.current) return;
    const offset = Math.max(0, row.final - (result.clip?.start || 0));
    videoRef.current.currentTime = offset;
  }

  return (
    <section className="report">
      <div className="panel">
        <div className="report-head">
          <div>
            <p className="eyebrow">Report</p>
            <h2>Reconciliation complete</h2>
          </div>
          <span className="status-pill">
            STT: {result.stt_backend}
            {result.whisper_model ? ` (${result.whisper_model})` : ""}
          </span>
        </div>
        {result.metadata_generated ? (
          <p className="banner">
            Metadata was generated, not loaded. No sidecar or chapters were
            found, so the backend used naive even-spacing ingest markers across
            the media duration (independent of STT).
          </p>
        ) : null}
        <div className="stats">
          <Stat label="Conflicts detected" value={result.conflicts_detected} />
          <Stat label="Conflicts resolved" value={result.conflicts_resolved} />
          <Stat label="Trusted most often" value={result.trusted_source} />
          <Stat label="Words aligned" value={result.words} />
        </div>
        <p className="notes">{result.stt_note}</p>
        <p className="notes">{result.metadata_note}</p>
      </div>

      <div className="panel">
        <div className="report-head">
          <h2>Output clip</h2>
          <a className="primary small" href={`${clipUrl}?download=true`}>
            Download clipped video
          </a>
        </div>
        <video ref={videoRef} className="player" src={clipUrl} controls playsInline />
        <p className="muted">
          Window {formatSeconds(result.clip.start)} → {formatSeconds(result.clip.end)}
          {result.clip.actual_duration != null
            ? ` · duration ${formatSeconds(result.clip.actual_duration)}`
            : ""}
        </p>
      </div>

      <div className="panel">
        <h2>Timestamp timeline</h2>
        <p className="muted legend">
          <span className="swatch meta" /> Metadata
          <span className="swatch stt" /> STT
          <span className="swatch final" /> Final
        </p>
        <Timeline rows={result.timeline} />
      </div>

      <div className="split">
        <div className="panel grow">
          <h2>Timestamp decisions</h2>
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>Word</th>
                  <th>Metadata</th>
                  <th>STT</th>
                  <th>Difference</th>
                  <th>Confidence</th>
                  <th>Decision</th>
                  <th>Reason</th>
                </tr>
              </thead>
              <tbody>
                {result.conflicts.map((row) => (
                  <tr
                    key={row.number}
                    className={selected?.number === row.number ? "active" : ""}
                    onClick={() => selectRow(row.number)}
                  >
                    <td>
                      {row.word}
                      {row.interpolated ? <em className="badge">interpolated</em> : null}
                      <div className="muted">final {formatSeconds(row.final)}</div>
                    </td>
                    <td>{formatSeconds(row.metadata)}</td>
                    <td>{formatSeconds(row.stt)}</td>
                    <td>{formatSeconds(row.difference)}</td>
                    <td>{row.confidence == null ? "—" : row.confidence.toFixed(2)}</td>
                    <td>
                      <span className={decisionClass(row.decision)}>{row.decision}</span>
                    </td>
                    <td className="reason">{row.reason}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
        <aside className="panel details">
          <h2>Decision details</h2>
          {selected ? (
            <DecisionDetails conflict={selected} />
          ) : (
            <p className="muted">No conflicts above 0.5 seconds.</p>
          )}
        </aside>
      </div>
    </section>
  );
}

function Stat({ label, value }) {
  return (
    <div className="stat">
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function Timeline({ rows }) {
  if (!rows?.length) return null;
  const times = rows.flatMap((row) => [row.metadata, row.stt, row.final]);
  const min = Math.min(...times);
  const max = Math.max(...times);
  const span = Math.max(max - min, 0.01);
  const pct = (value) => ((value - min) / span) * 100;

  return (
    <div className="timeline">
      {rows.map((row) => (
        <div key={row.index} className={row.conflict ? "tl-row conflict" : "tl-row"}>
          <div className="tl-word">
            {row.word}
            {row.conflict ? <em>conflict</em> : null}
            {row.interpolated ? <em>interpolated</em> : null}
          </div>
          <div className="tl-track">
            <i className="mark meta" style={{ left: `${pct(row.metadata)}%` }} title={`Metadata ${row.metadata}s`} />
            <i className="mark stt" style={{ left: `${pct(row.stt)}%` }} title={`STT ${row.stt}s`} />
            <i className="mark final" style={{ left: `${pct(row.final)}%` }} title={`Final ${row.final}s`} />
          </div>
        </div>
      ))}
    </div>
  );
}

function DecisionDetails({ conflict }) {
  const factors = conflict.factors || {};
  return (
    <div>
      <p>
        <strong>{conflict.word}</strong> · conflict #{conflict.number}
        {conflict.interpolated ? <em className="badge">interpolated STT</em> : null}
      </p>
      <MiniCompare conflict={conflict} />
      <dl className="facts">
        <div><dt>Metadata</dt><dd>{formatSeconds(conflict.metadata)}</dd></div>
        <div><dt>STT</dt><dd>{formatSeconds(conflict.stt)}</dd></div>
        <div><dt>Difference</dt><dd>{formatSeconds(conflict.difference)}</dd></div>
        <div><dt>Confidence</dt><dd>{conflict.confidence == null ? "—" : conflict.confidence.toFixed(2)}</dd></div>
        <div><dt>Winner</dt><dd>{conflict.decision}</dd></div>
        <div><dt>Final timestamp</dt><dd>{formatSeconds(conflict.final)}</dd></div>
      </dl>
      <p className="reason-block">{conflict.reason}</p>
      <h3>Factors scored</h3>
      <ul className="factors">
        {Object.entries(factors).map(([key, value]) => (
          <li key={key}>
            <span>{key.replaceAll("_", " ")}</span>
            <strong>{String(value)}</strong>
          </li>
        ))}
      </ul>
    </div>
  );
}

function MiniCompare({ conflict }) {
  const times = [conflict.metadata, conflict.stt, conflict.final];
  const min = Math.min(...times);
  const max = Math.max(...times);
  const span = Math.max(max - min, 0.01);
  const pct = (value) => ((value - min) / span) * 100;
  return (
    <div className="mini-track">
      <i className="mark meta" style={{ left: `${pct(conflict.metadata)}%` }} />
      <i className="mark stt" style={{ left: `${pct(conflict.stt)}%` }} />
      <i className="mark final" style={{ left: `${pct(conflict.final)}%` }} />
    </div>
  );
}
