import { useEffect, useMemo, useState } from "react";
import { AlertCircle, Braces, CircleCheck, CircleX, Download, LoaderCircle, X } from "lucide-react";
import { message, workbench } from "../api";
import type { Run, StoredDocument } from "../types";
import { downloadJson } from "../../recipe";
import ValueHighlighter, { confidenceLabel } from "./ValueHighlighter";
import RecordView from "./RecordView";

const statusIcon = (status: string) => status === "done" ? <CircleCheck size={13} className="ok" />
  : status === "error" ? <CircleX size={13} className="bad" />
    : status === "running" ? <LoaderCircle size={13} className="spin" /> : <span className="dot" />;

/** One run: per-document status, highlighted source text and the extracted record. */
export default function RunViewer({ runId, onClose }: { runId: string; onClose: () => void }) {
  const [run, setRun] = useState<Run | null>(null);
  const [documentId, setDocumentId] = useState<string | null>(null);
  const [source, setSource] = useState<StoredDocument | null>(null);
  const [selected, setSelected] = useState<string | null>(null);
  const [raw, setRaw] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let active = true;
    let timer: ReturnType<typeof setTimeout>;
    const load = async () => {
      try {
        const next = await workbench.run(runId);
        if (!active) return;
        setRun(next);
        setDocumentId(current => current ?? next.documents[0]?.document_id ?? null);
        if (next.active || next.status === "queued" || next.status === "running") timer = setTimeout(load, 2000);
      } catch (failure) { if (active) setError(message(failure)); }
    };
    void load();
    return () => { active = false; clearTimeout(timer); };
  }, [runId]);

  useEffect(() => {
    if (!documentId) return;
    let active = true;
    setSelected(null);
    workbench.document(documentId).then(next => { if (active) setSource(next); })
      .catch(failure => { if (active) { setSource(null); setError(message(failure)); } });
    return () => { active = false; };
  }, [documentId]);

  const result = run?.documents.find(item => item.document_id === documentId) ?? null;
  const grounded = result ? result.values.filter(value => value.grounding !== "none").length : 0;
  const confidences = useMemo(() => result?.values.map(value => value.confidence).filter((value): value is number => value !== null) ?? [], [result]);
  const stale = source && result && source.sha256 !== result.document_sha256;

  if (error && !run) return <div className="notice error"><AlertCircle size={16} /><span>{error}</span></div>;
  if (!run) return <p className="microcopy"><LoaderCircle size={13} className="spin" /> Loading run…</p>;
  const summary = run.summary;

  return <section className="run-viewer panel" aria-label={`Run ${run.label}`}>
    <div className="run-viewer-header">
      <div><span className="eyebrow">{run.solution_name.toUpperCase()} RUN</span><h2>{run.label}</h2>
        <p className="microcopy">{run.model} · schema {run.extraction_schema.name} v{run.extraction_schema.version} ·
          {" "}{summary.documents_done}/{summary.documents_total} documents{summary.documents_failed ? ` · ${summary.documents_failed} failed` : ""}
          {" "}· {summary.elapsed_seconds}s · {(summary.usage.prompt_tokens + summary.usage.completion_tokens).toLocaleString()} tokens</p></div>
      <span className={`run-status ${run.status}`}>{run.status}</span>
      <a className="secondary" href={`#/compare?runs=${run.id}`}>Compare</a>
      <button className="icon-button" aria-label="Export run JSON" title="Export run JSON"
        onClick={() => downloadJson(run, `${run.label.replace(/[^\w.-]+/g, "_")}.run.json`)}><Download size={14} /></button>
      <button className="icon-button" aria-label="Close run" onClick={onClose}><X size={14} /></button>
    </div>
    {run.error && <div className="notice error"><AlertCircle size={16} /><span>{run.error}</span></div>}
    <div className="run-viewer-body">
      <ul className="run-docs" aria-label="Documents in this run">
        {run.documents.map(item => <li key={item.document_id}>
          <button className={item.document_id === documentId ? "active" : ""} onClick={() => setDocumentId(item.document_id)}>
            {statusIcon(item.status)}<span><strong>{item.document_name}</strong>
              <small>{item.status === "done" ? `${item.values.length} values · ${item.elapsed_seconds ?? "?"}s` : item.status}</small></span>
          </button></li>)}
      </ul>
      <div className="run-source">
        {result?.status === "error" && <div className="notice error"><AlertCircle size={16} /><span>{result.error}</span></div>}
        {stale && <div className="notice warning"><span>The stored document text changed after this run; highlights may be misplaced.</span></div>}
        {source ? <ValueHighlighter text={source.text} values={result?.status === "done" ? result.values : []} selected={selected}
          onSelect={setSelected} pageStarts={source.page_starts} />
          : <p className="microcopy">Loading document…</p>}
      </div>
      <div className="run-record">
        {result && result.status === "done" && <>
          <div className="record-stats">
            <span><b>{result.values.length}</b> values</span>
            <span><b>{result.values.length ? Math.round((grounded / result.values.length) * 100) : 0}%</b> located in text</span>
            <span><b>{confidences.length ? confidenceLabel(confidences.reduce((a, b) => a + b, 0) / confidences.length) : "n/a"}</b> mean confidence</span>
            <div className="segmented"><button className={!raw ? "active" : ""} onClick={() => setRaw(false)}>Record</button>
              <button className={raw ? "active" : ""} onClick={() => setRaw(true)}><Braces size={12} />Raw</button></div>
          </div>
          {result.notes.length > 0 && <ul className="run-notes">{result.notes.map(note => <li key={note}>{note}</li>)}</ul>}
          {raw ? <pre className="results-json">{JSON.stringify({ data: result.data, raw: result.raw }, null, 2)}</pre>
            : <RecordView fields={run.extraction_schema.fields} data={result.data} values={result.values}
              selected={selected} onSelect={setSelected} />}
        </>}
        {result && (result.status === "queued" || result.status === "running") &&
          <p className="microcopy"><LoaderCircle size={13} className="spin" /> {result.status === "running" ? "Extracting…" : "Waiting in queue…"}</p>}
        <p className="microcopy confidence-method">Confidence: {run.confidence_method}</p>
      </div>
    </div>
  </section>;
}
