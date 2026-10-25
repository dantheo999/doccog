import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { ReactNode } from "react";
import {
  Ban, ChevronDown, ChevronRight, CircleAlert, Clock, Download, FileJson, GitCompareArrows, LoaderCircle,
  RefreshCw, TriangleAlert, X,
} from "lucide-react";
import { message, workbench } from "../api";
import { href, navigate, useRoute } from "../router";
import ValueHighlighter, { confidenceLabel, confidenceTone, formatValue, pathKey } from "../components/ValueHighlighter";
import type {
  CompareCell, CompareCellValue, CompareRow, CompareRunStats, Comparison, FieldValue, Grounding, Run, RunStatus,
  RunSummary, SchemaField, StoredDocument,
} from "../types";
import "../compare.css";

const ROUTE = "/compare";
const MAX_RUNS = 12; // POST /api/compare accepts at most 12 runs

/* ------------------------------------------------------------------ helpers */

const pct = (value: number | null | undefined) =>
  value === null || value === undefined ? "—" : `${Math.round(value * 100)}%`;
const compact = (n: number) => n >= 1e6 ? `${(n / 1e6).toFixed(1)}M` : n >= 1e4 ? `${Math.round(n / 1e3)}k`
  : n >= 1e3 ? `${(n / 1e3).toFixed(1)}k` : String(n);
const seconds = (s: number) => `${s < 10 ? s.toFixed(1) : Math.round(s)} s`;
const average = (xs: number[]) => xs.length ? xs.reduce((a, b) => a + b, 0) / xs.length : null;
const shortDate = (iso: string) => {
  const date = new Date(iso);
  return Number.isNaN(date.getTime()) ? iso
    : date.toLocaleString(undefined, { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" });
};
const isEmpty = (value: unknown): boolean => value === null || value === undefined || value === ""
  || (Array.isArray(value) && value.length === 0)
  || (typeof value === "object" && !Array.isArray(value) && Object.values(value as object).every(isEmpty));
const runName = (run: RunSummary) => run.label ? `${run.solution_name} — ${run.label}` : run.solution_name;
const fieldType = (field: SchemaField) => `${field.type}${field.array ? " list" : ""}`;
const cellKey = (documentId: string, field: string) => `${documentId}\u0001${field}`;
const groundingClass = (g: Grounding | null) => g === "exact" ? "exact" : g === "fuzzy" ? "approximate" : "ungrounded";
const groundingText = (g: Grounding | null) => g === "none" ? "not found" : g ?? "";

/** One agreement scale (red → amber → teal) shared by every matrix; always shown next to a % and a word. */
function tone(score: number | null | undefined, match: number) {
  if (score === null || score === undefined) return { cls: "cmp-none", word: "not scored" };
  if (score >= Math.max(match, 0.95)) return { cls: "cmp-s4", word: "strong agreement" };
  if (score >= match) return { cls: "cmp-s3", word: "agrees" };
  if (score >= 0.5) return { cls: "cmp-s2", word: "partial" };
  if (score >= 0.25) return { cls: "cmp-s1", word: "weak" };
  return { cls: "cmp-s0", word: "differs" };
}

function runState(run: RunSummary): { label: string; cls: string } {
  if (run.status === "done" && run.documents_failed) return { label: `done · ${run.documents_failed} failed`, cls: "warn" };
  if ((run.status === "error" || run.status === "cancelled") && run.documents_done) return { label: `${run.status} · partial`, cls: "warn" };
  return { label: run.status, cls: run.status === "done" ? "ok" : run.status === "error" ? "bad" : run.status === "running" || run.status === "queued" ? "busy" : "" };
}

/** Compact one-line summary of a list item: object items join their scalar child values ("Jane · Chair · present"). */
function summarize(item: unknown, field: SchemaField): string {
  if (field.type !== "object" || !item || typeof item !== "object") return formatValue(item);
  const record = item as Record<string, unknown>;
  return field.fields.flatMap(child => {
    const value = record[child.name];
    if (isEmpty(value)) return [];
    if (Array.isArray(value)) return [`${value.length} ${child.name}`];
    if (child.type === "object") return [summarize(value, child)];
    return [formatValue(value)];
  }).join(" · ") || "—";
}

function describe(item: unknown, field: SchemaField): string {
  if (field.type !== "object" || !item || typeof item !== "object") return formatValue(item);
  const record = item as Record<string, unknown>;
  return field.fields.map(child => `${child.name}: ${isEmpty(record[child.name]) ? "—" : formatValue(record[child.name])}`).join("\n");
}

/** The run other runs' items are aligned to. The server reports it; older responses fall back to its rule:
 * the reference if present, else the run with the most items. */
function pivotOf(cell: CompareCell, runs: CompareRunStats[], reference: string | null): string | null {
  if (cell.pivot) return cell.pivot;
  if (reference && cell.values[reference]) return reference;
  let best: string | null = null;
  let most = -1;
  for (const run of runs) {
    const value = cell.values[run.id]?.value;
    if (!cell.values[run.id]) continue;
    const count = Array.isArray(value) ? value.length : 0;
    if (count > most) { most = count; best = run.id; }
  }
  return best;
}

function documentScore(comparison: Comparison, cells: Map<string, CompareCell>, documentId: string, runId: string) {
  const scores: number[] = [];
  let filled = 0;
  for (const field of comparison.fields) {
    const cell = cells.get(cellKey(documentId, field.name));
    const score = cell?.agreement[runId];
    if (score !== null && score !== undefined) scores.push(score);
    if (cell?.values[runId] && !isEmpty(cell.values[runId].value)) filled += 1;
  }
  return { agreement: average(scores), filled, total: comparison.fields.length };
}

const csvCell = (value: unknown) => {
  const text = value === null || value === undefined ? "" : String(value);
  return /[",\r\n]/.test(text) ? `"${text.replace(/"/g, '""')}"` : text;
};
const toCsv = (rows: unknown[][]) => rows.map(row => row.map(csvCell).join(",")).join("\r\n");

function download(filename: string, type: string, content: string) {
  if (typeof URL.createObjectURL !== "function") return;
  const url = URL.createObjectURL(new Blob([content], { type }));
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  document.body.append(link);
  link.click();
  link.remove();
  setTimeout(() => URL.revokeObjectURL(url), 0);
}

/** Column names for exports; disambiguates runs that share solution and label. */
function exportNames(runs: RunSummary[]): Map<string, string> {
  const counts = new Map<string, number>();
  runs.forEach(run => counts.set(runName(run), (counts.get(runName(run)) ?? 0) + 1));
  return new Map(runs.map(run => [run.id, (counts.get(runName(run)) ?? 0) > 1 ? `${runName(run)} (${run.id.slice(0, 6)})` : runName(run)]));
}

/* ------------------------------------------------------------------ page */

interface SourceTarget { runId: string; documentId: string; field: string; index?: number; scalar: boolean }

export default function ComparePage() {
  const { query } = useRoute();
  const runParam = query.runs ?? "";
  const runIds = useMemo(() => [...new Set(runParam.split(",").map(id => id.trim()).filter(Boolean))], [runParam]);
  const reference = query.ref && runIds.includes(query.ref) ? query.ref : null;
  const documentId = query.doc ?? null;

  const [runs, setRuns] = useState<RunSummary[] | null>(null);
  const [runsError, setRunsError] = useState<string | null>(null);
  const [comparison, setComparison] = useState<Comparison | null>(null);
  const [compareError, setCompareError] = useState<string | null>(null);
  const [comparing, setComparing] = useState(false);
  const [reload, setReload] = useState(0);
  const [source, setSource] = useState<SourceTarget | null>(null);
  const drillRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    let cancelled = false;
    setRunsError(null);
    workbench.runs().then(list => { if (!cancelled) setRuns(list); },
      failure => { if (!cancelled) setRunsError(message(failure)); });
    return () => { cancelled = true; };
  }, [reload]);

  const compareKey = `${runIds.join(",")}|${reference ?? ""}`;
  useEffect(() => {
    if (!runIds.length) { setComparison(null); setCompareError(null); setComparing(false); return; }
    let cancelled = false;
    setComparing(true);
    setCompareError(null);
    workbench.compare(runIds, reference)
      .then(result => { if (!cancelled) setComparison(result); },
        failure => { if (!cancelled) { setComparison(null); setCompareError(message(failure)); } })
      .finally(() => { if (!cancelled) setComparing(false); });
    return () => { cancelled = true; };
  }, [compareKey, reload]); // compareKey captures runIds + reference

  // Cached loaders for the source panel; refreshed together with the comparison.
  const loaders = useMemo(() => {
    const documents = new Map<string, Promise<StoredDocument>>();
    const fullRuns = new Map<string, Promise<Run>>();
    function cached<T>(store: Map<string, Promise<T>>, id: string, fetcher: (id: string) => Promise<T>) {
      let pending = store.get(id);
      if (!pending) {
        pending = fetcher(id);
        store.set(id, pending);
        pending.catch(() => store.delete(id));
      }
      return pending;
    }
    return {
      document: (id: string) => cached(documents, id, workbench.document),
      run: (id: string) => cached(fullRuns, id, workbench.run),
    };
  }, [reload]);

  const setQuery = useCallback((patch: { runs?: string[]; ref?: string | null; doc?: string | null }) => {
    const nextRuns = patch.runs ?? runIds;
    const nextRef = patch.ref !== undefined ? patch.ref : reference;
    const nextDoc = patch.doc !== undefined ? patch.doc : documentId;
    navigate(ROUTE, {
      runs: nextRuns.join(",") || undefined,
      ref: nextRef && nextRuns.includes(nextRef) ? nextRef : undefined,
      doc: nextRuns.length && nextDoc ? nextDoc : undefined,
    });
  }, [runIds, reference, documentId]);

  const toggleRun = (id: string) =>
    setQuery({ runs: runIds.includes(id) ? runIds.filter(other => other !== id) : [...runIds, id].slice(0, MAX_RUNS) });

  const cells = useMemo(() => new Map((comparison?.cells ?? []).map(cell => [cellKey(cell.document_id, cell.field), cell])),
    [comparison]);
  const byId = useMemo(() => new Map((runs ?? []).map(run => [run.id, run])), [runs]);
  const current = comparison && comparison.runs.length ? comparison : null;
  const activeSource = source && source.documentId === documentId && current?.runs.some(run => run.id === source.runId)
    ? source : null;

  const hasComparison = !!current;
  useEffect(() => {
    if (documentId && hasComparison) drillRef.current?.scrollIntoView?.({ behavior: "smooth", block: "start" });
  }, [documentId, hasComparison]);

  function exportJson() {
    if (!current) return;
    download(`comparison-${new Date().toISOString().slice(0, 10)}.json`, "application/json",
      JSON.stringify({ exported_at: new Date().toISOString(), run_ids: runIds, reference, comparison: current }, null, 2));
  }

  return <div className="wb-page cmp-page">
    <header className="page-heading cmp-header">
      <div>
        <span className="eyebrow">COMPARE RUNS</span>
        <h1>How did each solution do?</h1>
        <p className="microcopy">Extracted values side by side for each run. There is no ground truth: agreement is
          measured against a reference run you choose, or against the consensus of the other runs.</p>
      </div>
      <div className="cmp-header-actions">
        <button type="button" className="icon-button" aria-label="Refresh runs and comparison" title="Refresh"
          onClick={() => setReload(value => value + 1)}><RefreshCw size={14} className={comparing ? "spin" : ""} /></button>
        <button type="button" className="secondary" onClick={exportJson} disabled={!current}>
          <FileJson size={14} /> Export JSON</button>
      </div>
    </header>

    <div className="cmp-layout">
      <RunPicker runs={runs} error={runsError} selected={runIds} onToggle={toggleRun}
        onRemove={ids => setQuery({ runs: runIds.filter(id => !ids.includes(id)) })}
        onClear={() => setQuery({ runs: [], ref: null, doc: null })} onRetry={() => setReload(value => value + 1)} />

      <div className="cmp-main" aria-busy={comparing}>
        {!runIds.length ? <EmptyState noRuns={runs !== null && runs.length === 0} />
          : <>
            <div className="cmp-toolbar panel">
              <label className="cmp-ref-select">
                <span className="eyebrow">AGREEMENT MEASURED AGAINST</span>
                <select value={reference ?? ""} onChange={event => setQuery({ ref: event.target.value || null })}>
                  <option value="">Consensus (no reference)</option>
                  {runIds.map(id => <option key={id} value={id}>
                    {byId.get(id) ? runName(byId.get(id)!) : current?.runs.find(run => run.id === id)?.solution_name ?? id}
                  </option>)}
                </select>
              </label>
              <ModeExplainer reference={reference ? current?.runs.find(run => run.id === reference) ?? byId.get(reference) ?? null : null}
                count={runIds.length} match={current?.thresholds.match ?? 0.8} />
            </div>
            {comparing && <p className="cmp-status-line"><LoaderCircle size={13} className="spin" />
              Comparing {runIds.length} run{runIds.length === 1 ? "" : "s"}…</p>}
            {compareError && <div className="notice" role="alert"><CircleAlert size={15} />
              <span>Comparison failed: {compareError}</span>
              <button type="button" onClick={() => setReload(value => value + 1)}>Retry</button></div>}
            {current?.warnings.map(warning => <div className="notice warning" key={warning}>
              <TriangleAlert size={15} /><span>{warning}</span></div>)}
            {current && <div className={`cmp-results ${comparing ? "cmp-stale" : ""}`}>
              <Scoreboard comparison={current} />
              <FieldMatrix comparison={current} />
              <DocumentMatrix comparison={current} cells={cells} selected={documentId}
                onOpen={id => setQuery({ doc: id === documentId ? null : id })} />
              {documentId && <div ref={drillRef} className={`cmp-drill-layout ${activeSource ? "with-source" : ""}`}>
                <Drilldown comparison={current} cells={cells} documentId={documentId} active={activeSource}
                  onClose={() => setQuery({ doc: null })} onSource={setSource} />
                {activeSource && <SourcePanel key={JSON.stringify(activeSource)} target={activeSource} loaders={loaders}
                  comparison={current} onClose={() => setSource(null)} />}
              </div>}
            </div>}
          </>}
      </div>
    </div>
  </div>;
}

/* ------------------------------------------------------------------ run picker */

interface PickerProps {
  runs: RunSummary[] | null;
  error: string | null;
  selected: string[];
  onToggle: (id: string) => void;
  onRemove: (ids: string[]) => void;
  onClear: () => void;
  onRetry: () => void;
}

function RunPicker({ runs, error, selected, onToggle, onRemove, onClear, onRetry }: PickerProps) {
  const [schemaFilter, setSchemaFilter] = useState("all");
  const groups = useMemo(() => {
    const bySchema = new Map<string, { id: string; name: string; runs: RunSummary[] }>();
    for (const run of [...(runs ?? [])].sort((a, b) => b.created_at.localeCompare(a.created_at))) {
      const group = bySchema.get(run.schema_id) ?? { id: run.schema_id, name: run.schema_name || "Untitled schema", runs: [] };
      group.runs.push(run);
      bySchema.set(run.schema_id, group);
    }
    return [...bySchema.values()];
  }, [runs]);
  const chosen = selected.map(id => runs?.find(run => run.id === id)).filter((run): run is RunSummary => !!run);
  const versions = [...new Set(chosen.map(run => `${run.schema_name} v${run.schema_version}`))];
  const mixedSchemas = new Set(chosen.map(run => `${run.schema_id}@${run.schema_version}`)).size > 1;
  const unfinished = chosen.filter(run => run.status !== "done");
  const unknown = runs ? selected.filter(id => !runs.some(run => run.id === id)) : [];
  const visible = groups.filter(group => schemaFilter === "all" || group.id === schemaFilter);

  return <aside className="panel cmp-picker" aria-label="Run picker">
    <div className="cmp-picker-head">
      <div>
        <span className="eyebrow">RUNS</span>
        <h2>Select runs</h2>
      </div>
      <span className="count">{selected.length} selected</span>
      {selected.length > 0 && <button type="button" className="cmp-text-button" onClick={onClear}>Clear</button>}
    </div>
    {groups.length > 1 && <label className="cmp-schema-filter">
      <span>Schema</span>
      <select value={schemaFilter} onChange={event => setSchemaFilter(event.target.value)}>
        <option value="all">All schemas</option>
        {groups.map(group => <option key={group.id} value={group.id}>{group.name}</option>)}
      </select>
    </label>}
    <div className="cmp-run-list">
      {runs === null && !error && <p className="cmp-status-line"><LoaderCircle size={13} className="spin" /> Loading runs…</p>}
      {error && <div className="notice" role="alert"><CircleAlert size={15} /><span>{error}</span>
        <button type="button" onClick={onRetry}>Retry</button></div>}
      {runs?.length === 0 && <p className="microcopy">No runs yet.</p>}
      {visible.map(group => <fieldset key={group.id} className="cmp-run-group">
        <legend>{group.name}
          <small>v{[...new Set(group.runs.map(run => run.schema_version))].sort((a, b) => a - b).join(", v")}</small></legend>
        {group.runs.map(run => {
          const checked = selected.includes(run.id);
          const state = runState(run);
          return <label key={run.id} className={`cmp-run-option ${checked ? "checked" : ""}`}>
            <input type="checkbox" checked={checked} disabled={!checked && selected.length >= MAX_RUNS}
              onChange={() => onToggle(run.id)} />
            <span className="cmp-run-text">
              <strong>{run.solution_name}</strong>
              <span>{[run.label || "Untitled run", run.model].filter(Boolean).join(" · ")}</span>
              <span className="cmp-run-meta">
                <span className={`cmp-state ${state.cls}`}>{state.label}</span>
                <span>{run.documents_done}/{run.documents_total} docs</span>
                <span>v{run.schema_version}</span>
                <span>{shortDate(run.created_at)}</span>
              </span>
            </span>
          </label>;
        })}
      </fieldset>)}
    </div>
    {(mixedSchemas || unfinished.length > 0 || unknown.length > 0 || selected.length >= MAX_RUNS) && <div className="cmp-picker-notes">
      {mixedSchemas && <p className="cmp-note warn"><TriangleAlert size={12} />
        Selected runs use different schemas ({versions.join(", ")}). Fields are matched by name.</p>}
      {unfinished.length > 0 && <p className="cmp-note"><Clock size={12} />
        {unfinished.map(run => `${run.solution_name} (${run.status})`).join(", ")} not finished — only completed
        documents are compared. Refresh for newer results.</p>}
      {unknown.length > 0 && <p className="cmp-note warn"><CircleAlert size={12} />
        {unknown.length} selected run{unknown.length === 1 ? " no longer exists" : "s no longer exist"}.
        <button type="button" className="cmp-text-button" onClick={() => onRemove(unknown)}>Remove</button></p>}
      {selected.length >= MAX_RUNS && <p className="cmp-note">Up to {MAX_RUNS} runs can be compared at once.</p>}
    </div>}
  </aside>;
}

function EmptyState({ noRuns }: { noRuns: boolean }) {
  return <div className="panel cmp-empty empty">
    <div className="empty-illustration"><GitCompareArrows size={30} /></div>
    <h3>{noRuns ? "Nothing to compare yet" : "Pick runs to compare"}</h3>
    <p>{noRuns ? "Run at least one solution, then select runs to compare."
      : "Select two or more runs on the left. Runs of the same schema version compare best; one run alone still shows fill rate, grounding and cost."}</p>
    {noRuns && <a className="secondary" href={href("/")}>Go to solutions</a>}
  </div>;
}

function ModeExplainer({ reference, count, match }: { reference: RunSummary | null; count: number; match: number }) {
  return <div className="cmp-explainer">
    <p className="microcopy">
      {reference ? <>Every run is scored against <b>{runName(reference)}</b> as if it were the answer key. The
        reference itself is not scored. High agreement means "same as the reference", not "correct".</>
        : count > 1 ? <>Each value is compared with the same field in every other selected run; agreement is the mean
          similarity. A run that agrees with the others scores high — that is consensus, not correctness.</>
          : <>Select another run to measure agreement. Fill rate, grounding and cost are shown for a single run.</>}
    </p>
    <details className="cmp-details">
      <summary>How similarity is scored</summary>
      <ul>
        <li>Text is fuzzy-matched after normalising case and whitespace; dates, numbers, enums and booleans must match exactly.</li>
        <li>Lists are aligned item by item (pairs ≥ 50% similar) and scored together; in object items the first text field counts double.</li>
        <li>Cells where both values are empty are skipped. A cell <b>agrees</b> at ≥ {Math.round(match * 100)}% similarity.</li>
      </ul>
    </details>
    <ToneLegend match={match} />
  </div>;
}

function ToneLegend({ match }: { match: number }) {
  const steps: [number, string][] = [[0, "< 25% differs"], [0.3, "25–50% weak"], [0.6, `50–${Math.round(match * 100)}% partial`],
    [match, `${Math.round(match * 100)}–95% agrees`], [1, "≥ 95% strong"]];
  return <ul className="cmp-legend" aria-label="Agreement colour scale">
    {steps.map(([score, label]) => <li key={label}><i className={`cmp-swatch ${tone(score, match).cls}`} />{label}</li>)}
    <li><i className="cmp-swatch cmp-none" />not scored</li>
  </ul>;
}

/* ------------------------------------------------------------------ shared table bits */

function RunHead({ run, reference, extra }: { run: CompareRunStats; reference: string | null; extra?: ReactNode }) {
  return <th scope="col" className={`cmp-run-col ${run.id === reference ? "cmp-is-ref" : ""}`}>
    <span className="cmp-run-name">{run.solution_name}</span>
    <small>{[run.label, run.model].filter(Boolean).join(" · ")}</small>
    {run.id === reference && <span className="cmp-ref-tag">reference</span>}
    {extra}
  </th>;
}

function Score({ score, match, isReference }: { score: number | null | undefined; match: number; isReference?: boolean }) {
  if (isReference) return <span className="cmp-badge cmp-none" title="Reference run (not scored)">ref</span>;
  const t = tone(score, match);
  return <span className={`cmp-badge ${t.cls}`} title={t.word}>{score === null || score === undefined ? "n/a" : pct(score)}</span>;
}

function StatusChip({ status }: { status: RunStatus | undefined }) {
  const icon = status === "error" ? <CircleAlert size={12} /> : status === "cancelled" ? <Ban size={12} />
    : status === "running" || status === "queued" ? <Clock size={12} /> : null;
  return <span className={`cmp-run-status ${status ?? "missing"}`}>{icon}{status ?? "not in run"}</span>;
}

/* ------------------------------------------------------------------ scoreboard */

interface Metric {
  key: string;
  label: string;
  help: string;
  better?: "high" | "low";
  bar?: boolean;
  toned?: boolean;
  value: (run: CompareRunStats) => number | null;
  format?: (value: number) => string;
  detail?: (run: CompareRunStats) => string;
  missing?: (run: CompareRunStats) => string;
}

const processed = (run: RunSummary) => run.documents_done + run.documents_failed;

function metrics(match: number): Metric[] {
  return [
    { key: "agreement", label: "Agreement", better: "high", bar: true, toned: true, value: run => run.agreement,
      help: "Mean similarity of this run's values to the reference (or the other runs) over all document × field cells." },
    { key: "agree_rate", label: "Agree rate", better: "high", bar: true, toned: true, value: run => run.agree_rate,
      help: `Share of scored cells at ≥ ${Math.round(match * 100)}% similarity.` },
    { key: "fill_rate", label: "Fill rate", better: "high", bar: true, value: run => run.fill_rate,
      help: "Share of fields with a value in completed documents. Only good if the values also agree." },
    { key: "grounded", label: "Grounded", better: "high", bar: true, value: run => run.values ? run.grounded / run.values : null,
      detail: run => `${run.grounded}/${run.values} values`, missing: () => "no values",
      help: "Share of extracted leaf values located in the source text (exact or fuzzy)." },
    { key: "confidence", label: "Mean confidence", value: run => run.mean_confidence,
      missing: run => `n/a — ${run.confidence_method || "not reported"}`, detail: run => run.confidence_method,
      help: "Self-reported; each solution uses its own method, so don't rank across solutions." },
    { key: "speed", label: "Time / document", better: "low",
      value: run => processed(run) && run.elapsed_seconds > 0 ? run.elapsed_seconds / processed(run) : null,
      format: seconds, detail: run => `${seconds(run.elapsed_seconds)} total`, missing: () => "not timed",
      help: "Processing time per finished or failed document. Imported runs are not timed." },
    { key: "tokens", label: "Tokens", better: "low",
      value: run => run.usage.calls ? run.usage.prompt_tokens + run.usage.completion_tokens : null, format: compact,
      detail: run => `${run.usage.calls} call${run.usage.calls === 1 ? "" : "s"} · ${compact(run.usage.prompt_tokens)} in / ${compact(run.usage.completion_tokens)} out`,
      missing: () => "not metered", help: "Prompt + completion tokens over all model calls. Local models may not report usage." },
    { key: "failed", label: "Failed documents", better: "low", value: run => run.documents_failed, format: String,
      detail: run => `${run.documents_done}/${run.documents_total} done`, help: "Documents that ended in an error." },
  ];
}

function bestRuns(metric: Metric, runs: CompareRunStats[]): Set<string> {
  if (!metric.better) return new Set();
  const scored = runs.map(run => [run.id, metric.value(run)] as const)
    .filter((entry): entry is readonly [string, number] => entry[1] !== null && Number.isFinite(entry[1]));
  const values = scored.map(([, value]) => value);
  if (scored.length < 2 || Math.max(...values) === Math.min(...values)) return new Set();
  const best = metric.better === "high" ? Math.max(...values) : Math.min(...values);
  return new Set(scored.filter(([, value]) => value === best).map(([id]) => id));
}

function Scoreboard({ comparison }: { comparison: Comparison }) {
  const match = comparison.thresholds.match;
  const list = metrics(match);
  const best = new Map(list.map(metric => [metric.key, bestRuns(metric, comparison.runs)]));
  return <section className="cmp-section" aria-labelledby="cmp-scoreboard">
    <div className="cmp-section-head">
      <div><h2 id="cmp-scoreboard">Scoreboard</h2>
        <p className="microcopy">One card per run. <span className="cmp-best-tag">best</span> marks the leader where a
          higher (or lower) value is clearly better.</p></div>
    </div>
    <div className="cmp-scroll">
      <table className="cmp-board" aria-label="Scoreboard">
        <thead><tr><td className="cmp-board-corner" />
          {comparison.runs.map(run => {
            const state = runState(run);
            return <RunHead key={run.id} run={run} reference={comparison.reference}
              extra={<span className="cmp-run-meta"><span className={`cmp-state ${state.cls}`}>{state.label}</span>
                <span>{run.documents_done}/{run.documents_total} docs</span><span>{shortDate(run.created_at)}</span></span>} />;
          })}</tr></thead>
        <tbody>{list.map(metric => <tr key={metric.key}>
          <th scope="row" title={metric.help}>{metric.label}<small>{metric.help}</small></th>
          {comparison.runs.map(run => {
            const isRef = run.id === comparison.reference && metric.toned;
            const value = metric.value(run);
            const isBest = best.get(metric.key)?.has(run.id) ?? false;
            return <td key={run.id} className={`${isBest ? "cmp-best" : ""} ${run.id === comparison.reference ? "cmp-is-ref" : ""}`}>
              {isRef ? <span className="cmp-metric muted">reference</span>
                : value === null ? <span className="cmp-metric muted">{metric.missing?.(run) ?? "—"}</span>
                  : <span className="cmp-metric">{(metric.format ?? pct)(value)}</span>}
              {isBest && <span className="cmp-best-tag">best</span>}
              {metric.bar && !isRef && value !== null && <span className={`cmp-bar ${metric.toned ? tone(value, match).cls : ""}`}>
                <i style={{ width: `${Math.max(2, Math.round(value * 100))}%` }} /></span>}
              {!isRef && value !== null && metric.detail && <small>{metric.detail(run)}</small>}
            </td>;
          })}
        </tr>)}</tbody>
      </table>
    </div>
  </section>;
}

/* ------------------------------------------------------------------ field matrix */

function FieldMatrix({ comparison }: { comparison: Comparison }) {
  const match = comparison.thresholds.match;
  function exportCsv() {
    const names = exportNames(comparison.runs);
    const header = ["field", "type", ...comparison.runs.flatMap(run => ["agreement", "agree_count", "scored", "filled", "documents"]
      .map(column => `${names.get(run.id)} ${column}`))];
    const rows = comparison.fields.map(field => [field.name, fieldType(field), ...comparison.runs.flatMap(run => {
      const score = run.fields[field.name];
      return score ? [score.agreement ?? "", score.agree_count, score.scored, score.filled, score.documents] : ["", "", "", "", ""];
    })]);
    const overall = ["all fields", "", ...comparison.runs.flatMap(run => [run.agreement ?? "", "", "", "", ""])];
    download("field-matrix.csv", "text/csv", toCsv([header, ...rows, overall]));
  }
  return <section className="cmp-section" aria-labelledby="cmp-fields">
    <div className="cmp-section-head">
      <div><h2 id="cmp-fields">Field matrix</h2>
        <p className="microcopy">Agreement per schema field across all documents, and how often each run filled it.</p></div>
      <button type="button" className="secondary cmp-small" onClick={exportCsv}><Download size={13} /> Field matrix CSV</button>
    </div>
    <div className="cmp-scroll">
      <table className="cmp-table cmp-matrix" aria-label="Field matrix">
        <thead><tr><th scope="col">Field</th>
          {comparison.runs.map(run => <RunHead key={run.id} run={run} reference={comparison.reference} />)}</tr></thead>
        <tbody>
          {comparison.fields.map(field => <tr key={field.name}>
            <th scope="row"><span className="cmp-field-name">{field.name}</span><small>{fieldType(field)}</small></th>
            {comparison.runs.map(run => {
              const score = run.fields[field.name];
              const isRef = run.id === comparison.reference;
              const t = tone(isRef ? null : score?.agreement, match);
              const label = `${runName(run)} on ${field.name}: ${isRef ? "reference" : `${pct(score?.agreement)} agreement, ${t.word}`}; filled ${score?.filled ?? 0} of ${score?.documents ?? 0} documents`;
              return <td key={run.id} className={`cmp-heat ${t.cls}`} aria-label={label}
                title={score && !isRef ? `${score.agree_count}/${score.scored} scored documents agree` : undefined}>
                <strong>{isRef ? "ref" : score?.agreement === null || score?.agreement === undefined ? "n/a" : pct(score.agreement)}</strong>
                <small>filled {score?.filled ?? 0}/{score?.documents ?? 0} docs</small>
              </td>;
            })}
          </tr>)}
          <tr className="cmp-total">
            <th scope="row">All fields</th>
            {comparison.runs.map(run => {
              const isRef = run.id === comparison.reference;
              const t = tone(isRef ? null : run.agreement, match);
              return <td key={run.id} className={`cmp-heat ${t.cls}`}
                aria-label={`${runName(run)} overall: ${isRef ? "reference" : `${pct(run.agreement)} agreement, ${t.word}`}`}>
                <strong>{isRef ? "ref" : run.agreement === null ? "n/a" : pct(run.agreement)}</strong>
                <small>fill {pct(run.fill_rate)}</small>
              </td>;
            })}
          </tr>
        </tbody>
      </table>
    </div>
  </section>;
}

/* ------------------------------------------------------------------ document matrix */

interface DocumentMatrixProps {
  comparison: Comparison;
  cells: Map<string, CompareCell>;
  selected: string | null;
  onOpen: (id: string) => void;
}

function DocumentMatrix({ comparison, cells, selected, onOpen }: DocumentMatrixProps) {
  const [sort, setSort] = useState<"name" | "disagreement">("name");
  const match = comparison.thresholds.match;
  const rows = useMemo(() => {
    const scored = comparison.documents.map(document => {
      const perRun = new Map(comparison.runs.map(run => [run.id, documentScore(comparison, cells, document.id, run.id)]));
      const lowest = Math.min(...[...perRun.values()].map(score => score.agreement ?? Infinity));
      return { document, perRun, lowest };
    });
    return sort === "name" ? scored : [...scored].sort((a, b) => a.lowest - b.lowest);
  }, [comparison, cells, sort]);

  return <section className="cmp-section" aria-labelledby="cmp-docs">
    <div className="cmp-section-head">
      <div><h2 id="cmp-docs">Document matrix</h2>
        <p className="microcopy">Mean agreement over all fields per document. Open a document to see every value.</p></div>
      <div className="segmented" role="group" aria-label="Sort documents">
        <button type="button" className={sort === "name" ? "active" : ""} aria-pressed={sort === "name"} onClick={() => setSort("name")}>By name</button>
        <button type="button" className={sort === "disagreement" ? "active" : ""} aria-pressed={sort === "disagreement"}
          onClick={() => setSort("disagreement")}>Most disagreement</button>
      </div>
    </div>
    {!rows.length ? <p className="microcopy cmp-pad">The selected runs have no documents.</p> : <div className="cmp-scroll">
      <table className="cmp-table cmp-matrix" aria-label="Document matrix">
        <thead><tr><th scope="col">Document</th>
          {comparison.runs.map(run => <RunHead key={run.id} run={run} reference={comparison.reference} />)}</tr></thead>
        <tbody>{rows.map(({ document, perRun }) => <tr key={document.id} className={`cmp-clickable ${selected === document.id ? "cmp-selected" : ""}`}
          onClick={() => onOpen(document.id)}>
          <th scope="row"><button type="button" className="cmp-link" aria-expanded={selected === document.id}
            onClick={event => { event.stopPropagation(); onOpen(document.id); }}>{document.name}</button></th>
          {comparison.runs.map(run => {
            const status = document.status[run.id];
            const score = perRun.get(run.id)!;
            const isRef = run.id === comparison.reference;
            if (status !== "done") return <td key={run.id} className="cmp-heat cmp-none" aria-label={`${runName(run)}: ${status ?? "not in run"}`}>
              <StatusChip status={status} /></td>;
            const t = tone(isRef ? null : score.agreement, match);
            return <td key={run.id} className={`cmp-heat ${t.cls}`}
              aria-label={`${runName(run)} on ${document.name}: ${isRef ? "reference" : `${pct(score.agreement)} agreement, ${t.word}`}; ${score.filled} of ${score.total} fields filled`}>
              <strong>{isRef ? "ref" : score.agreement === null ? "n/a" : pct(score.agreement)}</strong>
              <small>{score.filled}/{score.total} fields</small>
            </td>;
          })}
        </tr>)}</tbody>
      </table>
    </div>}
  </section>;
}

/* ------------------------------------------------------------------ drill-down */

interface DrilldownProps {
  comparison: Comparison;
  cells: Map<string, CompareCell>;
  documentId: string;
  active: SourceTarget | null;
  onClose: () => void;
  onSource: (target: SourceTarget) => void;
}

function Drilldown({ comparison, cells, documentId, active, onClose, onSource }: DrilldownProps) {
  const [expanded, setExpanded] = useState<Set<string>>(() => new Set());
  const [onlyDisagreement, setOnlyDisagreement] = useState(false);
  const match = comparison.thresholds.match;
  const doc = comparison.documents.find(item => item.id === documentId);
  if (!doc) return <div className="panel cmp-drill cmp-pad">
    <p className="microcopy">This document is not part of the selected runs.</p>
    <button type="button" className="secondary cmp-small" onClick={onClose}>Close</button></div>;

  const runs = comparison.runs;
  const disagrees = (field: SchemaField) => {
    const cell = cells.get(cellKey(documentId, field.name));
    return !!cell && Object.values(cell.agreement).some(score => score !== null && score < match);
  };
  const fields = comparison.fields.filter(field => !onlyDisagreement || disagrees(field));
  const toggle = (name: string) => setExpanded(current => {
    const next = new Set(current);
    if (next.has(name)) next.delete(name); else next.add(name);
    return next;
  });
  const isActive = (target: Omit<SourceTarget, "documentId">) => !!active && active.runId === target.runId
    && active.field === target.field && active.index === target.index;

  const exportCsv = () => {
    const names = exportNames(runs);
    const header = ["field", "item", ...runs.flatMap(run => ["value", "agreement", "confidence", "grounded"].map(column => `${names.get(run.id)} ${column}`))];
    const rows: unknown[][] = [];
    for (const field of comparison.fields) {
      const cell = cells.get(cellKey(documentId, field.name));
      rows.push([field.name, "", ...runs.flatMap(run => {
        const value = cell?.values[run.id];
        if (!value) return [doc.status[run.id] ?? "not in run", "", "", ""];
        return [value.display, cell?.agreement[run.id] ?? "", value.confidence ?? "", `${value.grounded}/${value.leaves}`];
      })]);
      if (!cell?.rows) continue;
      const pivot = pivotOf(cell, runs, comparison.reference);
      cell.rows.forEach((row, index) => rows.push([field.name, index + 1, ...runs.flatMap(run => row.cells[run.id] === undefined
        ? ["", "", "", ""]
        : [summarize(row.cells[run.id], field), run.id === pivot ? "pivot" : row.scores?.[run.id] ?? "unmatched", "", ""])]));
    }
    download(`${doc.name.replace(/[^\w.-]+/g, "_")}-comparison.csv`, "text/csv", toCsv([header, ...rows]));
  };

  return <section className="panel cmp-drill" aria-labelledby="cmp-drill-title">
    <div className="cmp-section-head cmp-drill-head">
      <div><span className="eyebrow">DOCUMENT DRILL-DOWN</span><h2 id="cmp-drill-title">{doc.name}</h2>
        <p className="microcopy">Click a value to see it in the source text. Lists expand into aligned items.</p></div>
      <div className="cmp-head-actions">
        <label className="cmp-check"><input type="checkbox" checked={onlyDisagreement}
          onChange={event => setOnlyDisagreement(event.target.checked)} /> Only fields with disagreement</label>
        <button type="button" className="secondary cmp-small" onClick={exportCsv}><Download size={13} /> Drill-down CSV</button>
      </div>
      <button type="button" className="icon-button" aria-label="Close drill-down" onClick={onClose}><X size={14} /></button>
    </div>
    <div className="cmp-scroll">
      <table className="cmp-table cmp-drill-table" aria-label={`Values in ${doc.name}`}>
        <thead><tr><th scope="col">Field</th>
          {runs.map(run => {
            const score = documentScore(comparison, cells, documentId, run.id);
            return <RunHead key={run.id} run={run} reference={comparison.reference}
              extra={doc.status[run.id] === "done"
                ? <span className="cmp-run-meta"><Score score={score.agreement} match={match} isReference={run.id === comparison.reference} />
                  <span>{score.filled}/{score.total} filled</span></span>
                : <span className="cmp-run-meta"><StatusChip status={doc.status[run.id]} /></span>} />;
          })}</tr></thead>
        <tbody>
          {!fields.length && <tr><td colSpan={runs.length + 1} className="microcopy">All fields agree for this document.</td></tr>}
          {fields.map(field => {
            const cell = cells.get(cellKey(documentId, field.name));
            const open = expanded.has(field.name);
            const rows = cell?.rows ?? [];
            const pivot = cell && field.array ? pivotOf(cell, runs, comparison.reference) : null;
            return [
              <tr key={field.name} className={field.array ? "cmp-array-row" : ""}>
                <th scope="row">
                  {field.array && rows.length > 0
                    ? <button type="button" className="cmp-expand" aria-expanded={open} onClick={() => toggle(field.name)}>
                      {open ? <ChevronDown size={13} /> : <ChevronRight size={13} />}
                      <span className="cmp-field-name">{field.name}</span><small>{rows.length} aligned</small></button>
                    : <><span className="cmp-field-name">{field.name}</span><small>{fieldType(field)}</small></>}
                </th>
                {runs.map(run => <ValueCell key={run.id} run={run} field={field} cell={cell} match={match}
                  status={doc.status[run.id]} isReference={run.id === comparison.reference}
                  active={isActive({ runId: run.id, field: field.name, scalar: !field.array })}
                  onSource={() => onSource({ runId: run.id, documentId, field: field.name, scalar: !field.array && field.type !== "object" })} />)}
              </tr>,
              ...(open && cell ? rows.map((row, index) => <AlignedRow key={`${field.name}-${index}`} row={row} index={index}
                field={field} cell={cell} runs={runs} pivot={pivot} match={match}
                isActive={runId => !!active && active.runId === runId && active.field === field.name && active.index === itemIndex(cell, runId, row)}
                onSource={(runId, itemAt) => onSource({ runId, documentId, field: field.name, index: itemAt, scalar: false })} />) : []),
            ];
          })}
        </tbody>
      </table>
    </div>
    {fields.some(field => field.array && expanded.has(field.name)) && <p className="microcopy cmp-pad">
      Items are aligned to the reference run, or to the run with the most items (marked <b>pivot</b>). Badges show each
      item's similarity to the aligned pivot item; <b>unmatched</b> items found no counterpart.</p>}
  </section>;
}

function itemIndex(cell: CompareCell, runId: string, row: CompareRow): number | undefined {
  if (row.indexes && runId in row.indexes) return row.indexes[runId];
  const items = cell.values[runId]?.value;
  if (!Array.isArray(items) || row.cells[runId] === undefined) return undefined;
  const wanted = JSON.stringify(row.cells[runId]);
  const index = items.findIndex(item => JSON.stringify(item) === wanted);
  return index >= 0 ? index : undefined;
}

interface ValueCellProps {
  run: CompareRunStats;
  field: SchemaField;
  cell: CompareCell | undefined;
  status: RunStatus | undefined;
  match: number;
  isReference: boolean;
  active: boolean;
  onSource: () => void;
}

function ValueCell({ run, field, cell, status, match, isReference, active, onSource }: ValueCellProps) {
  const value: CompareCellValue | undefined = cell?.values[run.id];
  if (!value) return <td className="cmp-heat cmp-none">{status === "done"
    ? <span className="cmp-run-status missing">no record</span> : <StatusChip status={status} />}</td>;
  const score = cell?.agreement[run.id];
  const empty = isEmpty(value.value);
  const t = tone(isReference || field.array ? null : score, match);
  const display = empty ? "—" : field.array ? `${value.count ?? 0} item${value.count === 1 ? "" : "s"}` : value.display || formatValue(value.value);
  return <td className={`cmp-heat ${field.array ? "cmp-none" : t.cls} ${active ? "cmp-active" : ""}`}>
    <button type="button" className="cmp-value" disabled={empty} onClick={onSource} title="Show in source">{display}</button>
    <span className="cmp-chips">
      <Score score={score} match={match} isReference={isReference} />
      {!empty && value.confidence !== null && <span className={`cmp-conf ${confidenceTone(value.confidence)}`}
        title={`Self-reported confidence (${run.confidence_method})`}>conf {confidenceLabel(value.confidence)}</span>}
      {!empty && (value.grounding
        ? <span className={`grounding ${groundingClass(value.grounding)}`}>{groundingText(value.grounding)}</span>
        : value.leaves > 0 && <span className={`grounding ${value.grounded === value.leaves ? "exact" : value.grounded ? "approximate" : "ungrounded"}`}>
          {value.grounded}/{value.leaves} grounded</span>)}
    </span>
  </td>;
}

interface AlignedRowProps {
  row: CompareRow;
  index: number;
  field: SchemaField;
  cell: CompareCell;
  runs: CompareRunStats[];
  pivot: string | null;
  match: number;
  isActive: (runId: string) => boolean;
  onSource: (runId: string, index: number | undefined) => void;
}

function AlignedRow({ row, index, field, cell, runs, pivot, match, isActive, onSource }: AlignedRowProps) {
  return <tr className="cmp-item-row">
    <th scope="row"><span className="cmp-item-index">{field.name} #{index + 1}</span></th>
    {runs.map(run => {
      const item = row.cells[run.id];
      if (item === undefined) return <td key={run.id} className="cmp-gap"
        aria-label={cell.values[run.id] ? `${runName(run)}: no matching item` : `${runName(run)}: no result`}>
        {cell.values[run.id] ? "—" : ""}</td>;
      const score = row.scores?.[run.id];
      return <td key={run.id} className={isActive(run.id) ? "cmp-active" : ""}>
        <button type="button" className="cmp-item" title={describe(item, field)}
          onClick={() => onSource(run.id, itemIndex(cell, run.id, row))}>{summarize(item, field)}</button>
        <span className="cmp-chips">
          {run.id === pivot ? <span className="cmp-badge cmp-none" title="Other runs' items are aligned to this one">pivot</span>
            : score !== undefined ? <Score score={score} match={match} />
              : <span className="cmp-badge cmp-s0" title="No counterpart in the pivot run">unmatched</span>}
        </span>
      </td>;
    })}
  </tr>;
}

/* ------------------------------------------------------------------ source panel */

interface SourcePanelProps {
  target: SourceTarget;
  comparison: Comparison;
  loaders: { document: (id: string) => Promise<StoredDocument>; run: (id: string) => Promise<Run> };
  onClose: () => void;
}

function SourcePanel({ target, comparison, loaders, onClose }: SourcePanelProps) {
  const [data, setData] = useState<{ document: StoredDocument; run: Run } | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [picked, setPicked] = useState<string | null>(null);
  const run = comparison.runs.find(item => item.id === target.runId);

  useEffect(() => {
    let cancelled = false;
    Promise.all([loaders.document(target.documentId), loaders.run(target.runId)])
      .then(([document, fullRun]) => { if (!cancelled) setData({ document, run: fullRun }); },
        failure => { if (!cancelled) setError(message(failure)); });
    return () => { cancelled = true; };
  }, [loaders, target.documentId, target.runId]);

  const values = useMemo<FieldValue[]>(() => data?.run.documents.find(item => item.document_id === target.documentId)
    ?.values.filter(value => value.path[0] === target.field) ?? [], [data, target.documentId, target.field]);
  const initial = useMemo(() => {
    if (target.scalar) return values.find(value => value.path.length === 1) ? pathKey([target.field]) : null;
    if (target.index === undefined) return null;
    const item = values.filter(value => value.path[1] === target.index);
    const hit = item.find(value => value.start !== null) ?? item[0];
    return hit ? pathKey(hit.path) : null;
  }, [values, target]);
  const selected = picked ?? initial;
  const located = values.filter(value => value.start !== null).length;

  return <aside className="panel cmp-source" aria-label="Source panel">
    <div className="cmp-section-head">
      <div><span className="eyebrow">SOURCE · {target.field}{target.index !== undefined ? ` #${target.index + 1}` : ""}</span>
        <h2>{run ? runName(run) : target.runId}</h2>
        <p className="microcopy">{data?.document.name ?? comparison.documents.find(item => item.id === target.documentId)?.name}</p></div>
      <button type="button" className="icon-button" aria-label="Close source panel" onClick={onClose}><X size={14} /></button>
    </div>
    {error ? <div className="notice cmp-pad" role="alert"><CircleAlert size={15} /><span>{error}</span></div>
      : !data ? <p className="cmp-status-line cmp-pad"><LoaderCircle size={13} className="spin" /> Loading source…</p>
        : <>
          <p className="microcopy cmp-pad">{values.length
            ? `${located} of ${values.length} value${values.length === 1 ? "" : "s"} located in the text.`
            : "This run returned no values for this field."}</p>
          {values.length > 0 && <ul className="cmp-leaves">{values.map(value => {
            const key = pathKey(value.path);
            const label = value.path.slice(1).map(part => typeof part === "number" ? `#${part + 1}` : part).join(" › ") || value.field;
            return <li key={key}><button type="button" className={selected === key ? "active" : ""} aria-pressed={selected === key}
              onClick={() => setPicked(key)}>
              <span className="cmp-leaf-path">{label}</span>
              <span className="cmp-leaf-value">{formatValue(value.value)}</span>
              {value.confidence !== null && <span className={`cmp-conf ${confidenceTone(value.confidence)}`}>conf {confidenceLabel(value.confidence)}</span>}
              <span className={`grounding ${groundingClass(value.grounding)}`}>{groundingText(value.grounding)}</span>
            </button></li>;
          })}</ul>}
          <div className="cmp-source-text">
            <ValueHighlighter text={data.document.text} values={values} selected={selected} onSelect={setPicked}
              pageStarts={data.document.page_starts} />
          </div>
        </>}
  </aside>;
}
