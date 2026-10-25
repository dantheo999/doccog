/**
 * Schema suggestions: the "Suggest with" picker (shared + per-solution), suggestion cards, the side-by-side
 * field comparison and the Solution page panel that tests one solution's suggestion on its own.
 */
import { useCallback, useEffect, useId, useMemo, useState } from "react";
import type { CSSProperties } from "react";
import { CircleCheck, FolderInput, GitCompareArrows, LoaderCircle, Sparkles, Trash2 } from "lucide-react";
import { message, workbench } from "../api";
import { href, navigate } from "../router";
import type { SchemaDraft, SchemaField, SchemaSuggestion, SettingSpec, SolutionInfo } from "../types";
import { useWorkspace } from "../workspace";
import SettingsForm, { settingsErrors } from "./SettingsForm";
import {
  MAX_EXEMPLARS, MAX_GUIDANCE, SHARED_ID, SHARED_NAME, compareDrafts, formatSeconds, formatTokens,
  formatWhen, loadSuggestionSettings, loadWorkspaceDraft, sameDraft, schemaDepth, storeSuggestionSettings,
  storeWorkspaceDraft, totalFields, typeLabel, whenLabels,
} from "./schemaSuggestionModel";
import type { ComparisonRow } from "./schemaSuggestionModel";

const SHARED_ACCENT = "#b4a2ff";

/** Re-render every second while `active`, for elapsed-time counters. */
function useNow(active: boolean): number {
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    if (!active) return;
    setNow(Date.now());
    const timer = setInterval(() => setNow(Date.now()), 1000);
    return () => clearInterval(timer);
  }, [active]);
  return now;
}

// --- Small building blocks ---------------------------------------------------------------------

/** Read-only, compact view of a schema's fields. */
export function FieldTree({ fields, label }: { fields: SchemaField[]; label?: string }) {
  return <ul className="field-tree" aria-label={label}>
    {fields.map((field, index) => <li key={`${field.name}-${index}`}>
      <div className="field-tree-row">
        <code>{field.name}</code>
        <span className="field-tree-type">{typeLabel(field)}</span>
        {field.array && <span className="list-mark">list</span>}
        {field.type === "enum" && field.options.length > 0 && <span className="field-tree-options" title={field.options.join(" / ")}>
          {field.options.join(" / ")}</span>}
      </div>
      {field.description && <small>{field.description}</small>}
      {field.type === "object" && (field.fields ?? []).length > 0 && <FieldTree fields={field.fields} />}
    </li>)}
  </ul>;
}

/**
 * SettingsForm for suggestion-only settings. Keys are prefixed so the generated control ids stay unique when
 * the same setting (e.g. reasoning_effort) also appears in a solution's run settings on the same page.
 */
export function SuggestionSettings({ specs, values, onChange, idPrefix, disabled }: {
  specs: SettingSpec[]; values: Record<string, unknown>; onChange: (values: Record<string, unknown>) => void;
  idPrefix: string; disabled?: boolean;
}) {
  const prefixed = useMemo(() => specs.map(spec => ({ ...spec, key: `${idPrefix}${spec.key}` })), [specs, idPrefix]);
  const shown = Object.fromEntries(specs.map(spec => [`${idPrefix}${spec.key}`, values[spec.key]]));
  return <div role="group" aria-label="Suggestion settings" className="suggestion-settings">
    <SettingsForm specs={prefixed} values={shown} disabled={disabled}
      onChange={next => onChange(Object.fromEntries(specs.map(spec => [spec.key, next[`${idPrefix}${spec.key}`]])))} />
  </div>;
}

export interface SuggestJob {
  status: "running" | "done" | "error";
  started: number;
  finished?: number;
  error?: string;
  fields?: number;
  /** Shared suggestion only: kept as a card, not loaded, because the editor draft changed while it ran. */
  kept?: boolean;
}

function JobStatus({ job, now, doneText }: { job?: SuggestJob; now: number; doneText?: string }) {
  if (!job) return null;
  const seconds = ((job.finished ?? now) - job.started) / 1000;
  if (job.status === "running") return <span className="suggest-job running">
    <LoaderCircle size={12} className="spin" />Suggesting… {formatSeconds(Math.max(0, seconds))}</span>;
  if (job.status === "error") return <span className="suggest-job error" role="alert">{job.error}</span>;
  return <span className="suggest-job done" role="status"><CircleCheck size={12} />
    {doneText ?? `${job.fields ?? 0} fields · in Suggestions`} · {formatSeconds(seconds)}</span>;
}

// --- Workspace: "Suggest with" picker ----------------------------------------------------------

export interface SuggestWithPickerProps {
  solutions: SolutionInfo[];
  selected: string[];
  onSelectedChange: (ids: string[]) => void;
  jobs: Record<string, SuggestJob>;
  effort: string;
  onEffortChange: (effort: string) => void;
  settings: Record<string, Record<string, unknown>>;
  onSettingsChange: (solutionId: string, values: Record<string, unknown>) => void;
  /** Invalid suggestion settings per chosen solution; Suggest stays disabled while any are listed. */
  errors?: Record<string, string[]>;
  /** Scroll to the schema editor (offered once the shared draft has loaded into it). */
  onShowEditor?: () => void;
}

/** The shared suggestion first (the default), then every solution that can suggest; the rest disabled with reasons. */
export function SuggestWithPicker({ solutions, selected, onSelectedChange, jobs, effort, onEffortChange, settings,
  onSettingsChange, errors = {}, onShowEditor }: SuggestWithPickerProps) {
  const now = useNow(Object.values(jobs).some(job => job.status === "running"));
  const toggle = (id: string) => onSelectedChange(selected.includes(id) ? selected.filter(item => item !== id) : [...selected, id]);
  const capable = solutions.filter(item => item.schema_suggestion?.supported);
  const incapable = solutions.filter(item => item.schema_suggestion && !item.schema_suggestion.supported);
  const sharedOn = selected.includes(SHARED_ID);
  const sharedJob = jobs[SHARED_ID];
  const sharedLoaded = sharedJob?.status === "done" && !sharedJob.kept;

  return <fieldset className="suggest-with">
    <legend className="field-label">Suggest with</legend>
    <div className={`suggest-option ${sharedOn ? "selected" : ""}`} style={{ "--suggest-accent": SHARED_ACCENT } as CSSProperties}>
      <input id="suggest-with-shared" type="checkbox" checked={sharedOn} onChange={() => toggle(SHARED_ID)}
        aria-describedby="suggest-with-shared-help" />
      <div className="suggest-option-text">
        <label htmlFor="suggest-with-shared">{SHARED_NAME}</label><span className="pill">default</span>
        <p id="suggest-with-shared-help" className="microcopy">Solution-neutral: one Azure OpenAI call reads the exemplars' full
          text and proposes a schema. Its draft loads into the editor below.</p>
        <JobStatus job={sharedJob} now={now}
          doneText={sharedJob?.kept ? "Kept under Suggestions; the editor draft changed while it ran" : "Loaded into the editor"} />
        {sharedLoaded && onShowEditor && <button type="button" className="text-button show-editor" onClick={onShowEditor}>Show in editor</button>}
      </div>
      <label className="inline-select"><span>Reasoning</span>
        <select aria-label="Shared suggestion reasoning effort" value={effort} onChange={event => onEffortChange(event.target.value)}>
          {["low", "medium", "high"].map(item => <option key={item}>{item}</option>)}</select></label>
    </div>
    {capable.map(solution => {
      const id = `suggest-with-${solution.id}`;
      const on = selected.includes(solution.id);
      const specs = solution.schema_suggestion?.settings ?? [];
      const ready = solution.status.ready;
      const problems = on && ready ? errors[solution.id] ?? [] : [];
      return <div key={solution.id} className={`suggest-option ${on ? "selected" : ""} ${ready ? "" : "disabled"}`}
        style={{ "--suggest-accent": solution.accent } as CSSProperties}>
        <input id={id} type="checkbox" checked={on && ready} disabled={!ready} onChange={() => toggle(solution.id)}
          aria-describedby={`${id}-help`} />
        <div className="suggest-option-text">
          <label htmlFor={id}>{solution.name}</label>
          <p id={`${id}-help`} className="microcopy">{ready ? solution.schema_suggestion?.method
            : `Not ready: ${solution.status.detail || "check the solution page."}`}</p>
          <JobStatus job={jobs[solution.id]} now={now} />
          {on && ready && specs.length > 0 && <details className="suggest-option-settings">
            <summary>{solution.name} suggestion settings</summary>
            <SuggestionSettings specs={specs} idPrefix={`ws-${solution.id}-`} values={settings[solution.id] ?? {}}
              onChange={values => onSettingsChange(solution.id, values)} />
          </details>}
          {problems.length > 0 && <ul className="suggest-option-errors" aria-label={`${solution.name} suggestion settings errors`}>
            {problems.map(item => <li key={item}>{item}</li>)}</ul>}
        </div>
      </div>;
    })}
    {incapable.map(solution => {
      const id = `suggest-with-${solution.id}`;
      return <div key={solution.id} className="suggest-option disabled">
        <input id={id} type="checkbox" checked={false} disabled aria-describedby={`${id}-help`} />
        <div className="suggest-option-text">
          <label htmlFor={id}>{solution.name}</label><span className="pill">not supported</span>
          <p id={`${id}-help`} className="microcopy">{solution.schema_suggestion?.method}</p>
        </div>
      </div>;
    })}
  </fieldset>;
}

// --- Workspace: suggestion cards and comparison ------------------------------------------------

interface CardProps {
  suggestion: SchemaSuggestion;
  /** When it ran, unique among suggestions from the same solution (see whenLabels). */
  when: string;
  accent: string;
  inEditor: boolean;
  compared: boolean;
  onCompare: (on: boolean) => void;
  onUse: () => void;
  onDelete: () => void;
}

function SuggestionCard({ suggestion, when, accent, inEditor, compared, onCompare, onUse, onDelete }: CardProps) {
  const fields = suggestion.draft?.fields ?? [];
  const settings = Object.entries(suggestion.settings ?? {});
  const notes = suggestion.notes ?? [];
  return <article className={`suggestion-card ${compared ? "compared" : ""}`} aria-label={`${suggestion.solution_name}, ${when}`}
    style={{ "--suggest-accent": accent } as CSSProperties}>
    <header>
      <label className="compare-toggle" title="Include in the comparison">
        <input type="checkbox" checked={compared} onChange={event => onCompare(event.target.checked)}
          aria-label={`Compare ${suggestion.solution_name} suggestion from ${when}`} />Compare</label>
      <strong>{suggestion.solution_name}</strong>
      {inEditor && <span className="pill success">in editor</span>}
      <span className="pill">{fields.length} fields · depth {schemaDepth(fields)}</span>
    </header>
    <p className="suggestion-method" title={suggestion.method}>{suggestion.method}</p>
    <dl className="suggestion-facts">
      <div><dt>Exemplars</dt><dd>{(suggestion.exemplars ?? []).join(", ") || "—"}</dd></div>
      {suggestion.guidance && <div><dt>Guidance</dt><dd className="guidance" title={suggestion.guidance}>“{suggestion.guidance}”</dd></div>}
      <div><dt>When</dt><dd>{when} · took {formatSeconds(suggestion.elapsed_seconds)}</dd></div>
      <div><dt>Tokens</dt><dd>{formatTokens(suggestion.usage)}</dd></div>
      {settings.length > 0 && <div><dt>Settings</dt><dd>{settings.map(([key, value]) => `${key} ${String(value)}`).join(" · ")}</dd></div>}
    </dl>
    {notes.length > 0 && <details className="suggestion-notes"><summary>{notes.length} note{notes.length === 1 ? "" : "s"}</summary>
      <ul>{notes.map((note, index) => <li key={index}>{note}</li>)}</ul></details>}
    <details className="suggestion-fields"><summary>{totalFields(fields)} fields in all</summary><FieldTree fields={fields} /></details>
    <footer>
      <button className="secondary" onClick={onUse}>Use as draft</button>
      <span className="spacer" />
      <button className="icon-button danger" aria-label={`Delete ${suggestion.solution_name} suggestion from ${when}`} title="Delete"
        onClick={onDelete}><Trash2 size={13} /></button>
    </footer>
  </article>;
}

export interface SuggestionsSectionProps {
  suggestions: SchemaSuggestion[];
  solutions: SolutionInfo[];
  draft: SchemaDraft | null;
  compare: string[];
  onCompareChange: (ids: string[]) => void;
  onUse: (suggestion: SchemaSuggestion) => void;
  onDelete: (suggestion: SchemaSuggestion) => void;
}

const VISIBLE_CARDS = 6;

/** Saved suggestions (shared and per-solution together) for the current document set, newest first. */
export function SuggestionsSection({ suggestions, solutions, draft, compare, onCompareChange, onUse, onDelete }: SuggestionsSectionProps) {
  const [showAll, setShowAll] = useState(false);
  const visible = showAll ? suggestions : suggestions.slice(0, VISIBLE_CARDS);
  const compared = compare.map(id => suggestions.find(item => item.id === id)).filter((item): item is SchemaSuggestion => !!item);
  const accent = (suggestion: SchemaSuggestion) => suggestion.solution === SHARED_ID ? SHARED_ACCENT
    : solutions.find(item => item.id === suggestion.solution)?.accent ?? "#5a6b88";
  const when = useMemo(() => whenLabels(suggestions), [suggestions]);

  return <section className="suggestions" aria-labelledby="suggestions-heading">
    <div className="suggestions-heading">
      <h3 id="suggestions-heading">Suggestions</h3>
      <span className="microcopy">{suggestions.length
        ? `${suggestions.length} for this document set · tick Compare on two or more to see them side by side`
        : "Each suggestion you run is kept here, shared and per-solution together, so methods can be compared."}</span>
      <span className="spacer" />
      {compare.length > 0 && <button className="text-button" onClick={() => onCompareChange([])}>Clear comparison</button>}
    </div>
    {suggestions.length > 0 && <div className="suggestion-grid">
      {visible.map(suggestion => <SuggestionCard key={suggestion.id} suggestion={suggestion} when={when[suggestion.id]} accent={accent(suggestion)}
        inEditor={!!draft && sameDraft(draft, suggestion.draft)} compared={compare.includes(suggestion.id)}
        onCompare={on => onCompareChange(on ? [...compare.filter(id => id !== suggestion.id), suggestion.id] : compare.filter(id => id !== suggestion.id))}
        onUse={() => onUse(suggestion)} onDelete={() => onDelete(suggestion)} />)}
    </div>}
    {suggestions.length > VISIBLE_CARDS && <button className="text-button show-all" onClick={() => setShowAll(value => !value)}>
      {showAll ? "Show fewer" : `Show all ${suggestions.length} suggestions`}</button>}
    {compared.length === 1 && <p className="microcopy compare-hint"><GitCompareArrows size={12} />Tick one more suggestion to compare.</p>}
    {compared.length >= 2 && <SuggestionComparison suggestions={compared} when={when} />}
  </section>;
}

function statusText(row: ComparisonRow, names: string[]): string {
  if (row.status === "differs") return "Types differ";
  if (row.status === "unique") return `Only in ${names[row.cells.findIndex(Boolean)]}`;
  if (row.status === "partial") return `In ${row.present} of ${row.cells.length}`;
  return "Same";
}

/**
 * Rows = union of field paths; columns = suggestions; flags fields only one has and fields whose types differ.
 * `when` (from whenLabels over all listed suggestions) keeps column names in step with the cards.
 */
export function SuggestionComparison({ suggestions, when: givenWhen }: { suggestions: SchemaSuggestion[]; when?: Record<string, string> }) {
  const [onlyDifferences, setOnlyDifferences] = useState(false);
  const headingId = useId();
  const rows = useMemo(() => compareDrafts(suggestions.map(item => item.draft)), [suggestions]);
  const counts = { unique: rows.filter(row => row.status === "unique").length, differs: rows.filter(row => row.status === "differs").length };
  const shown = onlyDifferences ? rows.filter(row => row.status !== "same") : rows;
  const repeated = (name: string) => suggestions.filter(item => item.solution_name === name).length > 1;
  const when = useMemo(() => givenWhen ?? whenLabels(suggestions), [givenWhen, suggestions]);
  const names = suggestions.map(item => repeated(item.solution_name)
    ? `${item.solution_name} (${when[item.id] ?? formatWhen(item.created_at)})` : item.solution_name);

  return <div className="suggestion-compare" role="region" aria-labelledby={headingId}>
    <div className="suggestion-compare-heading">
      <h4 id={headingId}><GitCompareArrows size={14} />Comparison</h4>
      <span className="microcopy">{rows.length} field paths · {counts.unique} only in one · {counts.differs} with different types</span>
      <span className="spacer" />
      <label className="list-toggle"><input type="checkbox" checked={onlyDifferences}
        onChange={event => setOnlyDifferences(event.target.checked)} />Only differences</label>
    </div>
    <div className="suggestion-compare-scroll">
      <table className="wb-table suggestion-compare-table">
        <thead><tr>
          <th scope="col">Field path</th>
          {suggestions.map((item, index) => <th scope="col" key={item.id}>
            <span className="compare-col-name">{names[index]}</span>
            <small>{item.draft.fields.length} top-level · depth {schemaDepth(item.draft.fields)}</small>
          </th>)}
          <th scope="col">Agreement</th>
        </tr></thead>
        <tbody>
          {shown.map(row => <tr key={row.key} className={`row-${row.status}`} data-status={row.status}>
            <th scope="row" className="compare-path"><span className="path-parent">{row.parent}</span><b>{row.name}</b></th>
            {row.cells.map((cell, index) => <td key={index}>
              {cell ? <span className="compare-type" title={cell.description || undefined}>{typeLabel(cell)}
                {cell.type === "enum" && cell.options.length > 0 && <small> ({cell.options.length})</small>}
                {cell.array && <span className="list-mark">list</span>}</span>
                : <><span aria-hidden="true" className="absent">—</span><span className="visually-hidden">absent</span></>}
            </td>)}
            <td><span className={`agreement ${row.status}`}>{statusText(row, names)}</span></td>
          </tr>)}
          {!shown.length && <tr><td colSpan={suggestions.length + 2} className="microcopy">No differences: every field path has the same type in all suggestions.</td></tr>}
        </tbody>
      </table>
    </div>
  </div>;
}

// --- Solution page: test one solution's suggestion independently ------------------------------

/** "Schema suggestion" panel for the Solution page (or a one-line note when the solution cannot suggest). */
export function SolutionSuggestPanel({ solution }: { solution: SolutionInfo }) {
  const capability = solution.schema_suggestion;
  if (!capability) return null;
  if (!capability.supported) return <section className="panel suggest-panel unsupported" aria-labelledby="solution-suggest-heading">
    <div className="panel-title"><h2 id="solution-suggest-heading">Schema suggestion</h2><span className="pill">not supported</span></div>
    <p className="microcopy">{capability.method}</p>
  </section>;
  return <SupportedSuggestPanel solution={solution} method={capability.method} specs={capability.settings ?? []} />;
}

function SupportedSuggestPanel({ solution, method, specs }: { solution: SolutionInfo; method: string; specs: SettingSpec[] }) {
  const workspace = useWorkspace();
  const documents = useMemo(() => workspace.collection?.documents ?? [], [workspace.collection]);
  const [exemplars, setExemplars] = useState<string[]>([]);
  const [guidance, setGuidance] = useState("");
  const [settings, setSettings] = useState(() => loadSuggestionSettings(solution));
  const [started, setStarted] = useState<number | null>(null);
  const [result, setResult] = useState<SchemaSuggestion | null>(null);
  const [recent, setRecent] = useState<SchemaSuggestion[]>([]);
  const [error, setError] = useState<string | null>(null);
  const now = useNow(started !== null);
  const errors = settingsErrors(specs, settings);
  const recentWhen = useMemo(() => whenLabels(recent), [recent]);

  useEffect(() => { storeSuggestionSettings(solution.id, settings); }, [solution.id, settings]);
  useEffect(() => {
    const ids = new Set(documents.map(item => item.id));
    setExemplars(current => current.filter(id => ids.has(id)));
  }, [documents]);
  const loadRecent = useCallback(async () => {
    try {
      const all = await workbench.schemaSuggestions();
      setRecent(all.filter(item => item.solution === solution.id).sort((a, b) => b.created_at.localeCompare(a.created_at)));
    } catch { /* older backend or offline: the list is a convenience */ }
  }, [solution.id]);
  useEffect(() => { void loadRecent(); }, [loadRecent]);

  async function suggest() {
    if (!exemplars.length || started !== null) return;
    setStarted(Date.now());
    setError(null);
    try {
      const suggestion = await workbench.suggestSchemaWith(solution.id, exemplars, guidance, settings);
      setResult(suggestion);
      setRecent(current => [suggestion, ...current.filter(item => item.id !== suggestion.id)]);
    } catch (failure) { setError(message(failure)); }
    finally { setStarted(null); }
  }

  function openInWorkspace(draft: SchemaDraft) {
    const existing = loadWorkspaceDraft();
    const saved = workspace.schema ? { name: workspace.schema.name, description: workspace.schema.description, fields: workspace.schema.fields } : null;
    if (existing && !sameDraft(existing, draft) && !sameDraft(existing, saved)
      && !window.confirm("Replace the unsaved schema draft in the Workspace with this suggestion?")) return;
    if (!storeWorkspaceDraft(draft)) { setError("Browser storage is unavailable, so the draft cannot be handed to the Workspace."); return; }
    navigate("/workspace");
  }

  const ready = solution.status.ready;
  const canSuggest = ready && exemplars.length > 0 && !errors.length && started === null;
  const fields = result?.draft.fields ?? [];

  return <section className="panel suggest-panel" aria-labelledby="solution-suggest-heading">
    <div className="panel-title"><h2 id="solution-suggest-heading">Schema suggestion</h2>
      <span className="pill">tested on its own</span></div>
    <div className="suggest-panel-body">
      <p className="microcopy suggest-method">{method}</p>
      {documents.length ? <fieldset className="exemplar-picker">
        <legend className="field-label">Exemplars <span className="microcopy">{exemplars.length}/{MAX_EXEMPLARS}</span></legend>
        <ul>{documents.map(item => {
          const on = exemplars.includes(item.id);
          return <li key={item.id}><label>
            <input type="checkbox" checked={on} disabled={!on && exemplars.length >= MAX_EXEMPLARS}
              aria-label={`Use ${item.path} as suggestion exemplar`}
              onChange={() => setExemplars(current => on ? current.filter(id => id !== item.id) : [...current, item.id])} />
            <span title={item.path}>{item.path}</span><small>{item.characters.toLocaleString()} chars</small></label></li>;
        })}</ul>
      </fieldset> : <p className="microcopy">No document set — <a href={href("/workspace")}>choose a folder</a> in the Workspace first.</p>}
      <label className="suggest-guidance"><span className="field-label">Guidance (optional)</span>
        <textarea rows={2} maxLength={MAX_GUIDANCE} value={guidance} onChange={event => setGuidance(event.target.value)}
          placeholder="e.g. Focus on attendees, decisions and actions with owners and due dates." /></label>
      {specs.length > 0 && <SuggestionSettings specs={specs} values={settings} onChange={setSettings}
        idPrefix={`suggest-${solution.id}-`} disabled={started !== null} />}
      {errors.length > 0 && <ul className="schema-errors">{errors.map(item => <li key={item}>{item}</li>)}</ul>}
      <div className="inline-actions">
        <button className="primary" onClick={() => void suggest()} disabled={!canSuggest}>
          {started !== null ? <LoaderCircle size={15} className="spin" /> : <Sparkles size={15} />}
          {started !== null ? "Suggesting…" : `Suggest schema from ${exemplars.length || "selected"} exemplar${exemplars.length === 1 ? "" : "s"}`}</button>
        <span className="microcopy">{started !== null ? `Running for ${formatSeconds((now - started) / 1000)}. Local models on CPU can take minutes.`
          : !ready ? `Not ready: ${solution.status.detail}` : "Kept in the Workspace's Suggestions for comparison."}</span>
      </div>
      {error && <div className="notice error" role="alert"><span>{error}</span></div>}
      {result && <div className="suggest-result" role="region" aria-label="Suggested schema">
        <div className="suggest-result-head"><strong>{result.draft.name || "Untitled schema"}</strong>
          <span className="pill">{fields.length} fields · depth {schemaDepth(fields)}</span></div>
        <p className="microcopy">{formatWhen(result.created_at)} · took {formatSeconds(result.elapsed_seconds)} · {formatTokens(result.usage)}
          {result.exemplars.length > 0 && ` · from ${result.exemplars.join(", ")}`}</p>
        {result.notes.length > 0 && <ul className="suggest-result-notes">{result.notes.map((note, index) => <li key={index}>{note}</li>)}</ul>}
        <FieldTree fields={fields} label="Suggested fields" />
        <div className="inline-actions">
          <button className="secondary" onClick={() => openInWorkspace(result.draft)}><FolderInput size={14} />Open in Workspace</button>
          <span className="microcopy">Loads it into the Workspace editor as the working draft.</span>
        </div>
      </div>}
      {recent.length > 0 && <details className="suggest-recent">
        <summary>Earlier {solution.name} suggestions ({recent.length})</summary>
        <ul>{recent.slice(0, 8).map(item => <li key={item.id}>
          <button className={`text-button ${result?.id === item.id ? "active" : ""}`} onClick={() => setResult(item)}
            aria-label={`View ${solution.name} suggestion from ${recentWhen[item.id]}`}>
            {recentWhen[item.id]}</button>
          <small>{item.draft.fields.length} fields · {item.exemplars.length} exemplar{item.exemplars.length === 1 ? "" : "s"}</small>
        </li>)}</ul>
      </details>}
    </div>
  </section>;
}
