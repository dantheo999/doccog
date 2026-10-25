import { useCallback, useEffect, useMemo, useState } from "react";
import type { ReactNode } from "react";
import { CircleCheck, CircleDashed, Eye, ExternalLink, LoaderCircle, Play, RefreshCw, Square, Trash2 } from "lucide-react";
import { message, workbench } from "../api";
import { href, navigate, useRoute } from "../router";
import type { RunSummary, SolutionInfo } from "../types";
import { useWorkspace } from "../workspace";
import SettingsForm, { defaults, settingsErrors } from "../components/SettingsForm";
import RunViewer from "../components/RunViewer";
import LangExtractStudio from "../../LangExtractStudio";
import { loadRecipe } from "../../recipe";
import UnstractPanel from "../components/UnstractPanel";
import { SolutionSuggestPanel } from "../components/SchemaSuggestions";

const settingsKey = (id: string) => `doc-cognition.settings.${id}`;

function loadSettings(solution: SolutionInfo): Record<string, unknown> {
  const base = defaults(solution.settings);
  try {
    const stored = JSON.parse(localStorage.getItem(settingsKey(solution.id)) ?? "{}");
    return Object.fromEntries(Object.entries(base).map(([key, value]) => [key, key in stored ? stored[key] : value]));
  } catch { return base; }
}

interface Props { solution: SolutionInfo; onRefresh: () => void }

export default function SolutionPage({ solution, onRefresh }: Props) {
  const route = useRoute();
  const isLangExtract = solution.id === "langextract";
  const tab = isLangExtract ? (route.query.tab ?? (route.query.run ? "runs" : "studio")) : "runs";
  return <div className={`wb-page solution-page ${tab === "studio" ? "studio-tab" : ""}`}>
    <div className="page-heading solution-heading" style={{ borderColor: solution.accent }}>
      <div><span className="eyebrow">{solution.technology.toUpperCase()}</span><h1>{solution.name}</h1><p>{solution.tagline}</p></div>
      <span className={`status-pill ${solution.status.ready ? "ready" : ""}`} title={solution.status.detail}>
        {solution.status.ready ? "Ready" : "Not ready"}{solution.status.model && ` · ${solution.status.model}`}</span>
      <button className="icon-button" onClick={onRefresh} aria-label="Refresh status" title="Refresh status"><RefreshCw size={14} /></button>
      {solution.external_url && <a className="primary" href={solution.external_url} target="_blank" rel="noreferrer">
        Open {solution.name}<ExternalLink size={14} /></a>}
    </div>
    {!solution.status.ready && <div className="notice warning"><span>{solution.status.detail}</span></div>}
    {isLangExtract && <div className="segmented page-tabs" role="tablist">
      <a role="tab" aria-selected={tab === "studio"} className={tab === "studio" ? "active" : ""} href={href(solution.route, { tab: "studio" })}>Recipe studio</a>
      <a role="tab" aria-selected={tab === "runs"} className={tab === "runs" ? "active" : ""} href={href(solution.route, { tab: "runs" })}>Batch runs</a>
    </div>}
    {tab === "studio" ? <div className="studio-host"><LangExtractStudio embedded /></div>
      : <RunsWorkbench solution={solution}
        extra={isLangExtract ? recipeRequirement : undefined}
        aside={<>
          {/* Test this solution's own schema suggestion (or say why it has none), above Runs. */}
          <SolutionSuggestPanel solution={solution} />
          {solution.id === "unstract" && <UnstractPanel solution={solution} />}
        </>} />}
  </div>;
}

/** LangExtract runs reuse the recipe that is active in the studio. */
function recipeRequirement(): { settings?: Record<string, unknown>; problem?: string; label?: string } {
  try {
    const recipe = loadRecipe();
    if (!recipe) return { problem: "Generate or import a recipe in the Recipe studio first." };
    return { settings: { recipe }, label: `recipe v${recipe.version} · ${recipe.examples.length} example${recipe.examples.length === 1 ? "" : "s"}` };
  } catch (failure) { return { problem: `The saved recipe is invalid: ${message(failure)}` }; }
}

interface WorkbenchProps {
  solution: SolutionInfo;
  extra?: () => { settings?: Record<string, unknown>; problem?: string; label?: string };
  aside?: ReactNode;
}

function RunsWorkbench({ solution, extra, aside }: WorkbenchProps) {
  const route = useRoute();
  const workspace = useWorkspace();
  const documents = workspace.collection?.documents ?? [];
  const [settings, setSettings] = useState(() => loadSettings(solution));
  const [selected, setSelected] = useState<string[] | null>(null);
  const [label, setLabel] = useState("");
  const [runs, setRuns] = useState<RunSummary[]>([]);
  const [preview, setPreview] = useState<Record<string, string> | null>(null);
  const [busy, setBusy] = useState<"start" | "preview" | "prepare" | null>(null);
  const [prepared, setPrepared] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const requirement = extra?.();
  const chosen = useMemo(() => selected ?? documents.map(item => item.id), [selected, documents]);
  const errors = settingsErrors(solution.settings, settings);
  const needsImages = typeof settings.input_mode === "string" && settings.input_mode.includes("image");

  const loadRuns = useCallback(async () => {
    try { setRuns((await workbench.runs()).filter(run => run.solution === solution.id)); }
    catch (failure) { setError(message(failure)); }
  }, [solution.id]);
  useEffect(() => { void loadRuns(); }, [loadRuns, route.query.run]);
  const anyActive = runs.some(run => run.active || run.status === "running" || run.status === "queued");
  useEffect(() => {
    if (!anyActive) return;
    const timer = setInterval(() => void loadRuns(), 2500);
    return () => clearInterval(timer);
  }, [anyActive, loadRuns]);
  useEffect(() => {
    try { localStorage.setItem(settingsKey(solution.id), JSON.stringify(settings)); } catch { /* quota */ }
    setPreview(null);
  }, [settings, solution.id]);
  useEffect(() => { setSelected(null); }, [workspace.collection?.id]);

  async function start() {
    if (!workspace.schema || !chosen.length || requirement?.problem) return;
    setBusy("start");
    setError(null);
    try {
      const run = await workbench.startRun({ solution: solution.id, label, schema_id: workspace.schema.id,
        settings: { ...settings, ...(requirement?.settings ?? {}) }, document_ids: chosen });
      setLabel("");
      await loadRuns();
      navigate(solution.route, { tab: route.query.tab, run: run.id });
    } catch (failure) { setError(message(failure)); }
    finally { setBusy(null); }
  }

  async function showPreview() {
    if (!workspace.schema) return;
    setBusy("preview");
    try { setPreview(await workbench.preview(solution.id, workspace.schema.id, settings)); }
    catch (failure) { setError(message(failure)); }
    finally { setBusy(null); }
  }

  async function prepare() {
    setBusy("prepare");
    setError(null);
    try {
      for (const [index, id] of chosen.entries()) {
        setPrepared(`Pre-processing ${index + 1}/${chosen.length}…`);
        await workbench.prepare(solution.id, id, settings);
      }
      setPrepared(`Pre-processed ${chosen.length} document${chosen.length === 1 ? "" : "s"}.`);
    } catch (failure) { setError(message(failure)); setPrepared(null); }
    finally { setBusy(null); }
  }

  async function cancel(id: string) {
    try { await workbench.cancelRun(id); await loadRuns(); } catch (failure) { setError(message(failure)); }
  }

  async function remove(id: string) {
    if (!window.confirm("Delete this run and its results?")) return;
    try {
      await workbench.deleteRun(id);
      if (route.query.run === id) navigate(solution.route, { tab: route.query.tab });
      await loadRuns();
    } catch (failure) { setError(message(failure)); }
  }

  const checks: { ok: boolean; label: ReactNode }[] = [
    { ok: documents.length > 0, label: documents.length ? `${documents.length} documents in “${workspace.collection?.name}”`
      : <>No document set — <a href={href("/workspace")}>choose a folder</a></> },
    { ok: !!workspace.schema, label: workspace.schema ? `Schema “${workspace.schema.name}” v${workspace.schema.version}`
      : <>No agreed schema — <a href={href("/workspace")}>agree one</a></> },
    { ok: solution.status.ready, label: solution.status.ready ? `${solution.name} ready` : solution.status.detail },
    ...(requirement ? [{ ok: !requirement.problem, label: requirement.problem ?? requirement.label ?? "Ready" }] : []),
  ];
  const canStart = checks.every(check => check.ok) && chosen.length > 0 && !errors.length && busy === null;

  return <>
    {error && <div className="notice error" role="alert"><span>{error}</span><button aria-label="Dismiss error" onClick={() => setError(null)}>×</button></div>}
    {route.query.run && <RunViewer key={route.query.run} runId={route.query.run} onClose={() => navigate(solution.route, { tab: route.query.tab })} />}
    <div className="solution-grid">
      <section className="panel run-setup" aria-labelledby="setup-heading">
        <div className="panel-title"><h2 id="setup-heading">New run</h2></div>
        <ul className="checklist">{checks.map((check, index) => <li key={index} className={check.ok ? "ok" : ""}>
          {check.ok ? <CircleCheck size={14} /> : <CircleDashed size={14} />}<span>{check.label}</span></li>)}</ul>
        {documents.length > 0 && <details className="doc-picker">
          <summary>{chosen.length} of {documents.length} documents selected</summary>
          <div className="doc-picker-actions">
            <button className="text-button" onClick={() => setSelected(documents.map(item => item.id))}>All</button>
            <button className="text-button" onClick={() => setSelected([])}>None</button>
          </div>
          <ul>{documents.map(item => <li key={item.id}><label>
            <input type="checkbox" checked={chosen.includes(item.id)} onChange={() => setSelected(
              chosen.includes(item.id) ? chosen.filter(id => id !== item.id) : [...chosen, item.id])} />
            <span>{item.path}</span><small>{item.characters.toLocaleString()} chars</small></label></li>)}</ul>
        </details>}
        <h3 className="subheading">Settings</h3>
        <SettingsForm specs={solution.settings} values={settings} onChange={setSettings} disabled={busy === "start"} />
        {errors.length > 0 && <ul className="schema-errors">{errors.map(item => <li key={item}>{item}</li>)}</ul>}
        {needsImages && <div className="prepare-row">
          <button className="secondary" onClick={() => void prepare()} disabled={busy !== null || !chosen.length}>
            {busy === "prepare" ? <LoaderCircle size={14} className="spin" /> : <Play size={14} />}Pre-process page images</button>
          <span className="microcopy">{prepared ?? "Renders PDF pages once (cached). Runs do this automatically if skipped."}</span>
        </div>}
        <label className="run-label"><span className="field-label">Run label (optional)</span>
          <input value={label} onChange={event => setLabel(event.target.value)} placeholder={`${solution.name} · ${chosen.length} docs`} /></label>
        <div className="run-actions">
          <button className="primary" onClick={() => void start()} disabled={!canStart}>
            {busy === "start" ? <LoaderCircle size={15} className="spin" /> : <Play size={15} />}
            Run on {chosen.length} document{chosen.length === 1 ? "" : "s"}</button>
          <button className="secondary" onClick={() => void showPreview()} disabled={!workspace.schema || busy !== null}>
            <Eye size={14} />What gets sent</button>
        </div>
        {preview && <div className="preview-blocks">{Object.entries(preview).length === 0
          ? <p className="microcopy">No preview for this solution.</p>
          : Object.entries(preview).map(([title, text]) => <details key={title} open>
            <summary>{title}</summary><pre>{text}</pre></details>)}</div>}
        <p className="microcopy">Confidence: {solution.confidence_method}</p>
      </section>

      <div className="solution-side">
        {aside}
        <section className="panel run-list" aria-labelledby="runs-heading">
          <div className="panel-title"><h2 id="runs-heading">Runs</h2><span className="spacer" />
            {runs.length > 1 && <a className="secondary" href={href("/compare", { runs: runs.filter(run => run.status === "done").slice(0, 4).map(run => run.id).join(",") })}>
              Compare recent</a>}</div>
          {!runs.length && <p className="microcopy">No runs yet. Results appear here and in Compare.</p>}
          <ul>{runs.map(run => {
            const active = run.active || run.status === "running" || run.status === "queued";
            const progress = run.documents_total ? (run.documents_done + run.documents_failed) / run.documents_total : 0;
            return <li key={run.id} className={route.query.run === run.id ? "active" : ""}>
              <a href={href(solution.route, { tab: route.query.tab, run: run.id })}>
                <strong>{run.label}</strong>
                <small>{new Date(run.created_at).toLocaleString()} · schema v{run.schema_version} · {run.model}</small>
                <span className="run-progress"><span className="progress-bar"><span style={{ width: `${progress * 100}%` }} /></span>
                  {run.documents_done}/{run.documents_total}{run.documents_failed ? ` · ${run.documents_failed} failed` : ""} · {run.elapsed_seconds}s</span>
              </a>
              <span className={`run-status ${run.status}`}>{active && <LoaderCircle size={11} className="spin" />}{run.status}</span>
              {active ? <button className="icon-button" aria-label={`Cancel ${run.label}`} title="Cancel" onClick={() => void cancel(run.id)}><Square size={12} /></button>
                : <button className="icon-button danger" aria-label={`Delete ${run.label}`} title="Delete" onClick={() => void remove(run.id)}><Trash2 size={12} /></button>}
            </li>;
          })}</ul>
        </section>
      </div>
    </div>
  </>;
}
