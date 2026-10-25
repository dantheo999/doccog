import { useRef, useState } from "react";
import { Download, ExternalLink, FileUp, LoaderCircle, Wand2 } from "lucide-react";
import { message, workbench } from "../api";
import { navigate } from "../router";
import type { SolutionInfo } from "../types";
import { useWorkspace } from "../workspace";
import { downloadJson } from "../../recipe";

/** Unstract runs in its own packaged platform; this panel bridges schema -> prompts and results -> runs. */
export default function UnstractPanel({ solution }: { solution: SolutionInfo }) {
  const workspace = useWorkspace();
  const fileInput = useRef<HTMLInputElement>(null);
  const [prompts, setPrompts] = useState<Record<string, string> | null>(null);
  const [busy, setBusy] = useState<"prompts" | "import" | null>(null);
  const [label, setLabel] = useState("");
  const [status, setStatus] = useState<string | null>(null);
  const [warnings, setWarnings] = useState<string[]>([]);
  const [error, setError] = useState<string | null>(null);

  const projectBlock = prompts ? Object.entries(prompts).find(([title]) => title.startsWith("Prompt Studio project")) : undefined;
  let project: unknown = null;
  try { project = projectBlock ? JSON.parse(projectBlock[1]) : null; } catch { project = null; }

  async function loadPrompts() {
    if (!workspace.schema) return;
    setBusy("prompts");
    setError(null);
    try { setPrompts(await workbench.preview(solution.id, workspace.schema.id, {})); }
    catch (failure) { setError(message(failure)); }
    finally { setBusy(null); }
  }

  async function importResults(files: FileList | null) {
    if (!files?.length || !workspace.schema || !workspace.collection) return;
    setBusy("import");
    setError(null);
    setStatus(null);
    setWarnings([]);
    try {
      const payloads = await Promise.all(Array.from(files).map(async file => ({ name: file.name, content: JSON.parse(await file.text()) })));
      const run = await workbench.importUnstract({
        schema_id: workspace.schema.id, label, document_ids: workspace.collection.documents.map(item => item.id), files: payloads,
      });
      setStatus(`Imported ${run.documents_done} of ${run.documents_total} matched documents.`);
      setWarnings(run.warnings ?? []);
      navigate(solution.route, { run: run.id });
    } catch (failure) { setError(message(failure)); }
    finally { setBusy(null); }
  }

  return <section className="panel unstract-panel" aria-labelledby="unstract-heading">
    <div className="panel-title"><h2 id="unstract-heading">Use Unstract's packaged platform</h2></div>
    <ol className="unstract-steps">
      <li><b>Open Unstract</b> (log in as <code>unstract</code> / <code>unstract</code> on a fresh install) and add its adapters:
        LLM (OpenAI Compatible, pointed at your Azure <code>/openai/v1/</code> URL), embedding, vector DB and text extractor.
        {solution.external_url && <a className="text-button" href={solution.external_url} target="_blank" rel="noreferrer">
          {solution.external_url}<ExternalLink size={12} /></a>}</li>
      <li><b>Create a Prompt Studio project</b> from the agreed schema: one prompt per field, keyed by field name.
        Upload your exemplar documents there to tune the prompts.
        <div className="inline-actions">
          <button className="secondary" onClick={() => void loadPrompts()} disabled={!workspace.schema || busy !== null}>
            {busy === "prompts" ? <LoaderCircle size={14} className="spin" /> : <Wand2 size={14} />}Generate prompts from schema</button>
          {project !== null && <button className="secondary" onClick={() => downloadJson(project,
            `${(workspace.schema?.name ?? "schema").replace(/[^\w.-]+/g, "_")}.unstract-project.json`)}>
            <Download size={14} />Download Prompt Studio project (.json)</button>}
        </div>
        {project !== null && <p className="microcopy">In Unstract: Prompt Studio → Import Project → choose this file and your adapters,
          run it on an exemplar, then Deploy as API.</p>}
        {prompts && Object.entries(prompts).map(([title, text]) => <details key={title}><summary>{title}</summary><pre>{text}</pre></details>)}</li>
      <li><b>Bring results back for comparison</b>, either:
        <ul>
          <li>deploy the project as an <b>API deployment</b>, set <code>UNSTRACT_API_URL</code> and <code>UNSTRACT_API_KEY</code> for the
            backend, and use <b>New run</b>: each document's original file is sent to Unstract and its JSON output collated, or</li>
          <li>import result JSON files downloaded from Unstract (API responses or per-file outputs); they are matched to workspace
            documents by file name. Unstract serves each API result only once, so keep the downloaded file.</li>
        </ul>
        <div className="inline-actions">
          <input className="compact-input" value={label} onChange={event => setLabel(event.target.value)} placeholder="Run label (optional)" aria-label="Imported run label" />
          <button className="secondary" onClick={() => fileInput.current?.click()} disabled={!workspace.schema || !workspace.collection || busy !== null}>
            {busy === "import" ? <LoaderCircle size={14} className="spin" /> : <FileUp size={14} />}Import results JSON</button>
          <input ref={fileInput} type="file" hidden multiple accept="application/json,.json" aria-label="Unstract results files"
            onChange={event => { void importResults(event.target.files); event.target.value = ""; }} />
        </div></li>
    </ol>
    {status && <p className="microcopy success">{status}</p>}
    {warnings.length > 0 && <ul className="run-notes">{warnings.map(item => <li key={item}>{item}</li>)}</ul>}
    {error && <div className="notice error" role="alert"><span>{error}</span></div>}
  </section>;
}
