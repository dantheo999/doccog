import { useEffect, useMemo, useRef, useState } from "react";
import { CheckCheck, Download, FileText, FolderOpen, LoaderCircle, Save, Sparkles, Upload, Wand2, X } from "lucide-react";
import { message, workbench } from "../api";
import type { Collection, SchemaDraft, StoredDocument } from "../types";
import { useWorkspace } from "../workspace";
import SchemaEditor, { schemaErrors } from "../components/SchemaEditor";
import { downloadJson } from "../../recipe";

const SUPPORTED = /\.(pdf|docx|txt|md)$/i;
const DRAFT_KEY = "doc-cognition.schema-draft.v1";

interface Progress { total: number; done: number; failed: { path: string; error: string }[]; current: string }

function loadDraft(): SchemaDraft | null {
  try { return JSON.parse(localStorage.getItem(DRAFT_KEY) ?? "null"); } catch { return null; }
}

export default function WorkspacePage() {
  const workspace = useWorkspace();
  const folderInput = useRef<HTMLInputElement>(null);
  const importInput = useRef<HTMLInputElement>(null);
  const [progress, setProgress] = useState<Progress | null>(null);
  const [collections, setCollections] = useState<(Omit<Collection, "documents"> & { documents: number })[]>([]);
  const [exemplars, setExemplars] = useState<string[]>([]);
  const [preview, setPreview] = useState<StoredDocument | null>(null);
  const [draft, setDraft] = useState<SchemaDraft | null>(() => loadDraft());
  const [notes, setNotes] = useState<string[]>([]);
  const [guidance, setGuidance] = useState("");
  const [effort, setEffort] = useState("medium");
  const [busy, setBusy] = useState<"suggest" | "save" | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [saved, setSaved] = useState<string | null>(null);
  const collection = workspace.collection;

  useEffect(() => { workbench.collections().then(setCollections).catch(() => setCollections([])); }, [collection?.id]);
  useEffect(() => {
    if (!draft && workspace.schema) setDraft({ name: workspace.schema.name, description: workspace.schema.description, fields: workspace.schema.fields });
  }, [workspace.schema, draft]);
  useEffect(() => {
    try { if (draft) localStorage.setItem(DRAFT_KEY, JSON.stringify(draft)); } catch { /* quota */ }
  }, [draft]);

  async function processFolder(files: FileList | null) {
    if (!files?.length) return;
    const all = Array.from(files);
    const accepted = all.filter(file => SUPPORTED.test(file.name))
      .map(file => ({ file, path: file.webkitRelativePath || file.name }))
      .sort((a, b) => a.path.localeCompare(b.path));
    if (!accepted.length) { setError("No PDF, DOCX, TXT or Markdown files in that folder."); return; }
    setError(null);
    setExemplars([]);
    const name = all[0]?.webkitRelativePath?.split("/")[0] || "Selected documents";
    const state: Progress = { total: accepted.length, done: 0, failed: [], current: "" };
    setProgress({ ...state });
    try {
      let created = await workbench.createCollection(name);
      // A few uploads in parallel: parsing happens server-side, once per file content.
      const queue = [...accepted];
      await Promise.all(Array.from({ length: 3 }, async () => {
        for (let item = queue.shift(); item; item = queue.shift()) {
          setProgress({ ...state, current: item.path });
          try {
            if (item.file.size > 20 * 1024 * 1024) throw new Error("exceeds the 20 MiB upload limit");
            await workbench.uploadToCollection(item.file, created.id, item.path);
          } catch (failure) { state.failed.push({ path: item.path, error: message(failure) }); }
          state.done++;
          setProgress({ ...state });
        }
      }));
      created = await workbench.collection(created.id);
      workspace.setCollection(created);
    } catch (failure) { setError(message(failure)); }
    finally { setProgress(current => current && { ...current, current: "" }); }
  }

  async function switchCollection(id: string) {
    try { workspace.setCollection(await workbench.collection(id)); setExemplars([]); }
    catch (failure) { setError(message(failure)); }
  }

  async function openPreview(id: string) {
    try { setPreview(await workbench.document(id)); } catch (failure) { setError(message(failure)); }
  }

  async function suggest() {
    setBusy("suggest");
    setError(null);
    try {
      const result = await workbench.suggestSchema(exemplars, guidance, effort);
      setDraft(result.draft);
      setNotes(result.notes);
      setSaved(null);
    } catch (failure) { setError(message(failure)); }
    finally { setBusy(null); }
  }

  async function save() {
    if (!draft) return;
    setBusy("save");
    setError(null);
    try {
      const exemplarNames = collection?.documents.filter(item => exemplars.includes(item.id)).map(item => item.name);
      const schema = await workbench.saveSchema(draft, workspace.schema?.id ?? null, exemplarNames?.length ? exemplarNames : undefined);
      workspace.setSchema(schema);
      setSaved(`Saved “${schema.name}” as version ${schema.version}. All solutions now extract against it.`);
    } catch (failure) { setError(message(failure)); }
    finally { setBusy(null); }
  }

  async function importSchema(file: File | undefined) {
    if (!file) return;
    try {
      const value = JSON.parse(await file.text());
      setDraft({ name: value.name ?? "Imported schema", description: value.description ?? "", fields: value.fields ?? [] });
      setNotes([]);
    } catch (failure) { setError(`Could not read the schema file: ${message(failure)}`); }
  }

  const errors = useMemo(() => draft ? schemaErrors(draft) : [], [draft]);
  const dirty = !!draft && (!workspace.schema || JSON.stringify({ name: draft.name, description: draft.description, fields: draft.fields })
    !== JSON.stringify({ name: workspace.schema.name, description: workspace.schema.description, fields: workspace.schema.fields }));
  const processing = !!progress && progress.done < progress.total;

  return <div className="wb-page workspace-page">
    <div className="page-heading"><div><span className="eyebrow">SHARED BY EVERY SOLUTION</span><h1>Workspace</h1>
      <p>Documents are processed once (text extraction, page offsets) when you choose a folder. The agreed schema is the
        contract every solution extracts against, so runs are comparable.</p></div></div>
    {error && <div className="notice error" role="alert"><span>{error}</span><button aria-label="Dismiss error" onClick={() => setError(null)}><X size={15} /></button></div>}

    <div className="workspace-grid">
      <section className="panel workspace-docs" aria-labelledby="docs-heading">
        <div className="panel-title"><h2 id="docs-heading">Document set</h2>
          <button className="secondary" onClick={() => folderInput.current?.click()} disabled={processing}><FolderOpen size={15} />Choose folder</button>
          <input type="file" hidden multiple aria-label="Choose workspace folder" ref={node => {
            folderInput.current = node;
            node?.setAttribute("webkitdirectory", "");
            node?.setAttribute("directory", "");
          }} onChange={event => { void processFolder(event.target.files); event.target.value = ""; }} />
        </div>
        <p className="microcopy">Try <b>samples/</b> for a quick check or <b>documents/</b> (20 meeting minutes) for full runs.</p>
        {collections.length > 1 && <label className="inline-select"><span>Document set</span>
          <select value={collection?.id ?? ""} onChange={event => void switchCollection(event.target.value)}>
            {!collection && <option value="">Choose…</option>}
            {collections.map(item => <option key={item.id} value={item.id}>{item.name} · {item.documents} docs · {new Date(item.created_at).toLocaleDateString()}</option>)}
          </select></label>}
        {progress && <div className="process-progress" aria-live="polite">
          <div className="progress-bar"><span style={{ width: `${(progress.done / progress.total) * 100}%` }} /></div>
          <span>{processing ? <><LoaderCircle size={13} className="spin" />Processing {progress.done}/{progress.total}{progress.current && ` · ${progress.current}`}</>
            : <><CheckCheck size={13} />Processed {progress.done - progress.failed.length} of {progress.total} documents</>}</span>
          {progress.failed.map(item => <span key={item.path} className="failed">{item.path}: {item.error}</span>)}
        </div>}
        {collection ? <>
          <div className="doc-list-heading"><span>{collection.name}</span><span>{exemplars.length}/5 exemplars</span></div>
          <ul className="workspace-doc-list">
            {collection.documents.map(item => <li key={item.id}>
              <input type="checkbox" aria-label={`Use ${item.path} as exemplar`} checked={exemplars.includes(item.id)}
                disabled={!exemplars.includes(item.id) && exemplars.length >= 5}
                onChange={() => setExemplars(current => current.includes(item.id) ? current.filter(id => id !== item.id) : [...current, item.id])} />
              <button className={preview?.id === item.id ? "active" : ""} onClick={() => void openPreview(item.id)} title={item.path}>
                <FileText size={14} /><span><strong>{item.name}</strong>
                  <small>{item.characters.toLocaleString()} chars{item.pages !== null && ` · ${item.pages} pages`}</small></span></button>
            </li>)}
          </ul>
        </> : <div className="wb-empty small"><FolderOpen size={26} /><p>No document set yet.</p></div>}
      </section>

      <section className="panel workspace-schema" aria-labelledby="schema-heading">
        <div className="panel-title"><h2 id="schema-heading">Agreed schema</h2>
          {workspace.schema && <span className="pill success">v{workspace.schema.version} · {workspace.schema.fields.length} fields</span>}
          <span className="spacer" />
          <button className="icon-button" title="Import schema JSON" aria-label="Import schema JSON" onClick={() => importInput.current?.click()}><Upload size={14} /></button>
          <input type="file" hidden accept="application/json,.json" ref={importInput} aria-label="Import schema file"
            onChange={event => { void importSchema(event.target.files?.[0]); event.target.value = ""; }} />
          <button className="icon-button" title="Export schema JSON" aria-label="Export schema JSON" disabled={!draft}
            onClick={() => draft && downloadJson(draft, `${draft.name || "schema"}.schema.json`)}><Download size={14} /></button>
        </div>
        <div className="suggest-box">
          <label><span className="field-label">Guidance for the suggestion (optional)</span>
            <textarea rows={2} value={guidance} onChange={event => setGuidance(event.target.value)}
              placeholder="e.g. Focus on governance: attendees and attendance, decisions, actions with owners and due dates." /></label>
          <div className="suggest-actions">
            <button className="primary" onClick={() => void suggest()} disabled={!exemplars.length || busy !== null}>
              {busy === "suggest" ? <LoaderCircle size={15} className="spin" /> : <Sparkles size={15} />}
              {busy === "suggest" ? "Analysing exemplars…" : `Suggest from ${exemplars.length || "selected"} exemplar${exemplars.length === 1 ? "" : "s"}`}</button>
            <label className="inline-select"><span>Reasoning</span><select value={effort} onChange={event => setEffort(event.target.value)}>
              {["low", "medium", "high"].map(item => <option key={item}>{item}</option>)}</select></label>
            <button className="secondary" onClick={() => void workbench.starterSchema().then(value => { setDraft(value); setNotes([]); })}>
              <Wand2 size={14} />Meeting-minutes starter</button>
          </div>
          {!exemplars.length && <p className="microcopy">Tick up to five exemplar documents on the left. The model reads their full text and proposes
            fields, types and nested object arrays for documents of this type.</p>}
        </div>
        {notes.length > 0 && <div className="suggest-notes"><span className="eyebrow">MODEL NOTES</span><ul>{notes.map(note => <li key={note}>{note}</li>)}</ul></div>}
        {draft ? <>
          <SchemaEditor draft={draft} onChange={next => { setDraft(next); setSaved(null); }} />
          {errors.length > 0 && <ul className="schema-errors">{errors.slice(0, 6).map(item => <li key={item}>{item}</li>)}</ul>}
          <div className="save-row">
            <button className="primary" onClick={() => void save()} disabled={!!errors.length || busy !== null || !dirty}>
              {busy === "save" ? <LoaderCircle size={15} className="spin" /> : <Save size={15} />}
              {workspace.schema ? (dirty ? "Save as new version" : "Saved") : "Save as agreed schema"}</button>
            {saved && <span className="microcopy success">{saved}</span>}
            {dirty && workspace.schema && <span className="microcopy">Unsaved changes; runs use the saved version.</span>}
          </div>
        </> : <div className="wb-empty small"><p>No schema draft. Suggest one from exemplars, start from the meeting-minutes starter or import JSON.</p></div>}
      </section>
    </div>

    {preview && <div className="preview-drawer panel" role="dialog" aria-label={`Preview of ${preview.name}`}>
      <div className="panel-title"><h2>{preview.name}</h2><span className="microcopy">{preview.characters.toLocaleString()} characters
        {preview.pages !== null && ` · ${preview.pages} pages`} · extracted text</span><span className="spacer" />
        <button className="icon-button" aria-label="Close preview" onClick={() => setPreview(null)}><X size={14} /></button></div>
      <pre className="preview-text">{preview.text}</pre>
    </div>}
  </div>;
}
