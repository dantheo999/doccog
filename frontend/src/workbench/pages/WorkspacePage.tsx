import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { CheckCheck, Download, FileText, FolderOpen, LoaderCircle, Save, Sparkles, Upload, Wand2, X } from "lucide-react";
import { message, workbench } from "../api";
import type { Collection, SchemaDraft, SchemaSuggestion, SchemaSummary, SolutionInfo, StoredDocument } from "../types";
import { useWorkspace } from "../workspace";
import SchemaEditor, { schemaErrors } from "../components/SchemaEditor";
import { SuggestWithPicker, SuggestionsSection } from "../components/SchemaSuggestions";
import type { SuggestJob } from "../components/SchemaSuggestions";
import { settingsErrors } from "../components/SettingsForm";
import {
  MAX_GUIDANCE, SCHEMA_DRAFT_KEY, SHARED_ID, loadSuggestionSettings, loadWorkspaceDraft, normaliseDraft, sameDraft, sharedSuggestion,
  storeSuggestionSettings, suggestionSpecs, suggestionsFor,
} from "../components/schemaSuggestionModel";
import { downloadJson } from "../../recipe";

const SUPPORTED = /\.(pdf|docx|txt|md)$/i;

interface Progress { total: number; done: number; failed: { path: string; error: string }[]; current: string }

export default function WorkspacePage() {
  const workspace = useWorkspace();
  const folderInput = useRef<HTMLInputElement>(null);
  const importInput = useRef<HTMLInputElement>(null);
  const [progress, setProgress] = useState<Progress | null>(null);
  const [collections, setCollections] = useState<(Omit<Collection, "documents"> & { documents: number })[]>([]);
  const [exemplars, setExemplars] = useState<string[]>([]);
  const [preview, setPreview] = useState<StoredDocument | null>(null);
  const [draft, setDraft] = useState<SchemaDraft | null>(() => loadWorkspaceDraft());
  /** The draft as last rendered, for suggestions that finish after the user changed it. */
  const draftRef = useRef(draft);
  const [notes, setNotes] = useState<string[]>([]);
  /** Where the notes above the editor came from (e.g. "Direct LLM's suggestion"), when not the shared suggestion just run. */
  const [notesSource, setNotesSource] = useState<string | null>(null);
  /** Bring the editor into view after a suggestion loads into it (it sits below the suggestion cards). */
  const schemaPanel = useRef<HTMLElement>(null);
  const [reveal, setReveal] = useState<{ count: number; focus: boolean }>({ count: 0, focus: false });
  const [guidance, setGuidance] = useState("");
  const [effort, setEffort] = useState("medium");
  const [busy, setBusy] = useState<"save" | "save-new" | null>(null);
  const [schemas, setSchemas] = useState<SchemaSummary[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [saved, setSaved] = useState<string | null>(null);
  // Suggestion methods: the shared one (default) plus any solutions that can suggest; each runs as its own job.
  const [solutions, setSolutions] = useState<SolutionInfo[]>([]);
  const [methods, setMethods] = useState<string[]>([SHARED_ID]);
  const [jobs, setJobs] = useState<Record<string, SuggestJob>>({});
  const [solutionSettings, setSolutionSettings] = useState<Record<string, Record<string, unknown>>>({});
  const [suggestions, setSuggestions] = useState<SchemaSuggestion[]>([]);
  const [compare, setCompare] = useState<string[]>([]);
  const listRequest = useRef(0);
  /** Latest preview / document-set request: a slower earlier response must not replace a later one. */
  const previewRequest = useRef(0);
  const collectionRequest = useRef(0);
  const collection = workspace.collection;

  useEffect(() => { workbench.collections().then(setCollections).catch(() => setCollections([])); }, [collection?.id]);
  const loadSchemas = useCallback(() => workbench.schemas().then(setSchemas).catch(() => setSchemas([])), []);
  useEffect(() => { void loadSchemas(); }, [loadSchemas]);
  useEffect(() => {
    if (!draft && workspace.schema) setDraft({ name: workspace.schema.name, description: workspace.schema.description, fields: workspace.schema.fields });
  }, [workspace.schema, draft]);
  useEffect(() => {
    draftRef.current = draft;
    try { if (draft) localStorage.setItem(SCHEMA_DRAFT_KEY, JSON.stringify(draft)); } catch { /* quota */ }
  }, [draft]);
  useEffect(() => {
    if (!reveal.count) return;
    schemaPanel.current?.scrollIntoView?.({ behavior: "smooth", block: "start" });
    if (reveal.focus) schemaPanel.current?.querySelector<HTMLElement>("#schema-heading")?.focus({ preventScroll: true });
  }, [reveal]);
  const showEditor = useCallback((focus: boolean) => setReveal(current => ({ count: current.count + 1, focus })), []);
  useEffect(() => {
    workbench.solutions().then(items => {
      setSolutions(items);
      setSolutionSettings(Object.fromEntries(items.filter(item => item.schema_suggestion?.supported)
        .map(item => [item.id, loadSuggestionSettings(item)])));
    }).catch(() => setSolutions([]));
  }, []);
  const loadSuggestions = useCallback(async () => {
    const request = ++listRequest.current;
    try {
      const items = await workbench.schemaSuggestions();
      if (request === listRequest.current) setSuggestions(items);
    } catch { /* older backend: keep what this page has seen */ }
  }, []);
  useEffect(() => { void loadSuggestions(); }, [loadSuggestions]);

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
    if (!id) return;
    const request = ++collectionRequest.current;
    try {
      const next = await workbench.collection(id);
      if (request !== collectionRequest.current) return;
      workspace.setCollection(next);
      setExemplars([]);
    } catch (failure) { if (request === collectionRequest.current) setError(message(failure)); }
  }

  async function openPreview(id: string) {
    const request = ++previewRequest.current;
    try {
      const next = await workbench.document(id);
      if (request === previewRequest.current) setPreview(next);
    } catch (failure) { if (request === previewRequest.current) setError(message(failure)); }
  }

  const capable = useMemo(() => new Set(solutions.filter(item => item.schema_suggestion?.supported && item.status.ready)
    .map(item => item.id)), [solutions]);
  const chosenMethods = methods.filter(id => id === SHARED_ID || capable.has(id));
  const runnable = chosenMethods.filter(id => jobs[id]?.status !== "running");
  const suggesting = chosenMethods.some(id => jobs[id]?.status === "running");
  // Same checks as the Solution page panel: the backend would otherwise silently fall back to defaults.
  const suggestionErrors: Record<string, string[]> = Object.fromEntries(solutions
    .filter(item => chosenMethods.includes(item.id))
    .map(item => [item.id, settingsErrors(suggestionSpecs(item), solutionSettings[item.id] ?? {})] as const)
    .filter(([, problems]) => problems.length > 0));
  const invalidMethods = solutions.filter(item => suggestionErrors[item.id]).map(item => item.name);

  function upsertSuggestion(suggestion: SchemaSuggestion) {
    listRequest.current++; // a list request already in flight predates this suggestion
    setSuggestions(current => [suggestion, ...current.filter(item => item.id !== suggestion.id)]);
  }

  /** One request per chosen method, in parallel; each has its own loading and error state. */
  async function suggest() {
    if (!exemplars.length || !runnable.length || invalidMethods.length) return;
    const documentIds = [...exemplars];
    const startDraft = draft;
    const text = guidance;
    const reasoning = effort;
    const batch = [...runnable];
    const started = Date.now();
    setError(null);
    if (batch.length > 1) setCompare([]);
    setJobs(current => ({ ...current, ...Object.fromEntries(batch.map(id => [id, { status: "running", started } satisfies SuggestJob])) }));
    await Promise.all(batch.map(async id => {
      try {
        let suggestionId: string | undefined;
        let fields: number;
        let kept = false;
        if (id === SHARED_ID) {
          const result = await workbench.suggestSchema(documentIds, text, reasoning);
          // Load it only if the user has not changed the draft meanwhile (e.g. "Use as draft" on a faster method).
          kept = !sameDraft(draftRef.current, startDraft);
          if (!kept) {
            setDraft(result.draft);
            setNotes(result.notes);
            setNotesSource(null);
            setSaved(null);
            if (batch.length === 1) showEditor(false);
          }
          if (result.id) upsertSuggestion(sharedSuggestion({ ...result, id: result.id }, documentIds, text, reasoning));
          suggestionId = result.id;
          fields = result.draft.fields.length;
        } else {
          const result = await workbench.suggestSchemaWith(id, documentIds, text, solutionSettings[id] ?? {});
          upsertSuggestion(result);
          suggestionId = result.id;
          fields = result.draft.fields.length;
        }
        const added = suggestionId;
        if (batch.length > 1 && added) setCompare(current => current.includes(added) ? current : [...current, added]);
        setJobs(current => ({ ...current, [id]: { status: "done", started, finished: Date.now(), fields, ...(kept ? { kept } : {}) } }));
      } catch (failure) {
        setJobs(current => ({ ...current, [id]: { status: "error", started, finished: Date.now(), error: message(failure) } }));
      }
      void loadSuggestions();
    }));
  }

  function applySuggestion(suggestion: SchemaSuggestion) {
    const kept = suggestions.some(item => sameDraft(item.draft, draft));
    if (dirty && !kept && !sameDraft(draft, suggestion.draft)
      && !window.confirm("Replace the current draft with this suggestion? Its unsaved changes will be lost.")) return;
    setDraft(suggestion.draft);
    setNotes(suggestion.notes ?? []);
    setNotesSource(suggestion.solution === SHARED_ID ? "the shared suggestion" : `${suggestion.solution_name}'s suggestion`);
    setSaved(null);
    showEditor(true);
  }

  async function deleteSuggestion(suggestion: SchemaSuggestion) {
    if (!window.confirm(`Delete this ${suggestion.solution_name} suggestion?`)) return;
    try {
      await workbench.deleteSchemaSuggestion(suggestion.id);
      listRequest.current++; // a list request already in flight may still include it
      setSuggestions(current => current.filter(item => item.id !== suggestion.id));
      setCompare(current => current.filter(id => id !== suggestion.id));
    } catch (failure) { setError(message(failure)); }
  }

  /** Save the draft as the next version of the agreed schema, or (asNew) as a separate schema. */
  async function save(asNew = false) {
    if (!draft) return;
    setBusy(asNew ? "save-new" : "save");
    setError(null);
    try {
      const exemplarNames = collection?.documents.filter(item => exemplars.includes(item.id)).map(item => item.name);
      const schema = await workbench.saveSchema(draft, asNew ? null : workspace.schema?.id ?? null,
        exemplarNames?.length ? exemplarNames : undefined);
      workspace.setSchema(schema);
      // The backend tidies the draft (e.g. trims the name); keep the editor in step so it is not "unsaved".
      setDraft({ name: schema.name, description: schema.description, fields: schema.fields });
      setSaved(`Saved “${schema.name}” as version ${schema.version}. All solutions now extract against it.`);
      void loadSchemas();
    } catch (failure) { setError(message(failure)); }
    finally { setBusy(null); }
  }

  /** Make a saved schema the agreed one and load it into the editor. */
  async function pickSchema(id: string) {
    if (!id || id === workspace.schema?.id) return;
    if (dirty && !window.confirm("Load this saved schema? Unsaved changes to the current draft will be lost.")) return;
    setError(null);
    try {
      const schema = await workbench.schema(id);
      workspace.setSchema(schema);
      setDraft({ name: schema.name, description: schema.description, fields: schema.fields });
      setNotes([]);
      setNotesSource(null);
      setSaved(null);
    } catch (failure) { setError(message(failure)); }
  }

  async function importSchema(file: File | undefined) {
    if (!file) return;
    try {
      const value = JSON.parse(await file.text());
      if (!value || typeof value !== "object" || !Array.isArray(value.fields)) throw new Error("it has no \"fields\" list.");
      setDraft(normaliseDraft(value));
      setNotes([]);
      setNotesSource(null);
    } catch (failure) { setError(`Could not read the schema file: ${message(failure)}`); }
  }

  const errors = useMemo(() => draft ? schemaErrors(draft) : [], [draft]);
  const dirty = !!draft && (!workspace.schema || JSON.stringify({ name: draft.name, description: draft.description, fields: draft.fields })
    !== JSON.stringify({ name: workspace.schema.name, description: workspace.schema.description, fields: workspace.schema.fields }));
  const processing = !!progress && progress.done < progress.total;
  const schemaOptions = useMemo(() => workspace.schema && !schemas.some(item => item.id === workspace.schema!.id)
    ? [{ id: workspace.schema.id, name: workspace.schema.name, version: workspace.schema.version,
      created_at: workspace.schema.created_at, fields: workspace.schema.fields.length }, ...schemas] : schemas,
  [schemas, workspace.schema]);
  const documentIds = useMemo(() => collection?.documents.map(item => item.id) ?? [], [collection]);
  const visibleSuggestions = useMemo(() => suggestionsFor(suggestions, documentIds), [suggestions, documentIds]);

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
        {(collections.length > 1 || (!collection && collections.length > 0)) && <label className="inline-select"><span>Document set</span>
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

      <div className="workspace-main">
        <section className="panel workspace-suggest" aria-labelledby="suggest-heading">
          <div className="panel-title"><h2 id="suggest-heading">Suggest a schema</h2>
            <span className="microcopy">{exemplars.length}/5 exemplars ticked</span></div>
          <div className="suggest-box">
            <label><span className="field-label">Guidance for the suggestion (optional)</span>
              <textarea rows={2} maxLength={MAX_GUIDANCE} value={guidance} onChange={event => setGuidance(event.target.value)}
                placeholder="e.g. Focus on governance: attendees and attendance, decisions, actions with owners and due dates." /></label>
            <SuggestWithPicker solutions={solutions} selected={methods} onSelectedChange={setMethods} jobs={jobs}
              effort={effort} onEffortChange={setEffort} settings={solutionSettings}
              onSettingsChange={(id, values) => {
                setSolutionSettings(current => ({ ...current, [id]: values }));
                storeSuggestionSettings(id, values);
              }} errors={suggestionErrors} onShowEditor={() => showEditor(true)} />
            <div className="suggest-actions">
              <button className="primary" onClick={() => void suggest()} disabled={!exemplars.length || !runnable.length || invalidMethods.length > 0}>
                {suggesting ? <LoaderCircle size={15} className="spin" /> : <Sparkles size={15} />}
                {suggesting && !runnable.length ? "Analysing exemplars…"
                  : `Suggest from ${exemplars.length || "selected"} exemplar${exemplars.length === 1 ? "" : "s"}${runnable.length > 1 ? ` · ${runnable.length} methods` : ""}`}</button>
              <button className="secondary" onClick={() => void workbench.starterSchema().then(value => { setDraft(value); setNotes([]); setNotesSource(null); })}>
                <Wand2 size={14} />Meeting-minutes starter</button>
            </div>
            {!exemplars.length ? <p className="microcopy">Tick up to five exemplar documents on the left. The model reads their full text and proposes
              fields, types and nested object arrays for documents of this type.</p>
              : !chosenMethods.length && <p className="microcopy">Choose at least one method to suggest with.</p>}
            {invalidMethods.length > 0 && <p className="microcopy suggest-invalid" role="status">Fix the suggestion settings
              of {invalidMethods.join(" and ")}, or untick {invalidMethods.length === 1 ? "it" : "them"}, to suggest.</p>}
            {chosenMethods.length > 1 && <p className="microcopy">The shared suggestion loads into the editor unless you change the draft
              while it runs; every result is kept under Suggestions below, ticked for comparison.</p>}
          </div>
          {collection && <SuggestionsSection suggestions={visibleSuggestions} solutions={solutions} draft={draft} compare={compare}
            onCompareChange={setCompare} onUse={applySuggestion} onDelete={suggestion => void deleteSuggestion(suggestion)} />}
        </section>

        <section className="panel workspace-schema" aria-labelledby="schema-heading" ref={schemaPanel}>
          <div className="panel-title"><h2 id="schema-heading" tabIndex={-1}>Agreed schema</h2>
            {schemaOptions.length > 0 && <select aria-label="Saved schema" title="Choose a saved schema"
              value={workspace.schema?.id ?? ""} onChange={event => void pickSchema(event.target.value)}>
              {!workspace.schema && <option value="">Choose a saved schema…</option>}
              {schemaOptions.map(item => <option key={item.id} value={item.id}>{item.name} · v{item.version} · {item.fields} fields</option>)}
            </select>}
            {workspace.schema && <span className="pill success">v{workspace.schema.version} · {workspace.schema.fields.length} fields</span>}
            <span className="spacer" />
            <button className="icon-button" title="Import schema JSON" aria-label="Import schema JSON" onClick={() => importInput.current?.click()}><Upload size={14} /></button>
            <input type="file" hidden accept="application/json,.json" ref={importInput} aria-label="Import schema file"
              onChange={event => { void importSchema(event.target.files?.[0]); event.target.value = ""; }} />
            <button className="icon-button" title="Export schema JSON" aria-label="Export schema JSON" disabled={!draft}
              onClick={() => draft && downloadJson(draft, `${draft.name || "schema"}.schema.json`)}><Download size={14} /></button>
          </div>
          {notes.length > 0 && <div className="suggest-notes" role="region" aria-label="Model notes"><span className="eyebrow">MODEL NOTES</span>
            {notesSource && <span className="microcopy notes-source"> · from {notesSource}</span>}
            <ul>{notes.map(note => <li key={note}>{note}</li>)}</ul></div>}
          {draft ? <>
            <SchemaEditor draft={draft} onChange={next => { setDraft(next); setSaved(null); }} />
            {errors.length > 0 && <ul className="schema-errors">{errors.slice(0, 6).map(item => <li key={item}>{item}</li>)}</ul>}
            <div className="save-row">
              <button className="primary" onClick={() => void save()} disabled={!!errors.length || busy !== null || !dirty}>
                {busy === "save" ? <LoaderCircle size={15} className="spin" /> : <Save size={15} />}
                {workspace.schema ? (dirty ? "Save as new version" : "Saved") : "Save as agreed schema"}</button>
              {workspace.schema && <button className="secondary" onClick={() => void save(true)} disabled={!!errors.length || busy !== null}
                title="Save the draft as a separate schema (for example for another document type), not as a version of this one">
                {busy === "save-new" ? <LoaderCircle size={15} className="spin" /> : <Save size={15} />}Save as new schema</button>}
              {saved && <span className="microcopy success">{saved}</span>}
              {dirty && workspace.schema && <span className="microcopy">Unsaved changes; runs use the saved version.</span>}
            </div>
          </> : <div className="wb-empty small"><p>No schema draft. Suggest one from exemplars, start from the meeting-minutes starter or import JSON.</p></div>}
        </section>
      </div>
    </div>

    {preview && <div className="preview-drawer panel" role="dialog" aria-label={`Preview of ${preview.name}`}>
      <div className="panel-title"><h2>{preview.name}</h2><span className="microcopy">{preview.characters.toLocaleString()} characters
        {preview.pages !== null && ` · ${preview.pages} pages`} · extracted text</span><span className="spacer" />
        <button className="icon-button" aria-label="Close preview" onClick={() => setPreview(null)}><X size={14} /></button></div>
      <pre className="preview-text">{preview.text}</pre>
    </div>}
  </div>;
}
