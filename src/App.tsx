import { useEffect, useMemo, useRef, useState } from "react";
import type { CSSProperties } from "react";
import { AlertCircle, ArrowRight, Braces, CheckCheck, ChevronRight, CircleDot, Download,
  FileSearch, FileText, Layers3, LoaderCircle, ScanText, Settings, Sparkles, X } from "lucide-react";
import { api } from "./api";
import { classColor } from "./highlights";
import { downloadJson, errorMessage, loadRecipe, saveRecipe } from "./recipe";
import type { DocumentPreview, ExtractionResult, LocalDocument, ModelConfig, OptimizationProposal, Recipe } from "./types";
import SourceViewer from "./components/SourceViewer";
import RecipePanel from "./components/RecipePanel";
import ExtractionSettingsDialog from "./components/ExtractionSettingsDialog";
import { DEFAULT_EXTRACTION_OPTIONS, loadExtractionOptions, saveExtractionOptions } from "./extractionOptions";
import type { ExtractionOptions } from "./extractionOptions";
import DocumentLibrary from "./components/DocumentLibrary";
import OptimizationReview from "./components/OptimizationReview";
import { assertOptimizationBase } from "./optimization";

const supported = /\.(pdf|docx|txt|md)$/i;

function initialRecipe(): { recipe: Recipe | null; error: string | null } {
  try { return { recipe: loadRecipe(), error: null }; }
  catch (error) { return { recipe: null, error: `Could not restore the saved recipe: ${errorMessage(error)}` }; }
}

function initialExtractionOptions(): { options: ExtractionOptions; warning: string | null } {
  try { return { options: loadExtractionOptions(), warning: null }; }
  catch (error) {
    return { options: { ...DEFAULT_EXTRACTION_OPTIONS },
      warning: `Could not restore extraction settings: ${errorMessage(error)}. Using defaults; save settings to replace the invalid value.` };
  }
}

export default function App() {
  const [initial] = useState(initialRecipe);
  const [initialOptions] = useState(initialExtractionOptions);
  const [options, setOptions] = useState<ExtractionOptions>(initialOptions.options);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [recipe, setRecipe] = useState<Recipe | null>(initial.recipe);
  const [goal, setGoal] = useState(initial.recipe?.goal ?? "");
  const [documents, setDocuments] = useState<LocalDocument[]>([]);
  const [folder, setFolder] = useState("");
  const [ignored, setIgnored] = useState(0);
  const [query, setQuery] = useState("");
  const [selectedPath, setSelectedPath] = useState("");
  const [exemplarPaths, setExemplarPaths] = useState<string[]>([]);
  const [preview, setPreview] = useState<DocumentPreview | null>(null);
  const [result, setResult] = useState<ExtractionResult | null>(null);
  const [appliedRecipe, setAppliedRecipe] = useState<Recipe | null>(null);
  const [review, setReview] = useState<{ proposal: OptimizationProposal; original: Recipe } | null>(null);
  const [selectedExtraction, setSelectedExtraction] = useState<string | null>(null);
  const [config, setConfig] = useState<ModelConfig | null>(null);
  const [busy, setBusy] = useState<"preview" | "generate" | "extract" | "optimize" | null>(null);
  const [error, setError] = useState<string | null>(initial.error);
  const [warning, setWarning] = useState<string | null>(initialOptions.warning);
  const [raw, setRaw] = useState(false);
  const [classFilter, setClassFilter] = useState("all");
  const selectionToken = useRef(0);
  const settingsButton = useRef<HTMLButtonElement>(null);
  const optimizeButton = useRef<HTMLButtonElement>(null);
  const recipeRef = useRef(recipe);
  const modelBusy = busy === "generate" || busy === "extract" || busy === "optimize" || review !== null;

  useEffect(() => {
    let active = true;
    api.config().then(value => { if (active) setConfig(value); })
      .catch(error => { if (active) setError(errorMessage(error)); });
    return () => { active = false; };
  }, []);

  function applyRecipe(next: Recipe) {
    recipeRef.current = next;
    setRecipe(next);
    setError(null);
    setWarning(null);
    try { saveRecipe(next); }
    catch (error) { setWarning(`Recipe is active, but browser persistence failed: ${errorMessage(error)}. Export it to keep a copy.`); }
  }

  function applyOptions(next: ExtractionOptions) {
    setOptions(next);
    setWarning(null);
    try { saveExtractionOptions(next); }
    catch (error) { setWarning(`Extraction settings are active, but browser persistence failed: ${errorMessage(error)}. They will not survive a reload.`); }
  }

  function chooseFolder(files: FileList | null) {
    if (!files || modelBusy) return;
    const all = Array.from(files);
    const accepted = all.filter(file => supported.test(file.name))
      .map(file => ({ file, path: file.webkitRelativePath || file.name }))
      .sort((a, b) => a.path.localeCompare(b.path));
    selectionToken.current++;
    setDocuments(accepted);
    setIgnored(all.length - accepted.length);
    setFolder(all[0]?.webkitRelativePath?.split("/")[0] || "Selected documents");
    setPreview(null);
    setResult(null);
    setSelectedPath("");
    setExemplarPaths([]);
    setAppliedRecipe(null);
    setSelectedExtraction(null);
    setBusy(null);
    setQuery("");
    setError(accepted.length ? null : "No supported documents found. Choose a folder containing PDF, DOCX, TXT or Markdown.");
  }

  function toggleExemplar(path: string) {
    if (modelBusy) return;
    setExemplarPaths(current => current.includes(path) ? current.filter(item => item !== path)
      : current.length < 5 ? [...current, path] : current);
  }

  async function upload(document: LocalDocument) {
    if (document.file.size > 20 * 1024 * 1024) throw new Error("Document exceeds the 20 MiB upload limit.");
    return api.upload(document.file);
  }

  async function selectDocument(document: LocalDocument) {
    if (modelBusy) return;
    const token = ++selectionToken.current;
    setSelectedPath(document.path);
    setPreview(null);
    setResult(null);
    setAppliedRecipe(null);
    setSelectedExtraction(null);
    setClassFilter("all");
    setRaw(false);
    setError(null);
    setBusy("preview");
    try {
      const loaded = await upload(document);
      if (token === selectionToken.current) setPreview(loaded);
    } catch (error) { if (token === selectionToken.current) setError(errorMessage(error)); }
    finally { if (token === selectionToken.current) setBusy(null); }
  }

  async function generate() {
    if ((!preview && !exemplarPaths.length) || !goal.trim() || busy || review) return;
    const token = selectionToken.current;
    setBusy("generate");
    setError(null);
    try {
      const ids: string[] = [];
      if (exemplarPaths.length) {
        for (const path of exemplarPaths) {
          const document = documents.find(item => item.path === path);
          if (!document) throw new Error("An exemplar is no longer in this folder. Reselect its document.");
          const source = preview && selectedPath === path ? preview : await upload(document);
          ids.push(source.id);
        }
      } else if (preview) ids.push(preview.id);
      const next = await api.generate(ids, goal.trim());
      if (token === selectionToken.current) applyRecipe(next);
    } catch (error) { setError(errorMessage(error)); }
    finally { setBusy(null); }
  }

  async function extract() {
    if (!preview || !recipe || busy || review) return;
    const token = selectionToken.current;
    setBusy("extract");
    setError(null);
    try {
      const next = await api.extract(preview.id, recipe, options);
      if (token === selectionToken.current) {
        setResult(next);
        setAppliedRecipe(recipe);
        setSelectedExtraction(null);
        setClassFilter("all");
      }
    } catch (error) { setError(errorMessage(error)); }
    finally { setBusy(null); }
  }

  const matchingResult = !!(preview && result && recipe && appliedRecipe
    && result.document_id === preview.id && result.text === preview.text
    && result.recipe_id === recipe.id && result.recipe_version === recipe.version
    && appliedRecipe.id === recipe.id && appliedRecipe.version === recipe.version);

  async function optimize() {
    if (!matchingResult || !preview || !appliedRecipe || !result || busy || review) return;
    const token = selectionToken.current;
    setBusy("optimize");
    setError(null);
    try {
      const proposal = await api.optimize(preview.id, appliedRecipe, result);
      if (token !== selectionToken.current || !recipeRef.current) throw new Error("The document changed. Extract again before optimizing.");
      assertOptimizationBase(proposal, recipeRef.current);
      setReview({ proposal, original: appliedRecipe });
    } catch (error) { setError(errorMessage(error)); }
    finally { setBusy(null); }
  }

  function applyOptimization() {
    if (!review || !recipeRef.current || !matchingResult) throw new Error("The recipe or document changed. Discard this proposal and Extract again.");
    assertOptimizationBase(review.proposal, recipeRef.current);
    applyRecipe(review.proposal.candidate);
    setReview(null);
  }

  const filteredDocuments = documents.filter(document => document.path.toLowerCase().includes(query.toLowerCase()));
  const classes = useMemo(() => [...new Set(result?.extractions.map(item => item.extraction_class) ?? [])], [result]);
  const findings = useMemo(() => result?.extractions.filter(item =>
    classFilter === "all" || item.extraction_class === classFilter) ?? [], [result, classFilter]);
  const staleResult = result && recipe && (result.recipe_id !== recipe.id || result.recipe_version !== recipe.version);
  const exactCount = result?.extractions.filter(item => item.grounding === "exact").length ?? 0;

  return <div className="app-shell">
    <header className="app-header">
      <div className="brand"><div className="brand-mark"><ScanText size={25} /></div>
        <div><h1>Document Cognition<span>STUDIO</span></h1><p>From unstructured text to grounded intelligence</p></div>
      </div>
      <div className="header-actions">
        <div className="connection"><span className={`connection-dot ${config?.configured ? "online" : ""}`} />
          <div><span className="eyebrow">AZURE FOUNDRY</span><strong>{config?.configured ? config.deployment : "Not configured"}</strong></div>
        </div>
        <button ref={settingsButton} className="icon-button settings-cog" aria-label="Extraction settings"
          title="Extraction settings" onClick={() => setSettingsOpen(true)}><Settings size={20} /></button>
      </div>
    </header>
    <div className="workflow-bar">
      <span className={documents.length ? "complete" : "active"}><b>01</b> Choose documents</span><ChevronRight size={15} />
      <span className={recipe ? "complete" : documents.length ? "active" : ""}><b>02</b> Generate a recipe</span><ChevronRight size={15} />
      <span className={result ? "complete" : recipe ? "active" : ""}><b>03</b> Extract & explore</span>
      <span className="powered"><Layers3 size={14} />Powered by LangExtract</span>
    </div>
    <div className="notifications" aria-live="polite">
      {error && <div className="notice error" role="alert"><AlertCircle size={18} /><span>{error}</span>
        <button aria-label="Dismiss error" onClick={() => setError(null)}><X size={16} /></button></div>}
      {warning && <div className="notice warning"><AlertCircle size={18} /><span>{warning}</span>
        <button aria-label="Dismiss warning" onClick={() => setWarning(null)}><X size={16} /></button></div>}
      {config && !config.configured && <div className="notice setup"><CircleDot size={17} />
        <span>{config.error || "Connect your Azure deployment in the local .env file, then recreate the backend container. Document browsing and recipe import are available now."}</span></div>}
    </div>
    <main className="workspace">
      <DocumentLibrary documents={documents} filteredDocuments={filteredDocuments} folder={folder} ignored={ignored}
        query={query} selectedPath={selectedPath} exemplarPaths={exemplarPaths} busy={modelBusy} onFolder={chooseFolder}
        onQuery={setQuery} onPreview={document => void selectDocument(document)} onExemplar={toggleExemplar} />
      <div className="document-stage">
        <section className="source-panel panel">
          <div className="panel-header">
            <div className="section-icon teal"><FileText size={18} /></div>
            <div className="source-heading"><span className="eyebrow">SOURCE DOCUMENT</span><h2>{preview?.name || selectedPath.split("/").pop() || "Select a document"}</h2></div>
            <button className="primary extract-button" disabled={!preview || !recipe || !config?.configured || busy !== null || review !== null} onClick={() => void extract()}>
              {busy === "extract" ? <LoaderCircle className="spin" size={16} /> : <ArrowRight size={16} />}{busy === "extract" ? "Extracting…" : "Extract"}
            </button>
          </div>
          {preview && <div className="source-meta"><span>{preview.characters.toLocaleString()} characters</span>
            {preview.pages !== null && <span>{preview.pages} pages</span>}<span>Extracted text view</span>
            <span className="source-hint">Click a finding to trace its source</span></div>}
          {classes.length > 0 && <div className="class-legend">
            <button className={classFilter === "all" ? "active" : ""} onClick={() => setClassFilter("all")}>All classes</button>
            {classes.map(name => <button key={name} className={classFilter === name ? "active" : ""}
              onClick={() => setClassFilter(name)}><i style={{ backgroundColor: classColor(name) }} />{name}</button>)}
          </div>}
          {busy === "preview" ? <div className="empty source-empty"><LoaderCircle size={35} className="spin" /><h3>Reading your document…</h3><p>Preparing text for grounded extraction.</p></div>
            : preview ? <SourceViewer text={preview.text} extractions={findings} selected={selectedExtraction} onSelect={setSelectedExtraction} />
              : <div className="empty source-empty"><div className="empty-illustration"><FileSearch size={43} /></div>
                <h3>Every finding starts with a source</h3><p>Choose a document from the sidebar to preview its text.<br />Teach once, then extract across your collection.</p>
                <span className="empty-badge"><CheckCheck size={14} />Exact source spans · Inspectable JSON</span></div>}
        </section>
        <section className="results-panel panel">
          <div className="results-header"><div><span className="eyebrow">GROUNDED INTELLIGENCE</span><h2>
            {result ? `${result.extractions.length} ${result.extractions.length === 1 ? "finding" : "findings"}` : "Extraction results"}</h2></div>
            <button ref={optimizeButton} className="secondary optimize-button"
              disabled={!matchingResult || !config?.configured || busy !== null || review !== null} onClick={() => void optimize()}>
              {busy === "optimize" ? <LoaderCircle className="spin" size={15} /> : <Sparkles size={15} />}
              {busy === "optimize" ? "Optimizing…" : "Optimize"}
            </button>
            {result && <><span className="microcopy">{exactCount} exact · {result.elapsed_seconds}s · recipe v{result.recipe_version}</span>
              <div className="segmented"><button className={!raw ? "active" : ""} onClick={() => setRaw(false)}>Findings</button><button className={raw ? "active" : ""} onClick={() => setRaw(true)}><Braces size={13} />JSON</button></div>
              <button className="icon-button" aria-label="Export extraction JSON" title="Export extraction JSON"
                onClick={() => downloadJson(result, `${preview?.name || "document"}-extractions.json`)}><Download size={16} /></button>
              <span className="run-settings" aria-label="Extraction run settings">
                {result.options.chunk_size.toLocaleString()} chars · {result.options.max_workers} concurrent · {result.options.extraction_passes} {result.options.extraction_passes === 1 ? "pass" : "passes"} · {result.options.context_window_chars ? `${result.options.context_window_chars.toLocaleString()} context` : "context off"}
              </span></>}
          </div>
          <p className="optimization-hint">{matchingResult
            ? "Optimize uses the applied recipe, all findings and full document text, not unapplied drafts. Review before applying."
            : "Extract with the active recipe on this document first to enable Optimize."}</p>
          {staleResult && <div className="stale-notice">These findings use an earlier recipe. Click Extract to run the active recipe.</div>}
          {busy === "extract" && <div className="processing-notice"><LoaderCircle size={16} className="spin" />Running LangExtract on the complete document. Long documents may take several minutes.</div>}
          {busy === "optimize" && <div className="processing-notice"><LoaderCircle size={16} className="spin" />Analyzing the complete document and all findings. The active recipe stays unchanged until you apply a revision.</div>}
          {!result ? <div className="empty results-empty"><ScanText size={25} /><p>{recipe ? "Select a document and click Extract to discover grounded findings." : "Generate or import a recipe, then run your first extraction."}</p></div>
            : raw ? <pre className="results-json">{JSON.stringify(result, null, 2)}</pre>
              : <div className="findings-list">
                {!findings.length && <p className="microcopy">No matching information found. Review the task, examples and source document.</p>}
                {findings.map(item => <button key={item.id} className={`finding-card ${selectedExtraction === item.id ? "selected" : ""}`}
                  style={{ "--entity-color": classColor(item.extraction_class) } as CSSProperties}
                  onClick={() => setSelectedExtraction(item.id)} aria-pressed={selectedExtraction === item.id}
                  aria-label={`${item.extraction_class}: ${item.extraction_text}, ${item.grounding}, ${item.start === null ? "no source span" : `source ${item.start} to ${item.end}`}`}>
                  <div className="finding-top"><span className="entity-class">{item.extraction_class}</span><span className={`grounding ${item.grounding}`}>{item.grounding}</span>
                    <span className="span-label">{item.start === null ? "No source span" : `${item.start}–${item.end}`}</span></div>
                  <strong>{item.extraction_text}</strong>
                  {Object.keys(item.attributes).length > 0 && <div className="attributes">{Object.entries(item.attributes).map(([key, value]) =>
                    <span key={key}><b>{key}</b> {typeof value === "string" ? value : JSON.stringify(value)}</span>)}</div>}
                </button>)}
              </div>}
        </section>
      </div>
      <RecipePanel recipe={recipe} goal={goal} setGoal={setGoal} busy={modelBusy}
        generating={busy === "generate"} canGenerate={(!!preview || exemplarPaths.length > 0) && !!config?.configured && busy === null && !review}
        exemplarNames={exemplarPaths} previewName={preview?.name}
        onGenerate={() => void generate()} onApply={applyRecipe} onError={setError} />
    </main>
    <footer className="app-footer"><span>DOCUMENT COGNITION <b>/</b> PROTOTYPE</span><span>Grounded by LangExtract. Reviewed by you.</span></footer>
    <ExtractionSettingsDialog open={settingsOpen} options={options} triggerRef={settingsButton}
      onSave={applyOptions} onClose={() => setSettingsOpen(false)} />
    {review && !settingsOpen && <OptimizationReview proposal={review.proposal} original={review.original} triggerRef={optimizeButton}
      onApply={applyOptimization} onDiscard={() => setReview(null)} />}
  </div>;
}
