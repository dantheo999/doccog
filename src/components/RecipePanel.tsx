import { useEffect, useId, useLayoutEffect, useRef, useState } from "react";
import { Braces, Check, Download, FileText, Maximize2, Pencil, Sparkles, Upload, X } from "lucide-react";
import { downloadJson, errorMessage, formatFewShotExamples, parseFewShotExamples, parseRecipe } from "../recipe";
import type { Recipe } from "../types";
import ExamplePassages from "./ExamplePassages";

interface Props {
  recipe: Recipe | null;
  goal: string;
  setGoal: (goal: string) => void;
  busy: boolean;
  generating: boolean;
  canGenerate: boolean;
  exemplarNames?: string[];
  previewName?: string;
  onGenerate: () => void;
  onApply: (recipe: Recipe) => void;
  onError: (message: string) => void;
}

type RecipeTab = "instructions" | "examples" | "text";

export default function RecipePanel({ recipe, goal, setGoal, busy, generating, canGenerate, exemplarNames = [],
  previewName, onGenerate, onApply, onError }: Props) {
  const [tab, setTab] = useState<RecipeTab>("instructions");
  const [instructions, setInstructions] = useState("");
  const [examples, setExamples] = useState("");
  const [expanded, setExpanded] = useState(false);
  const [editorError, setEditorError] = useState<string | null>(null);
  const importInput = useRef<HTMLInputElement>(null);
  const dialogRef = useRef<HTMLDialogElement>(null);
  const expandTrigger = useRef<RecipeTab | null>(null);
  const expandButtons = useRef<Partial<Record<RecipeTab, HTMLButtonElement | null>>>({});
  const dialogTitle = useId();
  const errorRef = useRef<HTMLDivElement>(null);
  const busyRef = useRef(busy);
  const importEpoch = useRef(0);
  busyRef.current = busy;
  // Cancel before reads can resume after a committed recipe/conflict lifecycle change.
  useLayoutEffect(() => {
    importEpoch.current++;
    if (importInput.current) importInput.current.value = "";
    return () => { importEpoch.current++; };
  }, [recipe]);
  useLayoutEffect(() => {
    if (busy) {
      importEpoch.current++;
      if (importInput.current) importInput.current.value = "";
    }
  }, [busy]);
  useEffect(() => {
    setInstructions(recipe?.instructions ?? "");
    setExamples(recipe ? formatFewShotExamples(recipe.examples) : "");
    setEditorError(null);
  }, [recipe]);
  useEffect(() => {
    const dialog = dialogRef.current;
    if (!dialog) return;
    if (expanded) {
      if (!dialog.open) dialog.showModal();
      dialog.querySelector<HTMLButtonElement>(".close-full-page")?.focus();
    } else {
      if (dialog.open) dialog.close();
      if (expandTrigger.current) expandButtons.current[expandTrigger.current]?.focus();
    }
  }, [expanded]);
  useEffect(() => { if (editorError) errorRef.current?.focus(); }, [editorError]);
  const dirty = recipe !== null && (instructions !== recipe.instructions || examples !== formatFewShotExamples(recipe.examples));

  function applyEdits() {
    if (!recipe) return;
    try {
      const validated = parseRecipe(JSON.stringify({
        ...recipe, version: recipe.version + 1,
        instructions, examples: parseFewShotExamples(examples, recipe.examples),
      }), recipe);
      setEditorError(null);
      onApply(validated);
    } catch (error) {
      const message = errorMessage(error);
      setEditorError(message);
      errorRef.current?.focus();
      onError(message);
    }
  }

  async function importRecipe(file: File | undefined) {
    if (!file || busyRef.current) return;
    const epoch = ++importEpoch.current;
    try {
      if (file.size > 256_000) throw new Error("Recipe JSON must be smaller than 256 KB.");
      const text = await file.text();
      if (epoch !== importEpoch.current || busyRef.current) return;
      const imported = parseRecipe(text, "import");
      onApply({ ...imported, id: crypto.randomUUID(), version: 1 });
      setGoal(imported.goal);
    } catch (error) {
      if (epoch !== importEpoch.current || busyRef.current) return;
      const message = errorMessage(error);
      setEditorError(message);
      onError(message);
    }
    finally {
      if (epoch === importEpoch.current && importInput.current) importInput.current.value = "";
    }
  }

  const recipeTabs = [
    { key: "instructions", label: "Instructions", Icon: Pencil },
    { key: "examples", label: "Few-shot JSON", Icon: Braces },
    { key: "text", label: "Example text", Icon: FileText },
  ] as const;

  const editor = <div className="recipe-editor">
    <div className="tabs" role="tablist" aria-label="Recipe playground">
      {recipeTabs.map(({ key, label, Icon }) => <div className="recipe-tab" key={key}>
        <button role="tab" aria-selected={tab === key} onClick={() => setTab(key)}><Icon size={14} />{label}</button>
        {!expanded && <button className="expand-tab" aria-label={`Expand ${label}`} title={`Expand ${label}`}
          ref={node => { expandButtons.current[key] = node; }}
          disabled={!recipe || busy} onClick={() => {
            expandTrigger.current = key;
            setTab(key);
            setExpanded(true);
          }}><Maximize2 size={12} /></button>}
      </div>)}
    </div>
    {editorError && <div ref={errorRef} className="notice error recipe-editor-error" role="alert" tabIndex={-1}>
      <span>{editorError}</span>
    </div>}
    <div className="playground">
      {!recipe ? <div className="empty recipe-empty"><Braces size={34} /><h3>Your extraction recipe lives here</h3>
        <p>Inspect and refine the instructions and example JSON before running on other documents.</p></div> : <>
        <div hidden={tab !== "instructions"} className="editor-pane">
          <span className="editor-caption">Generic extraction instructions <span>EDITABLE</span></span>
          <textarea aria-label="Generic extraction instructions" className="code-editor instructions-editor"
            disabled={busy} value={instructions} onChange={event => setInstructions(event.target.value)} spellCheck={false} />
        </div>
        <div hidden={tab !== "examples"} className="editor-pane">
          <span className="editor-caption">Annotations only · one entry per exemplar, in source order <span>EDITABLE</span></span>
          <textarea aria-label="Few-shot extraction JSON" className="code-editor" disabled={busy}
            value={examples} onChange={event => setExamples(event.target.value)} spellCheck={false} />
        </div>
        <div hidden={tab !== "text"} className="example-text-pane">
          <ExamplePassages examples={recipe.examples} />
        </div>
      </>}
    </div>
    <div className="recipe-actions">
      <button className="secondary" disabled={!dirty || busy} onClick={applyEdits}><Check size={15} />Apply edits</button>
      <button className="icon-button" title="Export recipe JSON" aria-label="Export recipe JSON"
        disabled={!recipe || busy} onClick={() => recipe && downloadJson(recipe, "extraction-recipe.json")}><Download size={17} /></button>
      <button className="icon-button" title="Import recipe JSON" aria-label="Import recipe JSON" disabled={busy}
        onClick={() => importInput.current?.click()}><Upload size={17} /></button>
      <input type="file" ref={importInput} accept=".json,application/json" hidden
        aria-label="Recipe JSON file" disabled={busy} onChange={event => { if (!busy) void importRecipe(event.target.files?.[0]); }} />
    </div>
    <div className={`recipe-footer ${dirty ? "dirty" : ""}`}>
      {dirty ? "Unapplied edits · Extract still uses the active recipe." : "Recipe stays active when you switch documents. Saved in this browser."}
    </div>
  </div>;

  return <aside className="recipe-panel panel">
    <div className="panel-header"><div className="section-icon violet"><Sparkles size={18} /></div>
      <div><h2>Teach the extractor</h2><p>One recipe. Any document.</p></div>
      <span className={`pill ${recipe ? "success" : ""}`}>{recipe ? "Recipe ready" : "No recipe"}</span>
    </div>
    <div className="task-section">
      <label className="field-label" htmlFor="goal">What do you want to extract?</label>
      <textarea id="goal" className="goal-input" value={goal} maxLength={4_000}
        disabled={busy} onChange={event => setGoal(event.target.value)}
        placeholder="e.g. Find privacy obligations, named organisations, applicable regulations and retention periods. Include the responsible party for each obligation." />
      <button className="primary generate-button" disabled={!canGenerate || !goal.trim() || busy} onClick={onGenerate}>
        <Sparkles size={16} />{generating ? "Generating recipe…" : "Generate example"}
      </button>
      <p className="microcopy exemplar-selection" aria-live="polite">
        {exemplarNames.length ? `${exemplarNames.length} exemplar document${exemplarNames.length === 1 ? "" : "s"}: ${exemplarNames.join(", ")}`
          : previewName ? `No exemplars checked · uses current preview: ${previewName}` : "Check exemplar documents or choose a preview to generate."}
      </p>
      <p className="microcopy">Creates generic instructions and one whole-document example per checked source, with representative annotations rather than every match.
        Complete examples are sent with each extraction chunk, increasing token usage and processing time.</p>
      <p className="microcopy">Unique whitespace-only quote differences are restored from the source. Changed words, punctuation or ambiguous quotes are rejected.</p>
      <p className="microcopy excerpt-privacy">Whole exemplar texts persist in this browser and recipe JSON exports. Review sensitive source content before sharing.</p>
    </div>
    <div className="recipe-meta">
      <span className="eyebrow">ACTIVE RECIPE</span>
      {recipe ? <><span>v{recipe.version} · from {recipe.source_name}</span>
        <p title={recipe.goal}>{recipe.goal}</p>
        {recipe.source_characters_total > recipe.source_characters_used && recipe.source_characters_used > 0 &&
          <p className="sampling-note">This older recipe used {recipe.source_characters_used.toLocaleString()} of {recipe.source_characters_total.toLocaleString()} characters. Regenerate to use whole-document examples.</p>}
      </> : <p>Select a document and describe your task to get started, or import a saved recipe.</p>}
    </div>
    {!expanded && editor}
    <dialog ref={dialogRef} className="recipe-full-page" aria-labelledby={dialogTitle}
      onCancel={event => { event.preventDefault(); setExpanded(false); }}
      onClose={() => setExpanded(false)}>
      {expanded && <>
        <div className="full-page-header">
          <div className="section-icon violet"><Sparkles size={20} /></div>
          <div><h2 id={dialogTitle}>Active recipe</h2>
            <p>{recipe ? `v${recipe.version} · ${recipe.source_name}` : "Recipe playground"}</p></div>
          <button className="secondary close-full-page" onClick={() => setExpanded(false)}>
            <X size={17} />Close full-page view
          </button>
        </div>
        {editor}
      </>}
    </dialog>
  </aside>;
}
