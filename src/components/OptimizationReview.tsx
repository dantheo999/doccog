import { useEffect, useId, useRef, useState } from "react";
import type { RefObject } from "react";
import { Sparkles, X } from "lucide-react";
import { errorMessage, formatFewShotExamples } from "../recipe";
import type { OptimizationProposal, Recipe } from "../types";
import ExamplePassages from "./ExamplePassages";

interface Props {
  proposal: OptimizationProposal;
  original: Recipe;
  triggerRef: RefObject<HTMLButtonElement | null>;
  onApply: () => void;
  onDiscard: () => void;
}

export default function OptimizationReview({ proposal, original, triggerRef, onApply, onDiscard }: Props) {
  const dialogRef = useRef<HTMLDialogElement>(null);
  const errorRef = useRef<HTMLDivElement>(null);
  const titleId = useId();
  const [tab, setTab] = useState<"instructions" | "examples" | "text">("instructions");
  const [error, setError] = useState<string | null>(null);
  const tabs = [["instructions", "Instructions"], ["examples", "Few-shot JSON"], ["text", "Example text"]] as const;
  useEffect(() => {
    const dialog = dialogRef.current;
    if (dialog && !dialog.open) dialog.showModal();
    dialog?.querySelector<HTMLButtonElement>(".close-review")?.focus();
    const trigger = triggerRef.current;
    return () => {
      // DOM removal leaves the native top layer; close() here would discard on StrictMode's effect replay.
      trigger?.focus();
    };
  }, [triggerRef]);
  useEffect(() => { if (error) errorRef.current?.focus(); }, [error]);

  return <dialog ref={dialogRef} className="recipe-full-page optimization-review" aria-labelledby={titleId}
    onCancel={event => { event.preventDefault(); onDiscard(); }} onClose={onDiscard}>
    <div className="full-page-header">
      <div className="section-icon violet"><Sparkles size={20} /></div>
      <div><h2 id={titleId}>Review optimization</h2><p>Proposed v{proposal.candidate.version} · not yet applied</p></div>
      <button className="secondary close-full-page close-review" onClick={onDiscard}><X size={17} />Close review</button>
    </div>
    <div className="optimization-summary">
      <p>Optimize analyzes the applied recipe and the complete extraction, not editor drafts. Apply revision replaces any unapplied recipe drafts.
        All original exemplar texts and their order are retained; only verified exemplar annotations are optimized.
        Imported or legacy unverified examples stay unchanged.
        Findings and example annotations are not ground truth; this proposal is not a measured improvement.</p>
      <h3>Reasons for changes</h3>
      <ul>{proposal.reasons.map((reason, index) => <li key={index}>{reason}</li>)}</ul>
    </div>
    {error && <div ref={errorRef} className="notice error recipe-editor-error" role="alert" tabIndex={-1}>{error}</div>}
    <div className="tabs" role="tablist" aria-label="Optimization comparison">
      {tabs.map(([key, label], index) => <button key={key} id={`${titleId}-${key}-tab`} role="tab" aria-selected={tab === key}
        aria-controls={`${titleId}-panel`} tabIndex={tab === key ? 0 : -1} onClick={() => setTab(key)}
        onKeyDown={event => {
          if (!["ArrowLeft", "ArrowRight", "Home", "End"].includes(event.key)) return;
          event.preventDefault();
          const next = event.key === "Home" ? 0 : event.key === "End" ? tabs.length - 1
            : (index + (event.key === "ArrowRight" ? 1 : tabs.length - 1)) % tabs.length;
          setTab(tabs[next][0]);
          event.currentTarget.parentElement?.querySelectorAll<HTMLButtonElement>('[role="tab"]')[next]?.focus();
        }}>{label}</button>)}
    </div>
    <div className="optimization-comparison" role="tabpanel" id={`${titleId}-panel`} aria-labelledby={`${titleId}-${tab}-tab`}>
      {([{ label: "Before", recipe: original }, { label: "After", recipe: proposal.candidate }]).map(side =>
        <section key={side.label} className="comparison-side" aria-label={side.label}>
          <h3>{side.label} · recipe v{side.recipe.version} · {side.recipe.examples.length} {side.recipe.examples.length === 1 ? "example" : "examples"}</h3>
          <div className="comparison-content">
            {tab === "instructions" ? <pre>{side.recipe.instructions}</pre>
              : tab === "examples" ? <pre>{formatFewShotExamples(side.recipe.examples)}</pre>
                : <ExamplePassages examples={side.recipe.examples} />}
          </div>
        </section>)}
    </div>
    <div className="optimization-actions">
      <p>Applying keeps the current findings as an earlier run. Click Extract yourself to evaluate the revision.</p>
      <button className="secondary" onClick={onDiscard}>Discard</button>
      <button className="primary" onClick={() => {
        try { onApply(); } catch (error) { setError(errorMessage(error)); }
      }}>Apply revision</button>
    </div>
  </dialog>;
}
