import { useEffect, useId, useRef, useState } from "react";
import type { RefObject } from "react";
import { Check, RotateCcw, Settings, X } from "lucide-react";
import { DEFAULT_EXTRACTION_OPTIONS, EXTRACTION_OPTION_FIELDS, parseExtractionOptions } from "../extractionOptions";
import type { ExtractionOptions } from "../extractionOptions";
import { errorMessage } from "../recipe";

interface Props {
  open: boolean;
  options: ExtractionOptions;
  triggerRef: RefObject<HTMLButtonElement | null>;
  onSave: (options: ExtractionOptions) => void;
  onClose: () => void;
}

function draftFrom(options: ExtractionOptions) {
  return {
    chunk_size: String(options.chunk_size), max_workers: String(options.max_workers),
    extraction_passes: String(options.extraction_passes), context_window_chars: String(options.context_window_chars),
  };
}

export default function ExtractionSettingsDialog({ open, options, triggerRef, onSave, onClose }: Props) {
  const dialogRef = useRef<HTMLDialogElement>(null);
  const errorRef = useRef<HTMLDivElement>(null);
  const wasOpen = useRef(false);
  const id = useId();
  const [draft, setDraft] = useState(() => draftFrom(options));
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (open) {
      setDraft(draftFrom(options));
      setError(null);
    }
  }, [open, options]);

  useEffect(() => {
    const dialog = dialogRef.current;
    if (!dialog) return;
    if (open) {
      if (!dialog.open) dialog.showModal();
      dialog.querySelector<HTMLInputElement>("input")?.focus();
      wasOpen.current = true;
    } else if (wasOpen.current) {
      if (dialog.open) dialog.close();
      triggerRef.current?.focus();
      wasOpen.current = false;
    }
  }, [open, triggerRef]);

  useEffect(() => {
    if (error) errorRef.current?.focus();
  }, [error]);

  function save() {
    let validated: ExtractionOptions;
    try {
      const number = (value: string) => value.trim() === "" ? NaN : Number(value);
      validated = parseExtractionOptions({
        chunk_size: number(draft.chunk_size), max_workers: number(draft.max_workers),
        extraction_passes: number(draft.extraction_passes), context_window_chars: number(draft.context_window_chars),
      });
    } catch (error) {
      setError(errorMessage(error));
      errorRef.current?.focus();
      return;
    }
    onSave(validated);
    onClose();
  }

  return <dialog ref={dialogRef} className="settings-dialog" aria-labelledby={`${id}-title`}
    onCancel={event => { event.preventDefault(); onClose(); }} onClose={onClose}>
    <div className="settings-header">
      <div className="section-icon violet"><Settings size={19} /></div>
      <div><h2 id={`${id}-title`}>Extraction settings</h2><p>Processing options, separate from your recipe.</p></div>
      <button className="icon-button" aria-label="Close settings" onClick={onClose}><X size={18} /></button>
    </div>
    <form noValidate onSubmit={event => { event.preventDefault(); save(); }}>
      <div className="settings-content">
        <p className="settings-intro">Saved in this browser. Changes affect your next extraction, not a run already in progress.
          Faster processing is not guaranteed; smaller chunks and extra passes can increase token usage and costs.</p>
        {error && <div ref={errorRef} className="notice error" role="alert" tabIndex={-1}><span>{error}</span></div>}
        <div className="settings-fields">
          {EXTRACTION_OPTION_FIELDS.map(({ key, label, min, max, help }) => <div className="settings-field" key={key}>
            <label htmlFor={`${id}-${key}`}>{label}</label>
            <input id={`${id}-${key}`} type="number" min={min} max={max} step={1} required
              aria-describedby={`${id}-${key}-help`} value={draft[key]}
              onChange={event => setDraft(current => ({ ...current, [key]: event.target.value }))} />
            <p id={`${id}-${key}-help`}>{help}<span>Whole numbers: {min.toLocaleString()} to {max.toLocaleString()}.</span></p>
          </div>)}
        </div>
      </div>
      <div className="settings-actions">
        <button type="button" className="secondary reset-settings" onClick={() => {
          setDraft(draftFrom(DEFAULT_EXTRACTION_OPTIONS));
          setError(null);
        }}><RotateCcw size={15} />Reset to defaults</button>
        <button type="button" className="secondary" onClick={onClose}>Cancel</button>
        <button type="submit" className="primary"><Check size={16} />Save settings</button>
      </div>
    </form>
  </dialog>;
}
