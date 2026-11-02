import { useState } from "react";
import { FileImage, FileText } from "lucide-react";
import type { FieldValue, StoredDocument } from "../types";
import ValueHighlighter from "./ValueHighlighter";
import PdfPages from "./PdfPages";

const VIEW_KEY = "doc-cognition.source-view";
type View = "text" | "pdf";

function storedView(): View {
  try { return localStorage.getItem(VIEW_KEY) === "text" ? "text" : "pdf"; } catch { return "pdf"; }
}

interface Props {
  document: StoredDocument;
  values: FieldValue[];
  selected: string | null;
  onSelect: (key: string) => void;
}

/** A document's source with its values highlighted: the extracted text, or (for PDFs) the pages themselves. */
export default function SourceView({ document, values, selected, onSelect }: Props) {
  const [view, setView] = useState<View>(storedView);
  const isPdf = !!document.pages && document.name.toLowerCase().endsWith(".pdf");
  const shown: View = isPdf ? view : "text";

  function choose(next: View) {
    setView(next);
    try { localStorage.setItem(VIEW_KEY, next); } catch { /* storage unavailable */ }
  }

  return <div className="source-view">
    {isPdf && <div className="segmented source-view-toggle" role="group" aria-label="Source view">
      <button type="button" className={shown === "pdf" ? "active" : ""} aria-pressed={shown === "pdf"} onClick={() => choose("pdf")}>
        <FileImage size={12} />PDF pages</button>
      <button type="button" className={shown === "text" ? "active" : ""} aria-pressed={shown === "text"} onClick={() => choose("text")}>
        <FileText size={12} />Text</button>
    </div>}
    {shown === "pdf"
      ? <PdfPages documentId={document.id} values={values} selected={selected} onSelect={onSelect} />
      : <ValueHighlighter text={document.text} values={values} selected={selected} onSelect={onSelect}
        pageStarts={document.page_starts} />}
  </div>;
}
