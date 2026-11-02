import { useEffect, useMemo, useRef, useState } from "react";
import type { CSSProperties } from "react";
import { AlertCircle, LoaderCircle } from "lucide-react";
import { message, workbench } from "../api";
import type { FieldValue, PageBoxes, PageRect } from "../types";
import { fieldColor, formatValue, pathKey } from "./ValueHighlighter";

interface Props {
  documentId: string;
  values: FieldValue[];
  selected: string | null;
  onSelect: (key: string) => void;
}

const pct = (value: number) => `${(value * 100).toFixed(3)}%`;

/** The PDF's rendered pages with a box over every located value; clicking a box selects the value. */
export default function PdfPages({ documentId, values, selected, onSelect }: Props) {
  const [layout, setLayout] = useState<PageBoxes | null>(null);
  const [error, setError] = useState<string | null>(null);
  const viewer = useRef<HTMLDivElement>(null);
  const spans = useMemo(() => values.filter(value => value.start !== null && value.end !== null && value.end > value.start)
    .map(value => ({ id: pathKey(value.path), start: value.start!, end: value.end! })), [values]);
  const spansKey = JSON.stringify(spans);
  const byKey = useMemo(() => new Map(values.map(value => [pathKey(value.path), value])), [values]);

  useEffect(() => {
    let active = true;
    setLayout(null);
    setError(null);
    workbench.documentBoxes(documentId, JSON.parse(spansKey))
      .then(next => { if (active) setLayout(next); })
      .catch(failure => { if (active) setError(message(failure)); });
    return () => { active = false; };
  }, [documentId, spansKey]);

  const marks = useMemo(() => {
    const perPage = new Map<number, { key: string; rect: PageRect }[]>();
    for (const [key, rects] of Object.entries(layout?.boxes ?? {})) {
      for (const rect of rects) perPage.set(rect.page, [...(perPage.get(rect.page) ?? []), { key, rect }]);
    }
    return perPage;
  }, [layout]);
  const unplaced = layout ? spans.filter(span => !layout.boxes[span.id]?.length).length : 0;

  useEffect(() => {
    if (selected === null || !layout) return;
    viewer.current?.querySelector<HTMLElement>(`[data-id="${CSS.escape(selected)}"]`)
      ?.scrollIntoView?.({ behavior: "smooth", block: "center" });
  }, [selected, layout]);

  if (error) return <div className="notice error pdf-notice" role="alert"><AlertCircle size={15} /><span>{error}</span></div>;
  if (!layout) return <p className="microcopy pdf-notice"><LoaderCircle size={13} className="spin" /> Rendering pages…</p>;
  return <div className="pdf-pages" ref={viewer} aria-label="Document pages">
    {unplaced > 0 && <p className="microcopy pdf-unplaced">{unplaced} located value{unplaced === 1 ? "" : "s"} could not be placed on
      the page (switch to Text to see {unplaced === 1 ? "it" : "them"}).</p>}
    {layout.pages.map((page, index) => <div key={index} className="pdf-page"
      style={{ aspectRatio: `${page.width} / ${page.height}` }}>
      <img src={workbench.pageImageUrl(documentId, index)} alt={`Page ${index + 1}`} loading="lazy" draggable={false} />
      {(marks.get(index) ?? []).map(({ key, rect }, markIndex) => {
        const value = byKey.get(key);
        if (!value) return null;
        return <button type="button" key={`${key}-${markIndex}`} data-id={key}
          className={`pdf-mark ${selected === key ? "selected" : ""} ${value.grounding === "fuzzy" ? "approximate" : ""}`}
          style={{ left: pct(rect.x), top: pct(rect.y), width: pct(rect.w), height: pct(rect.h),
            "--entity-color": fieldColor(value.field) } as CSSProperties}
          aria-pressed={selected === key} aria-label={`${value.field}: ${formatValue(value.value)}`}
          title={`${value.field}: ${formatValue(value.value)}`} onClick={() => onSelect(key)} />;
      })}
      <span className="pdf-page-number" aria-hidden="true">{index + 1}</span>
    </div>)}
  </div>;
}
