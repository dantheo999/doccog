import { useEffect, useMemo, useRef } from "react";
import type { CSSProperties } from "react";
import { buildSegments, classColor } from "../../highlights";
import type { FieldValue } from "../types";

export const pathKey = (path: (string | number)[]) => path.join(".");

/** Colour by top-level field so a record's parts share a hue. */
export const fieldColor = (field: string) => classColor(field.split(/[.[]/)[0]);

interface Props {
  text: string;
  values: FieldValue[];
  selected: string | null;
  onSelect: (key: string) => void;
  pageStarts?: number[];
}

const NO_PAGES: number[] = [];

/** Source text with every grounded value highlighted; clicks cycle overlapping values. */
export default function ValueHighlighter({ text, values, selected, onSelect, pageStarts = NO_PAGES }: Props) {
  const reader = useRef<HTMLDivElement>(null);
  const spans = useMemo(() => values.map(value => ({ id: pathKey(value.path), start: value.start, end: value.end })),
    [values]);
  const byKey = useMemo(() => new Map(values.map(value => [pathKey(value.path), value])), [values]);
  const segments = useMemo(() => buildSegments(text, spans, pageStarts), [text, spans, pageStarts]);
  const pageOffsets = useMemo(() => new Set(pageStarts.slice(1)), [pageStarts]);

  useEffect(() => {
    if (selected === null) return;
    const mark = [...(reader.current?.querySelectorAll<HTMLElement>("[data-ids]") ?? [])]
      .find(node => node.dataset.ids?.split("|").includes(selected));
    mark?.scrollIntoView?.({ behavior: "smooth", block: "center" });
  }, [selected]);

  let offset = 0;
  return <div className="source-text value-source" ref={reader} aria-label="Document source text">
    {segments.map((segment, index) => {
      const start = offset;
      offset += Array.from(segment.text).length;
      const pageBreak = pageOffsets.has(start) ? <span className="page-break" aria-hidden="true">
        page {pageStarts.indexOf(start) + 1}</span> : null;
      if (!segment.ids.length) return <span key={index}>{pageBreak}{segment.text}</span>;
      const selectedIndex = selected === null ? -1 : segment.ids.indexOf(selected);
      const value = byKey.get(selectedIndex >= 0 ? selected! : segment.ids[0])!;
      return <span key={index}>{pageBreak}<button type="button"
        className={`source-mark ${selectedIndex >= 0 ? "selected" : ""} ${value.grounding === "fuzzy" ? "approximate" : ""}`}
        style={{ "--entity-color": fieldColor(value.field) } as CSSProperties}
        data-ids={segment.ids.join("|")} aria-pressed={selectedIndex >= 0}
        title={segment.ids.map(key => byKey.get(key)).map(item => `${item?.field}: ${formatValue(item?.value)}`).join("\n")}
        onClick={() => onSelect(segment.ids[(selectedIndex + 1) % segment.ids.length])}>
        {segment.text}
      </button></span>;
    })}
  </div>;
}

export function formatValue(value: unknown): string {
  if (value === null || value === undefined) return "—";
  if (typeof value === "string") return value;
  return JSON.stringify(value);
}

export function confidenceLabel(confidence: number | null | undefined): string {
  return confidence === null || confidence === undefined ? "n/a" : `${Math.round(confidence * 100)}%`;
}

export function confidenceTone(confidence: number | null | undefined): string {
  if (confidence === null || confidence === undefined) return "none";
  return confidence >= 0.85 ? "high" : confidence >= 0.6 ? "medium" : "low";
}
