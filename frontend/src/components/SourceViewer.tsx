import { useEffect, useMemo, useRef } from "react";
import type { CSSProperties } from "react";
import { buildSegments, classColor } from "../highlights";
import type { Extraction } from "../types";

interface Props {
  text: string;
  extractions: Extraction[];
  selected: string | null;
  onSelect: (id: string) => void;
}

export default function SourceViewer({ text, extractions, selected, onSelect }: Props) {
  const reader = useRef<HTMLDivElement>(null);
  const segments = useMemo(() => buildSegments(text, extractions), [text, extractions]);
  const items = useMemo(() => new Map(extractions.map(item => [item.id, item])), [extractions]);

  useEffect(() => {
    if (selected === null) return;
    const mark = [...(reader.current?.querySelectorAll<HTMLButtonElement>("[data-ids]") ?? [])]
      .find(node => node.dataset.ids?.split(",").includes(selected));
    mark?.scrollIntoView?.({ behavior: "smooth", block: "center" });
  }, [selected]);

  return <div className="source-text" ref={reader} aria-label="Document source text">
    {segments.map((segment, index) => {
      if (!segment.ids.length) return <span key={index}>{segment.text}</span>;
      const selectedIndex = selected === null ? -1 : segment.ids.indexOf(selected);
      const id = selectedIndex >= 0 ? selected! : segment.ids[0];
      const item = items.get(id)!;
      return <button key={index} type="button" className={`source-mark ${selectedIndex >= 0 ? "selected" : ""} ${item.grounding}`}
        style={{ "--entity-color": classColor(item.extraction_class) } as CSSProperties}
        data-ids={segment.ids.join(",")} aria-pressed={selectedIndex >= 0}
        title={segment.ids.map(key => `${items.get(key)?.extraction_class}: ${items.get(key)?.grounding}`).join(" · ")}
        onClick={() => onSelect(segment.ids[(selectedIndex + 1) % segment.ids.length])}>
        {segment.text}
      </button>;
    })}
  </div>;
}
