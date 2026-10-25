import type { CSSProperties, ReactElement } from "react";
import type { FieldValue, SchemaField } from "../types";
import { confidenceLabel, confidenceTone, fieldColor, formatValue, pathKey } from "./ValueHighlighter";

interface Props {
  fields: SchemaField[];
  data: Record<string, unknown> | null;
  values: FieldValue[];
  selected: string | null;
  onSelect: (key: string) => void;
}

const isEmpty = (value: unknown) => value === null || value === undefined || value === ""
  || (Array.isArray(value) && value.length === 0);

/** Nested view of a schema record; each leaf shows confidence and grounding and selects its highlight. */
export default function RecordView({ fields, data, values, selected, onSelect }: Props) {
  const byKey = new Map(values.map(value => [pathKey(value.path), value]));

  function leaf(field: SchemaField, value: unknown, path: (string | number)[], label: string) {
    const key = pathKey(path);
    const meta = byKey.get(key);
    const empty = isEmpty(value);
    return <button type="button" key={key} disabled={empty}
      className={`record-leaf ${selected === key ? "selected" : ""} ${empty ? "empty" : ""}`}
      style={{ "--entity-color": fieldColor(path[0] as string) } as CSSProperties}
      aria-pressed={selected === key} onClick={() => onSelect(key)}
      aria-label={`${label}: ${empty ? "no value" : formatValue(value)}`}>
      <span className="leaf-label">{label}</span>
      <span className="leaf-value">{empty ? "—" : formatValue(value)}</span>
      {!empty && meta && <span className="leaf-meta">
        <span className={`confidence ${confidenceTone(meta.confidence)}`} title="Confidence">{confidenceLabel(meta.confidence)}</span>
        <span className={`grounding ${meta.grounding === "none" ? "ungrounded" : meta.grounding === "fuzzy" ? "approximate" : "exact"}`}
          title={meta.quote ? `Evidence: ${meta.quote}` : "Grounding"}>{meta.grounding === "none" ? "not found" : meta.grounding}</span>
      </span>}
    </button>;
  }

  function node(field: SchemaField, value: unknown, path: (string | number)[]): ReactElement {
    if (field.array) {
      const items = Array.isArray(value) ? value : [];
      return <div className="record-group" key={pathKey(path)}>
        <div className="group-title">{field.name}<span>{items.length} item{items.length === 1 ? "" : "s"}</span></div>
        {items.length === 0 && <div className="record-leaf empty"><span className="leaf-value">—</span></div>}
        {items.map((item, index) => field.type === "object"
          ? <div className="record-item" key={index}><span className="item-index">{index + 1}</span>
            <div>{field.fields.map(child => node(child, (item as Record<string, unknown>)?.[child.name], [...path, index, child.name]))}</div></div>
          : leaf(field, item, [...path, index], `${index + 1}`))}
      </div>;
    }
    if (field.type === "object") {
      const item = (value ?? {}) as Record<string, unknown>;
      return <div className="record-group" key={pathKey(path)}><div className="group-title">{field.name}</div>
        {field.fields.map(child => node(child, item[child.name], [...path, child.name]))}</div>;
    }
    return leaf(field, value, path, field.name);
  }

  if (!data) return <p className="microcopy">No record.</p>;
  return <div className="record-view">{fields.map(field => node(field, data[field.name], [field.name]))}</div>;
}
