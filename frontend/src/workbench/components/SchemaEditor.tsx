import { ArrowDown, ArrowUp, ListPlus, Plus, Trash2 } from "lucide-react";
import type { FieldType, SchemaDraft, SchemaField } from "../types";

const TYPES: { value: FieldType; label: string }[] = [
  { value: "text", label: "Text" }, { value: "verbatim", label: "Verbatim text" }, { value: "date", label: "Date" },
  { value: "number", label: "Number" }, { value: "integer", label: "Integer" }, { value: "boolean", label: "Yes / no" },
  { value: "enum", label: "Choice" }, { value: "object", label: "Object" },
];
const MAX_DEPTH = 3;

export function newField(name = "new_field"): SchemaField {
  return { name, type: "text", description: "", array: false, options: [], fields: [] };
}

/** Client-side mirror of the backend schema rules, reported per field path. */
export function schemaErrors(draft: SchemaDraft): string[] {
  const errors: string[] = [];
  if (!draft.name.trim()) errors.push("Schema name is required.");
  if (!draft.fields.length) errors.push("Add at least one field.");
  function check(fields: SchemaField[], prefix: string, depth: number) {
    const seen = new Set<string>();
    for (const field of fields) {
      const path = prefix ? `${prefix}.${field.name}` : field.name;
      if (!/^[a-z][a-z0-9_]{0,63}$/.test(field.name)) errors.push(`${path || "(unnamed)"}: use lower_snake_case starting with a letter.`);
      if (seen.has(field.name)) errors.push(`${path}: duplicate name.`);
      seen.add(field.name);
      if (field.type === "enum" && !field.options.length) errors.push(`${path}: a choice field needs options.`);
      if (field.type === "object") {
        if (!field.fields.length) errors.push(`${path}: an object needs child fields.`);
        if (depth >= MAX_DEPTH) errors.push(`${path}: nesting is limited to ${MAX_DEPTH} levels.`);
        check(field.fields, path, depth + 1);
      }
    }
  }
  check(draft.fields, "", 1);
  return errors;
}

function normalise(field: SchemaField): SchemaField {
  return { ...field, options: field.type === "enum" ? field.options : [], fields: field.type === "object" ? field.fields : [] };
}

interface FieldsProps { fields: SchemaField[]; depth: number; onChange: (fields: SchemaField[]) => void; parent?: string }

function FieldList({ fields, depth, onChange, parent }: FieldsProps) {
  const update = (index: number, next: SchemaField) => onChange(fields.map((field, i) => i === index ? normalise(next) : field));
  const move = (index: number, delta: number) => {
    const next = [...fields];
    const [item] = next.splice(index, 1);
    next.splice(index + delta, 0, item);
    onChange(next);
  };
  return <div className={`schema-fields depth-${depth}`}>
    {fields.map((field, index) => {
      const label = parent ? `${parent}.${field.name}` : field.name;
      return <div key={index} className="schema-field">
        <div className="schema-row">
          <input className="field-name" aria-label={`Field name ${label}`} value={field.name}
            onChange={event => update(index, { ...field, name: event.target.value.replace(/\s+/g, "_").toLowerCase() })} />
          <select aria-label={`Type of ${label}`} value={field.type}
            onChange={event => {
              const type = event.target.value as FieldType;
              update(index, { ...field, type, fields: type === "object" && !field.fields.length ? [newField("name")] : field.fields });
            }}>
            {TYPES.filter(type => type.value !== "object" || depth < MAX_DEPTH).map(type =>
              <option key={type.value} value={type.value}>{type.label}</option>)}
          </select>
          <label className="list-toggle" title="Repeating values / array of objects">
            <input type="checkbox" checked={field.array} onChange={event => update(index, { ...field, array: event.target.checked })} />List</label>
          <input className="field-description" aria-label={`Description of ${label}`} placeholder="Extraction guidance for this field"
            value={field.description} onChange={event => update(index, { ...field, description: event.target.value })} />
          <span className="row-actions">
            <button type="button" className="icon-button" aria-label={`Move ${label} up`} disabled={index === 0} onClick={() => move(index, -1)}><ArrowUp size={13} /></button>
            <button type="button" className="icon-button" aria-label={`Move ${label} down`} disabled={index === fields.length - 1} onClick={() => move(index, 1)}><ArrowDown size={13} /></button>
            <button type="button" className="icon-button danger" aria-label={`Remove ${label}`} onClick={() => onChange(fields.filter((_, i) => i !== index))}><Trash2 size={13} /></button>
          </span>
        </div>
        {field.type === "enum" && <input className="field-options" aria-label={`Options for ${label}`} placeholder="Comma-separated options"
          value={field.options.join(", ")}
          onChange={event => update(index, { ...field, options: event.target.value.split(",").map(item => item.trim()).filter(Boolean) })} />}
        {field.type === "object" && <div className="schema-children">
          <span className="children-label">{field.array ? "Each item has" : "Contains"}</span>
          <FieldList fields={field.fields} depth={depth + 1} parent={label} onChange={children => update(index, { ...field, fields: children })} />
          <button type="button" className="text-button" onClick={() => update(index, { ...field, fields: [...field.fields, newField()] })}>
            <ListPlus size={13} />Add child field</button>
        </div>}
      </div>;
    })}
  </div>;
}

export default function SchemaEditor({ draft, onChange }: { draft: SchemaDraft; onChange: (draft: SchemaDraft) => void }) {
  return <div className="schema-editor">
    <div className="schema-meta">
      <label><span className="field-label">Schema name</span>
        <input value={draft.name} onChange={event => onChange({ ...draft, name: event.target.value })} /></label>
      <label><span className="field-label">Document type and purpose</span>
        <textarea rows={2} value={draft.description} onChange={event => onChange({ ...draft, description: event.target.value })} /></label>
    </div>
    <div className="schema-header-row" aria-hidden="true"><span>Field</span><span>Type</span><span>List</span><span>Guidance</span><span /></div>
    <FieldList fields={draft.fields} depth={1} onChange={fields => onChange({ ...draft, fields })} />
    <button type="button" className="secondary add-field" onClick={() => onChange({ ...draft, fields: [...draft.fields, newField()] })}>
      <Plus size={14} />Add field</button>
  </div>;
}
