import type { SettingSpec } from "../types";

interface Props {
  specs: SettingSpec[];
  values: Record<string, unknown>;
  onChange: (values: Record<string, unknown>) => void;
  disabled?: boolean;
}

export function defaults(specs: SettingSpec[]): Record<string, unknown> {
  return Object.fromEntries(specs.map(spec => [spec.key, spec.default]));
}

/** Whether a stored value has the spec's type (a cleared number field is stored as null, from NaN). */
export function storedValueFits(spec: SettingSpec, value: unknown): boolean {
  switch (spec.type) {
    case "boolean": return typeof value === "boolean";
    case "integer": case "number": return typeof value === "number" && Number.isFinite(value);
    case "select": return typeof value === "string" && (!spec.options || spec.options.includes(value));
    default: return typeof value === "string";
  }
}

export function settingsErrors(specs: SettingSpec[], values: Record<string, unknown>): string[] {
  return specs.flatMap(spec => {
    const value = values[spec.key];
    if (spec.type !== "integer" && spec.type !== "number") return [];
    if (typeof value !== "number" || Number.isNaN(value)) return [`${spec.label}: enter a number.`];
    if (spec.type === "integer" && !Number.isInteger(value)) return [`${spec.label}: enter a whole number.`];
    if ((spec.min !== null && value < spec.min) || (spec.max !== null && value > spec.max)) {
      return [`${spec.label}: must be between ${spec.min} and ${spec.max}.`];
    }
    return [];
  });
}

/** Form generated from a solution's SettingSpec list. */
export default function SettingsForm({ specs, values, onChange, disabled }: Props) {
  const set = (key: string, value: unknown) => onChange({ ...values, [key]: value });
  return <div className="settings-form">
    {specs.map(spec => {
      const id = `setting-${spec.key}`;
      const value = values[spec.key] ?? spec.default;
      let control;
      if (spec.type === "boolean") {
        control = <input id={id} type="checkbox" checked={!!value} disabled={disabled} onChange={event => set(spec.key, event.target.checked)} />;
      } else if (spec.type === "select") {
        control = <select id={id} value={String(value)} disabled={disabled} onChange={event => set(spec.key, event.target.value)}>
          {(spec.options ?? []).map(option => <option key={option} value={option}>{option}</option>)}</select>;
      } else if (spec.type === "text") {
        control = <textarea id={id} rows={2} value={String(value ?? "")} disabled={disabled} onChange={event => set(spec.key, event.target.value)} />;
      } else {
        control = <input id={id} type="number" value={typeof value === "number" && !Number.isNaN(value) ? value : ""}
          min={spec.min ?? undefined} max={spec.max ?? undefined} step={spec.type === "integer" ? 1 : "any"} disabled={disabled}
          onChange={event => set(spec.key, event.target.value === "" ? Number.NaN : Number(event.target.value))} />;
      }
      return <div key={spec.key} className={`setting ${spec.type}`}>
        <label htmlFor={id}>{spec.label}</label>{control}
        {spec.help && <p className="microcopy">{spec.help}</p>}
      </div>;
    })}
  </div>;
}
