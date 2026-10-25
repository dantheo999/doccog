export type FieldType = "text" | "verbatim" | "date" | "number" | "integer" | "boolean" | "enum" | "object";

export interface SchemaField {
  name: string;
  type: FieldType;
  description: string;
  array: boolean;
  options: string[];
  fields: SchemaField[];
}

export interface SchemaDraft {
  name: string;
  description: string;
  fields: SchemaField[];
}

export interface ExtractionSchema extends SchemaDraft {
  id: string;
  version: number;
  created_at: string;
  exemplars: string[];
}

export interface SchemaSummary { id: string; name: string; version: number; created_at: string; fields: number }

export interface SettingSpec {
  key: string;
  label: string;
  type: "integer" | "number" | "boolean" | "select" | "text";
  default: unknown;
  help: string;
  min: number | null;
  max: number | null;
  options: string[] | null;
}

export interface SolutionStatus { ready: boolean; detail: string; model: string }

export interface SolutionInfo {
  id: string;
  name: string;
  tagline: string;
  description: string;
  technology: string;
  runtime: "cloud" | "local";
  capabilities: string[];
  confidence_method: string;
  route: string;
  accent: string;
  external_url: string | null;
  settings: SettingSpec[];
  status: SolutionStatus;
  /** Optional per-solution schema suggestion (absent on older backends). */
  schema_suggestion?: SchemaSuggestionCapability;
}

/** How (or why not) a solution suggests a schema from exemplar documents. */
export interface SchemaSuggestionCapability {
  supported: boolean;
  /** The method when supported, otherwise the reason it is not. */
  method: string;
  /** Suggestion-only settings, rendered with SettingsForm. */
  settings: SettingSpec[];
}

export interface CollectionDocument {
  id: string;
  path: string;
  name: string;
  characters: number;
  pages: number | null;
  sha256: string;
}

export interface Collection {
  id: string;
  name: string;
  created_at: string;
  updated_at: string;
  documents: CollectionDocument[];
}

export interface StoredDocument {
  id: string;
  name: string;
  text: string;
  characters: number;
  pages: number | null;
  page_starts: number[];
  sha256: string;
}

export type Grounding = "exact" | "fuzzy" | "none";

export interface FieldValue {
  path: (string | number)[];
  field: string;
  value: unknown;
  quote: string | null;
  confidence: number | null;
  grounding: Grounding;
  start: number | null;
  end: number | null;
}

export interface Usage { calls: number; prompt_tokens: number; completion_tokens: number }

export type RunStatus = "queued" | "running" | "done" | "error" | "cancelled";

export interface DocumentResult {
  document_id: string;
  document_name: string;
  document_sha256: string;
  status: RunStatus;
  error: string | null;
  data: Record<string, unknown> | null;
  values: FieldValue[];
  elapsed_seconds: number | null;
  usage: Usage;
  notes: string[];
  raw: unknown;
  started_at: string | null;
  finished_at: string | null;
}

export interface RunSummary {
  id: string;
  solution: string;
  solution_name: string;
  label: string;
  model: string;
  status: RunStatus;
  error: string | null;
  schema_id: string;
  schema_version: number;
  schema_name: string;
  created_at: string;
  started_at: string | null;
  finished_at: string | null;
  documents_total: number;
  documents_done: number;
  documents_failed: number;
  document_ids: string[];
  elapsed_seconds: number;
  values: number;
  grounded: number;
  mean_confidence: number | null;
  usage: Usage;
  confidence_method: string;
  active?: boolean;
}

export interface Run {
  id: string;
  solution: string;
  solution_name: string;
  label: string;
  model: string;
  confidence_method: string;
  status: RunStatus;
  error: string | null;
  extraction_schema: ExtractionSchema;
  settings: Record<string, unknown>;
  created_at: string;
  documents: DocumentResult[];
  summary: RunSummary;
  active: boolean;
}

export interface CompareCellValue {
  value: unknown;
  display: string;
  count: number | null;
  confidence: number | null;
  grounded: number;
  leaves: number;
  grounding: Grounding | null;
  start: number | null;
  end: number | null;
}

export interface CompareRow { cells: Record<string, unknown>; indexes?: Record<string, number>; scores?: Record<string, number> }

export interface CompareCell {
  document_id: string;
  field: string;
  values: Record<string, CompareCellValue>;
  agreement: Record<string, number | null>;
  pivot?: string;
  rows?: CompareRow[];
}

export interface FieldScore { filled: number; documents: number; agreement: number | null; agree_count: number; scored: number }

export interface CompareRunStats extends RunSummary {
  agreement: number | null;
  agree_rate: number | null;
  fill_rate: number | null;
  fields: Record<string, FieldScore>;
}

export interface Comparison {
  reference: string | null;
  runs: CompareRunStats[];
  documents: { id: string; name: string; status: Record<string, RunStatus> }[];
  fields: SchemaField[];
  cells: CompareCell[];
  warnings: string[];
  thresholds: { match: number; pair: number };
}

/** A persisted schema suggestion: the shared, solution-neutral one (solution "shared") or a solution's own. */
export interface SchemaSuggestion {
  id: string;
  solution: string;
  solution_name: string;
  method: string;
  draft: SchemaDraft;
  notes: string[];
  usage: Usage;
  elapsed_seconds: number;
  exemplars: string[];
  document_ids: string[];
  guidance: string;
  settings: Record<string, unknown>;
  created_at: string;
  raw?: unknown;
}

/** Response of POST /api/schemas/suggest (the shared suggestion). id and elapsed_seconds are absent on older backends. */
export interface SharedSuggestionResult {
  draft: SchemaDraft;
  notes: string[];
  usage: Usage;
  exemplars: string[];
  id?: string;
  elapsed_seconds?: number;
}
